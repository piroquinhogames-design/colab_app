"""Illustrious LoRA Studio — servidor para execução integral no Google Colab.

Variáveis obrigatórias: STUDIO_PASSWORD, MEGA_EMAIL, MEGA_PASSWORD.
Variáveis opcionais: CIVITAI_TOKEN, MODEL_URL, MODEL_PATH, MEGA_FOLDER,
STUDIO_SECRET e PORT.
"""

from __future__ import annotations

import base64
import asyncio
import io
import functools
import hmac
import json
import os
import queue
import re
import shutil
import threading
import time
import types
import uuid
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from flask import Flask, jsonify, redirect, request, send_file, session, url_for

# mega.py ainda depende do decorador removido no Python 3.12. A adaptação é
# aplicada antes de a cadeia de importação do cliente MEGA carregar tenacity.
if not hasattr(asyncio, "coroutine"):
    asyncio.coroutine = types.coroutine  # type: ignore[attr-defined]

from mega import Mega
from PIL import Image


ROOT = Path(os.environ.get("STUDIO_ROOT", "/content/illustrious-studio")).resolve()
MODELS = ROOT / "models"
LORAS = ROOT / "loras"
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "uploads"
for directory in (MODELS, LORAS, OUTPUTS, UPLOADS):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_LORAS = 3
MODEL_URL = os.environ.get(
    "MODEL_URL", "https://civitai.com/api/download/models/2883731"
)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS / "waiIllustriousSDXL_v170.safetensors"))
MEGA_FOLDER = os.environ.get("MEGA_FOLDER", "IllustriousStudio")
CIVITAI_BASE = "https://civitai.com/api/v1"
LAST_SETTINGS_NAME = "last_settings.json"

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config.update(
    SECRET_KEY=os.environ.get("STUDIO_SECRET") or os.urandom(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def secure_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return value[:180] or "file"


def parse_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def civitai_headers() -> dict[str, str]:
    token = os.environ.get("CIVITAI_TOKEN", "").strip()
    headers = {"User-Agent": "Illustrious-LoRA-Studio/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def authentication_required(view: Callable):
    @functools.wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        if not session.get("authenticated"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Não autenticado."}), 401
            return redirect(url_for("index"))
        return view(*args, **kwargs)

    return wrapped


def csrf_required(view: Callable):
    @functools.wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        token = request.headers.get("X-CSRF-Token", "")
        if not token or not hmac.compare_digest(token, session.get("csrf", "")):
            return jsonify({"error": "Token de segurança inválido. Atualize a página e tente novamente."}), 403
        return view(*args, **kwargs)

    return wrapped


@dataclass
class LoRASelection:
    version_id: int
    model_id: int | None
    name: str
    weight: float


@dataclass
class GenerationParams:
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance: float
    width: int
    height: int
    strength: float
    mode: str
    loras: list[LoRASelection] = field(default_factory=list)
    source_image: str | None = None
    edit_level: str = "medium"


def saved_settings(params: GenerationParams) -> dict[str, Any]:
    """Campos reutilizáveis pelo painel; caminhos temporários nunca são arquivados."""
    return {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "seed": params.seed,
        "steps": params.steps,
        "guidance": params.guidance,
        "width": params.width,
        "height": params.height,
        "strength": params.strength,
        "mode": params.mode,
        "edit_level": params.edit_level,
        "loras": [asdict(item) for item in params.loras],
    }


@dataclass
class Job:
    id: str
    created_at: str
    status: str
    progress: int
    params: GenerationParams
    updated_at: str | None = None
    completed_at: str | None = None
    filename: str | None = None
    mega_synced: bool = False
    error: str | None = None
    vram_gb: float | None = None

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["params"]["loras"] = [asdict(item) for item in self.params.loras]
        data["image_url"] = f"/api/history/{self.id}/image" if self.filename else None
        return data


class MegaArchive:
    """Adaptador mínimo que mantém imagens e metadados no mesmo diretório MEGA."""

    def __init__(self) -> None:
        self.client = None
        self.folder = None
        self.available = False
        self.error: str | None = None
        self.lock = threading.Lock()

    def connect(self) -> None:
        email = os.environ.get("MEGA_EMAIL", "").strip()
        password = os.environ.get("MEGA_PASSWORD", "")
        if not email or not password:
            self.error = "Configure MEGA_EMAIL e MEGA_PASSWORD para ativar o arquivo persistente."
            return
        try:
            self.client = Mega().login(email, password)
            found = self.client.find(MEGA_FOLDER)
            self.folder = self._first_node(found)
            if not self.folder:
                self.client.create_folder(MEGA_FOLDER)
                # mega.py retorna um dicionário no create_folder(), mas upload()
                # exige o nó remoto (o primeiro item de find()).
                self.folder = self._first_node(self.client.find(MEGA_FOLDER))
            if not self.folder:
                raise RuntimeError(f"A pasta MEGA {MEGA_FOLDER!r} foi criada, mas seu nó não foi localizado")
            self.available = True
            self.error = None
        except Exception as exc:  # credenciais e rede não devem derrubar o servidor
            self.available = False
            self.error = f"Não foi possível conectar ao MEGA: {str(exc)[:180]}"

    @staticmethod
    def _first_node(value: Any) -> Any:
        """Obtém o identificador do primeiro resultado para operações de pasta/upload."""
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, tuple) and len(value) == 2:
            return value[0]
        return value

    @staticmethod
    def _download_node(value: Any) -> Any:
        """Preserva o par ``(handle, atributos)`` exigido por mega.py.download()."""
        if isinstance(value, list):
            return value[0] if value else None
        if isinstance(value, tuple) and len(value) == 2:
            return value
        if isinstance(value, dict) and isinstance(value.get("a"), dict):
            handle = value.get("h")
            return (handle, value) if handle else value
        return value

    @staticmethod
    def _node_name(value: Any) -> str:
        """Lê o nome tanto de um nó mega.py quanto do formato dos testes."""
        node = value[1] if isinstance(value, tuple) and len(value) == 2 else value
        if not isinstance(node, dict):
            return ""
        attributes = node.get("a")
        return attributes.get("n", "") if isinstance(attributes, dict) else ""

    def _upload(self, path: Path) -> None:
        if not self.available or not self.client or not self.folder:
            raise RuntimeError("MEGA não está conectado a uma pasta de destino válida")
        with self.lock:
            existing = self._first_node(self.client.find(path.name))
            if existing:
                try:
                    self.client.destroy(existing)
                except Exception:
                    pass
            uploaded = self.client.upload(str(path), self.folder)
            if uploaded is None:
                raise RuntimeError(f"O cliente MEGA não confirmou o upload de {path.name}")

    def save_job(self, job: Job, image_path: Path | None) -> bool:
        if not self.available:
            return False
        metadata_path = OUTPUTS / f"{job.id}.json"
        try:
            # O estado só é confirmado depois que imagem e manifesto forem
            # aceitos pelo cliente MEGA. O primeiro manifesto é provisório;
            # o segundo registra a confirmação para restauração futura.
            job.mega_synced = False
            if image_path and image_path.exists():
                self._upload(image_path)
            metadata_path.write_text(json.dumps(job.public(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._upload(metadata_path)
            job.mega_synced = True
            metadata_path.write_text(json.dumps(job.public(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._upload(metadata_path)
            return True
        except Exception as exc:
            job.mega_synced = False
            self.error = f"Falha ao enviar ao MEGA: {str(exc)[:180]}"
            return False

    def save_last_settings(self, settings: dict[str, Any]) -> bool:
        """Substitui o manifesto único do último sinal renderizado no arquivo MEGA."""
        settings_path = OUTPUTS / LAST_SETTINGS_NAME
        payload = {"updated_at": now_iso(), "settings": settings}
        settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.available:
            return False
        try:
            self._upload(settings_path)
            return True
        except Exception as exc:
            self.error = f"Falha ao salvar preferências no MEGA: {str(exc)[:180]}"
            return False

    def load_last_settings(self) -> dict[str, Any] | None:
        """Recupera o manifesto único de preferências sem misturá-lo ao histórico de jobs."""
        if not self.available or not self.client:
            return None
        cache = ROOT / "mega-cache"
        cache.mkdir(exist_ok=True)
        try:
            node = self._download_node(self.client.find(LAST_SETTINGS_NAME))
            if not node:
                return None
            downloaded = self.client.download(node, str(cache))
            local = Path(downloaded) if downloaded else cache / LAST_SETTINGS_NAME
            if not local.exists():
                return None
            payload = json.loads(local.read_text(encoding="utf-8"))
            settings = payload.get("settings") if isinstance(payload, dict) else None
            return settings if isinstance(settings, dict) else None
        except Exception as exc:
            self.error = f"Falha ao recuperar preferências do MEGA: {str(exc)[:180]}"
            return None

    @staticmethod
    def _file_nodes(value: Any, handle: str | None = None) -> list[Any]:
        """Extrai nós de arquivos de mapas planos, árvores e tuplas do mega.py."""
        found: list[Any] = []
        if isinstance(value, dict):
            attributes = value.get("a")
            if isinstance(attributes, dict) and isinstance(attributes.get("n"), str):
                node_handle = value.get("h") or handle
                found.append((node_handle, value) if node_handle else value)
            else:
                for key, child in value.items():
                    found.extend(MegaArchive._file_nodes(child, str(key)))
        elif isinstance(value, (list, tuple)):
            for child in value:
                found.extend(MegaArchive._file_nodes(child, handle))
        return found

    def sync_last_settings(self) -> bool:
        """Reenvia o manifesto local quando o upload original ficou pendente."""
        if not self.available or not self.client:
            return False
        settings_path = OUTPUTS / LAST_SETTINGS_NAME
        if not settings_path.exists():
            return False
        try:
            self._upload(settings_path)
            return True
        except Exception as exc:
            self.error = f"Falha ao reenviar preferências ao MEGA: {str(exc)[:180]}"
            return False

    def list_remote_metadata(self) -> list[dict[str, Any]]:
        if not self.available or not self.client:
            return []
        cache = ROOT / "mega-cache"
        cache.mkdir(exist_ok=True)
        jobs: list[dict[str, Any]] = []
        try:
            files = self.client.get_files()
            for node in self._file_nodes(files):
                name = self._node_name(node)
                if name == LAST_SETTINGS_NAME or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}\.json", name, re.I):
                    continue
                try:
                    downloaded = self.client.download(node, str(cache))
                    local = Path(downloaded) if downloaded else cache / name
                    if local.exists():
                        payload = json.loads(local.read_text(encoding="utf-8"))
                        if isinstance(payload, dict) and payload.get("id"):
                            jobs.append(payload)
                except Exception:
                    continue
        except Exception as exc:
            self.error = f"Falha ao ler o histórico MEGA: {str(exc)[:180]}"
        return jobs

    def restore_image(self, job_id: str) -> Path | None:
        destination = OUTPUTS / f"{job_id}.png"
        if destination.exists():
            return destination
        if not self.available or not self.client:
            return None
        try:
            node = self._download_node(self.client.find(destination.name))
            if node:
                downloaded = self.client.download(node, str(OUTPUTS))
                result = Path(downloaded) if downloaded else destination
                return result if result.exists() else None
        except Exception as exc:
            self.error = f"Falha ao recuperar imagem do MEGA: {str(exc)[:180]}"
        return None


class GeneratorEngine:
    """Carrega o checkpoint uma única vez e aplica LoRAs somente durante o job ativo."""

    def __init__(self) -> None:
        self.pipe = None
        self.img_pipe = None
        self.device = "cuda"
        self.load_lock = threading.Lock()

    def ensure_checkpoint(self) -> None:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 500 * 1024 * 1024:
            return
        target = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
        target.unlink(missing_ok=True)
        with requests.get(MODEL_URL, headers=civitai_headers(), stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if target.stat().st_size < 500 * 1024 * 1024:
            target.unlink(missing_ok=True)
            raise RuntimeError("O checkpoint baixado parece incompleto; tente novamente.")
        target.replace(MODEL_PATH)

    @staticmethod
    def _configure_memory_savers(pipeline: Any) -> None:
        """Configura economia de VRAM sem usar os atalhos obsoletos do Diffusers."""
        pipeline.enable_model_cpu_offload()
        if getattr(pipeline, "vae", None) is not None:
            pipeline.vae.enable_slicing()

    def _load_pipeline(self) -> None:
        if self.pipe is not None:
            return
        self.ensure_checkpoint()
        import torch
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Nenhuma GPU CUDA foi detectada. Ative uma sessão T4 no Colab e reinicie o servidor.")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            str(MODEL_PATH), torch_dtype=torch.float16, use_safetensors=True
        )
        self._configure_memory_savers(self.pipe)
        self.img_pipe = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
        self._configure_memory_savers(self.img_pipe)

    def _download_lora(self, version_id: int) -> Path:
        destination = LORAS / f"civitai_{version_id}.safetensors"
        if destination.exists() and destination.stat().st_size > 1024 * 1024:
            return destination
        temporary = destination.with_suffix(".part")
        temporary.unlink(missing_ok=True)
        url = f"https://civitai.com/api/download/models/{version_id}"
        with requests.get(url, headers=civitai_headers(), stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if temporary.stat().st_size < 1024 * 1024:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("O arquivo LoRA recebido é inválido ou incompleto.")
        temporary.replace(destination)
        return destination

    @staticmethod
    def _is_unsupported_lora_key(key: str) -> bool:
        """Identifica o metadado alpha que o conversor SGM do Diffusers rejeita."""
        return key.startswith("lora_") and key.endswith(".alpha")

    @classmethod
    def _prepare_lora_file(cls, source: Path) -> Path:
        """Cria uma cópia compatível quando a LoRA traz chaves alpha legadas."""
        compatible = source.with_name(f"{source.stem}_compatible{source.suffix}")
        if compatible.exists() and compatible.stat().st_mtime >= source.stat().st_mtime:
            return compatible

        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            raise RuntimeError("A dependência safetensors é necessária para preparar esta LoRA.") from exc

        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            unsupported = [key for key in keys if cls._is_unsupported_lora_key(key)]
            if not unsupported:
                return source
            tensors = {key: handle.get_tensor(key) for key in keys if key not in unsupported}
            metadata = handle.metadata() or {}

        temporary = compatible.with_suffix(".part")
        temporary.unlink(missing_ok=True)
        try:
            save_file(tensors, str(temporary), metadata=metadata)
            temporary.replace(compatible)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return compatible

    @staticmethod
    def _vram() -> float | None:
        try:
            import torch
            return round(torch.cuda.memory_allocated() / (1024**3), 2)
        except Exception:
            return None

    def generate(self, job: Job, update: Callable[[int, float | None], None]) -> Path:
        import torch

        with self.load_lock:
            self._load_pipeline()
            assert self.pipe is not None and self.img_pipe is not None
            pipeline = self.img_pipe if job.params.mode == "img2img" else self.pipe
            try:
                try:
                    self.pipe.unload_lora_weights()
                except Exception:
                    pass

                adapters: list[str] = []
                weights: list[float] = []
                for index, selected in enumerate(job.params.loras):
                    downloaded = self._download_lora(selected.version_id)
                    path = self._prepare_lora_file(downloaded)
                    adapter = f"lora_{index}_{selected.version_id}"
                    try:
                        self.pipe.load_lora_weights(str(path), adapter_name=adapter)
                    except ValueError as exc:
                        if "Checkpoint not supported because layer" in str(exc):
                            raise RuntimeError(
                                f"A LoRA '{selected.name}' não é compatível com o checkpoint Illustrious/SDXL."
                            ) from exc
                        raise
                    adapters.append(adapter)
                    weights.append(selected.weight)
                if adapters:
                    self.pipe.set_adapters(adapters, adapter_weights=weights)

                generator = torch.Generator(device="cuda").manual_seed(job.params.seed)
                callback_steps = max(job.params.steps, 1)

                def progress_callback(_: Any, step: int, __: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                    update(min(98, int((step + 1) * 100 / callback_steps)), self._vram())
                    return callback_kwargs

                options: dict[str, Any] = {
                    "prompt": job.params.prompt,
                    "negative_prompt": job.params.negative_prompt,
                    "num_inference_steps": job.params.steps,
                    "guidance_scale": job.params.guidance,
                    "generator": generator,
                    "callback_on_step_end": progress_callback,
                }
                if job.params.mode == "img2img":
                    if not job.params.source_image:
                        raise RuntimeError("O modo img2img requer uma imagem-base.")
                    with Image.open(job.params.source_image) as source:
                        options["image"] = source.convert("RGB")
                        options["strength"] = job.params.strength
                        result = pipeline(**options).images[0]
                else:
                    options.update({"width": job.params.width, "height": job.params.height})
                    result = pipeline(**options).images[0]

                output = OUTPUTS / f"{job.id}.png"
                result.save(output, format="PNG")
                update(100, self._vram())
                return output
            finally:
                try:
                    self.pipe.unload_lora_weights()
                except Exception:
                    pass
                torch.cuda.empty_cache()


class JobManager:
    def __init__(self, archive: MegaArchive, engine: GeneratorEngine) -> None:
        self.archive = archive
        self.engine = engine
        self.jobs: dict[str, Job] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.RLock()
        self.worker = threading.Thread(target=self._run, name="generation-worker", daemon=True)
        self.worker.start()

    def restore(self) -> None:
        for data in self.archive.list_remote_metadata():
            try:
                params = data["params"]
                loras = [LoRASelection(**item) for item in params.get("loras", [])]
                restored = Job(
                    id=data["id"], created_at=data["created_at"], status=data.get("status", "completed"),
                    progress=data.get("progress", 100), params=GenerationParams(
                        prompt=params["prompt"], negative_prompt=params.get("negative_prompt", ""), seed=params["seed"],
                        steps=params["steps"], guidance=params["guidance"], width=params["width"], height=params["height"],
                        strength=params.get("strength", 0.65), mode=params["mode"], loras=loras,
                        edit_level=params.get("edit_level", "medium"),
                    ), updated_at=data.get("updated_at"), completed_at=data.get("completed_at"),
                    filename=data.get("filename") or f"{data['id']}.png", mega_synced=True, error=data.get("error"), vram_gb=data.get("vram_gb"),
                )
                self.jobs[restored.id] = restored
            except (KeyError, TypeError, ValueError):
                continue

    def enqueue(self, params: GenerationParams) -> Job:
        job = Job(id=str(uuid.uuid4()), created_at=now_iso(), status="queued", progress=0, params=params, updated_at=now_iso())
        with self.lock:
            self.jobs[job.id] = job
        self.pending.put(job.id)
        return job

    def _update(self, job: Job, progress: int, vram: float | None) -> None:
        with self.lock:
            job.progress, job.vram_gb, job.updated_at = progress, vram, now_iso()

    def _run(self) -> None:
        while True:
            job_id = self.pending.get()
            with self.lock:
                job = self.jobs.get(job_id)
                if not job:
                    continue
                job.status, job.updated_at = "running", now_iso()
            try:
                image = self.engine.generate(job, lambda progress, vram: self._update(job, progress, vram))
                with self.lock:
                    job.filename = image.name
                    job.status, job.progress, job.completed_at, job.updated_at = "completed", 100, now_iso(), now_iso()
                    job.mega_synced = self.archive.save_job(job, image)
            except Exception as exc:
                with self.lock:
                    job.status, job.error, job.updated_at = "failed", str(exc)[:500], now_iso()
                    job.mega_synced = self.archive.save_job(job, None)
            finally:
                self.pending.task_done()

    def sync_pending(self) -> tuple[int, int]:
        """Reenvia PNGs e manifestos locais de jobs concluídos ainda pendentes."""
        with self.lock:
            pending = [
                job for job in self.jobs.values()
                if job.status == "completed" and not job.mega_synced
            ]
        synced = 0
        for job in pending:
            image = OUTPUTS / (job.filename or f"{job.id}.png")
            if not image.exists():
                continue
            if self.archive.save_job(job, image):
                synced += 1
        return len(pending), synced

    def public_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            return [job.public() for job in sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)]

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)


archive = MegaArchive()
engine = GeneratorEngine()
manager = JobManager(archive, engine)
archive_ready = threading.Event()
archive_restore_lock = threading.Lock()


def restore_archive() -> None:
    """Serializa restaurações para não competir com a preparação inicial do arquivo."""
    with archive_restore_lock:
        manager.restore()


def initialize_archive() -> None:
    """Prepara o MEGA em segundo plano para não bloquear a abertura do servidor."""
    try:
        archive.connect()
        restore_archive()
    finally:
        archive_ready.set()


threading.Thread(target=initialize_archive, name="archive-initializer", daemon=True).start()


def validate_params(raw: dict[str, Any], source_image: str | None) -> GenerationParams:
    mode = raw.get("mode", "text2img")
    if mode not in {"text2img", "img2img"}:
        raise ValueError("Modo de geração inválido.")
    edit_level = str(raw.get("edit_level", "medium")).strip().lower()
    if edit_level not in {"low", "medium", "high"}:
        raise ValueError("Nível de edição inválido. Use baixo, médio ou alto.")
    prompt = str(raw.get("prompt", "")).strip()
    if not prompt or len(prompt) > 4000:
        raise ValueError("Informe um prompt entre 1 e 4000 caracteres.")
    try:
        seed = int(raw.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        steps = int(raw.get("steps", 28))
        width = int(raw.get("width", 1024))
        height = int(raw.get("height", 1024))
        guidance = float(raw.get("guidance", 6.5))
        strength = float(raw.get("strength", 0.65))
    except (TypeError, ValueError) as exc:
        raise ValueError("Há um parâmetro numérico inválido.") from exc
    if not 10 <= steps <= 60 or not 1 <= guidance <= 15 or not 0.05 <= strength <= 1:
        raise ValueError("Steps, guidance ou strength estão fora dos limites aceitos.")
    if width not in range(512, 1025, 64) or height not in range(512, 1025, 64):
        raise ValueError("Largura e altura devem ser múltiplos de 64 entre 512 e 1024.")
    if mode == "img2img" and not source_image:
        raise ValueError("Envie uma imagem-base para usar img2img.")
    parsed_loras: list[LoRASelection] = []
    for candidate in raw.get("loras", [])[:MAX_LORAS]:
        try:
            version_id = int(candidate["version_id"])
            model_id = int(candidate["model_id"]) if candidate.get("model_id") else None
            weight = float(candidate.get("weight", 0.8))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A seleção de LoRA é inválida.") from exc
        if version_id <= 0 or not 0 <= weight <= 1.5:
            raise ValueError("O peso de LoRA deve estar entre 0 e 1,5.")
        parsed_loras.append(LoRASelection(version_id, model_id, str(candidate.get("name", "LoRA"))[:120], weight))
    return GenerationParams(
        prompt=prompt, negative_prompt=str(raw.get("negative_prompt", ""))[:4000], seed=seed,
        steps=steps, guidance=guidance, width=width, height=height, strength=strength,
        mode=mode, loras=parsed_loras, source_image=source_image, edit_level=edit_level,
    )


@app.route("/")
def index():
    if not session.get("authenticated"):
        return send_file(Path(app.static_folder) / "login.html")
    return send_file(Path(app.static_folder) / "index.html")


@app.route("/api/login", methods=["POST"])
def login():
    submitted = str((request.get_json(silent=True) or {}).get("password", ""))
    configured = os.environ.get("STUDIO_PASSWORD", "")
    if not configured or not hmac.compare_digest(submitted, configured):
        return jsonify({"error": "Senha inválida."}), 401
    session.clear()
    session["authenticated"] = True
    session["csrf"] = base64.urlsafe_b64encode(os.urandom(24)).decode()
    return jsonify({"csrf": session["csrf"]})


@app.route("/api/logout", methods=["POST"])
@authentication_required
@csrf_required
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/bootstrap")
@authentication_required
def bootstrap():
    # Restaura manifestos também quando a sessão MEGA já está autenticada.
    if archive_ready.is_set():
        restore_archive()
    last_settings = archive.load_last_settings()
    return jsonify({
        "csrf": session.get("csrf"), "jobs": manager.public_jobs(),
        "archive": {
            "available": archive.available, "ready": archive_ready.is_set(),
            "error": archive.error, "folder": MEGA_FOLDER,
        },
        "last_settings": last_settings,
        "last_settings_source": "mega" if last_settings else None,
        "limits": {"maxLoras": MAX_LORAS, "sizes": list(range(512, 1025, 64))},
        "model": {"name": "WAI-illustrious-SDXL", "cached": MODEL_PATH.exists()},
    })


@app.route("/api/catalog")
@authentication_required
def catalog():
    include_adult = request.args.get("include_adult", "").strip().lower() in {"1", "true", "yes"}
    if include_adult and not os.environ.get("CIVITAI_TOKEN", "").strip():
        return jsonify({"error": "Defina CIVITAI_TOKEN no servidor para consultar conteúdo adulto autorizado."}), 400
    params: dict[str, Any] = {
        "limit": min(max(int(request.args.get("limit", 24)), 1), 48), "types": "LORA",
        "baseModels": "Illustrious", "sort": request.args.get("sort", "Most Downloaded"),
        "period": request.args.get("period", "AllTime"), "primaryFileOnly": "true",
        "nsfw": "true" if include_adult else "false",
    }
    if request.args.get("cursor"):
        params["cursor"] = request.args["cursor"]
    if request.args.get("query"):
        params["query"] = request.args["query"][:120]
    if request.args.get("tag"):
        params["tag"] = request.args["tag"][:80]
    try:
        response = requests.get(f"{CIVITAI_BASE}/models", params=params, headers=civitai_headers(), timeout=25)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return jsonify({"error": f"Não foi possível consultar o catálogo Civitai: {str(exc)[:160]}"}), 502
    items = []
    for model in payload.get("items", []):
        versions = [item for item in model.get("modelVersions", []) if item.get("baseModel") == "Illustrious"]
        if not versions:
            continue
        version = versions[0]
        image = next((item.get("url") for item in version.get("images", []) if item.get("url")), None)
        version_items = []
        for candidate in versions:
            candidate_image = next((item.get("url") for item in candidate.get("images", []) if item.get("url")), None)
            version_items.append({
                "id": candidate.get("id"), "name": candidate.get("name") or f"Versão {candidate.get('id')}",
                "image": candidate_image, "downloads": candidate.get("stats", {}).get("downloadCount", 0),
                "created_at": candidate.get("createdAt"), "updated_at": candidate.get("updatedAt"),
            })
        items.append({
            "id": model.get("id"), "name": model.get("name"), "creator": model.get("creator", {}).get("username"),
            "tags": model.get("tags", [])[:10], "version_id": version.get("id"), "version": version.get("name"),
            "versions": version_items, "image": image, "downloads": version.get("stats", {}).get("downloadCount", 0), "mature": bool(model.get("nsfw")),
        })
    return jsonify({
        "items": items, "next_cursor": payload.get("metadata", {}).get("nextCursor"),
        "includes_adult": include_adult, "catalog_query": {"nsfw": params["nsfw"], "authenticated": bool(os.environ.get("CIVITAI_TOKEN", "").strip())},
    })


@app.route("/api/prompt-store")
@authentication_required
def prompt_store():
    include_adult = request.args.get("include_adult", "").strip().lower() in {"1", "true", "yes"}
    if include_adult and not os.environ.get("CIVITAI_TOKEN", "").strip():
        return jsonify({"error": "Defina CIVITAI_TOKEN no servidor para consultar conteúdo adulto autorizado."}), 400
    sort = request.args.get("sort", "Most Reactions")
    if sort not in {"Most Reactions", "Random", "Newest"}:
        sort = "Most Reactions"
    try:
        limit = min(max(int(request.args.get("limit", 24)), 1), 48)
    except (TypeError, ValueError):
        limit = 24
    params: dict[str, Any] = {
        "limit": limit,
        "sort": sort,
        "period": request.args.get("period", "AllTime"),
        "type": "image",
        "withMeta": "true",
        "nsfw": "true" if include_adult else "false",
    }
    if request.args.get("cursor"):
        params["cursor"] = request.args["cursor"]
    try:
        response = requests.get(f"{CIVITAI_BASE}/images", params=params, headers=civitai_headers(), timeout=25)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return jsonify({"error": f"Não foi possível consultar a Loja de Prompts no Civitai: {str(exc)[:160]}"}), 502

    search = request.args.get("query", "").strip().lower()[:120]
    filters = [term.strip().lower() for term in request.args.get("filters", "").split(",") if term.strip()][:8]
    items: list[dict[str, Any]] = []
    for image in payload.get("items", []):
        meta = image.get("meta") or {}
        prompt = str(meta.get("prompt") or meta.get("Prompt") or "").strip()
        negative_prompt = str(meta.get("negativePrompt") or meta.get("Negative prompt") or "").strip()
        resources = meta.get("civitaiResources") or []
        loras: list[dict[str, Any]] = []
        for resource in resources:
            if str(resource.get("type", "")).lower() != "lora":
                continue
            try:
                version_id = int(resource.get("modelVersionId") or resource.get("versionId"))
            except (TypeError, ValueError):
                continue
            loras.append({
                "version_id": version_id,
                "model_id": None,
                "name": f"LoRA // {version_id}",
                "weight": float(resource.get("weight", 0.8) or 0.8),
            })
        tags = [str(tag) for tag in (image.get("tags") or [])]
        haystack = " ".join([prompt, negative_prompt, str(image.get("username", "")), " ".join(tags)]).lower()
        if search and search not in haystack:
            continue
        if filters and not all(term in haystack for term in filters):
            continue
        items.append({
            "id": image.get("id"), "image": image.get("url"), "prompt": prompt,
            "negative_prompt": negative_prompt, "seed": meta.get("seed"),
            "steps": meta.get("steps"), "guidance": meta.get("cfgScale") or meta.get("guidanceScale"),
            "width": image.get("width") or (str(meta.get("Size", "")).split("x")[0] if "x" in str(meta.get("Size", "")) else None),
            "height": image.get("height") or (str(meta.get("Size", "")).split("x")[-1] if "x" in str(meta.get("Size", "")) else None),
            "username": image.get("username"), "created_at": image.get("createdAt"),
            "nsfw": bool(image.get("nsfw")), "nsfw_level": image.get("nsfwLevel"),
            "tags": tags[:12], "loras": loras[:MAX_LORAS],
            "reactions": (image.get("stats") or {}).get("heartCount", 0),
        })
    return jsonify({
        "items": items, "next_cursor": (payload.get("metadata") or {}).get("nextCursor"),
        "includes_adult": include_adult, "catalog_query": {"sort": sort, "authenticated": bool(os.environ.get("CIVITAI_TOKEN", "").strip())},
    })


@app.route("/api/prompt-store/image")
@authentication_required
def prompt_store_image():
    target = request.args.get("url", "").strip()
    parsed = urlparse(target)
    if parsed.scheme != "https" or parsed.hostname not in {"image.civitai.com", "images.civitai.com", "civitai.com"}:
        return jsonify({"error": "Origem de imagem não autorizada."}), 400
    try:
        response = requests.get(target, headers=civitai_headers(), timeout=25)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if not content_type.startswith("image/") or len(response.content) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "A imagem do Civitai não é válida para remix."}), 400
        return send_file(io.BytesIO(response.content), mimetype=content_type, download_name="civitai-remix.jpg")
    except requests.RequestException as exc:
        return jsonify({"error": f"Não foi possível baixar a imagem para remix: {str(exc)[:160]}"}), 502


@app.route("/api/jobs", methods=["POST"])
@authentication_required
@csrf_required
def create_job():
    raw = parse_json(request.form.get("payload"), request.get_json(silent=True) or {})
    source = None
    upload = request.files.get("image")
    if upload and upload.filename:
        extension = Path(upload.filename).suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
            return jsonify({"error": "A imagem-base deve ser PNG, JPG ou WEBP."}), 400
        source_path = UPLOADS / f"{uuid.uuid4()}{extension}"
        upload.save(source_path)
        try:
            with Image.open(source_path) as image:
                image.verify()
            source = str(source_path)
        except Exception:
            source_path.unlink(missing_ok=True)
            return jsonify({"error": "Não foi possível validar a imagem-base enviada."}), 400
    try:
        params = validate_params(raw, source)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    preferences_persisted = archive.save_last_settings(saved_settings(params))
    job = manager.enqueue(params)
    payload = job.public()
    payload["preferences_persisted"] = preferences_persisted
    return jsonify(payload), 202


@app.route("/api/jobs/<job_id>")
@authentication_required
def get_job(job_id: str):
    job = manager.get(job_id)
    return (jsonify(job.public()), 200) if job else (jsonify({"error": "Job não encontrado."}), 404)


@app.route("/api/history")
@authentication_required
def history():
    if request.args.get("sync") == "1":
        restore_archive()
    return jsonify({"items": manager.public_jobs()})


@app.route("/api/history/sync", methods=["POST"])
@authentication_required
@csrf_required
def history_sync():
    if not archive.available:
        archive.connect()
    pending, synced = manager.sync_pending()
    last_settings_synced = archive.sync_last_settings()
    before = len(manager.jobs)
    restore_archive()
    return jsonify({
        "items": manager.public_jobs(),
        "restored": max(0, len(manager.jobs) - before),
        "pending": pending,
        "synced": synced,
        "last_settings_synced": last_settings_synced,
        "archive": {"available": archive.available, "error": archive.error, "folder": MEGA_FOLDER},
    })


@app.route("/api/history/<job_id>/image")
@authentication_required
def history_image(job_id: str):
    job = manager.get(job_id)
    if not job or not job.filename:
        return jsonify({"error": "Imagem não encontrada."}), 404
    image = OUTPUTS / job.filename
    if not image.exists():
        image = archive.restore_image(job_id) or image
    if not image.exists():
        return jsonify({"error": "A imagem não está disponível no cache nem no MEGA."}), 404
    return send_file(image, mimetype="image/png", as_attachment=request.args.get("download") == "1", download_name=image.name)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok", "ready": archive_ready.is_set(),
        "queue": manager.pending.qsize(), "archive": archive.available,
    })


@app.errorhandler(413)
def too_large(_: Any):
    return jsonify({"error": "A imagem enviada excede 16 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
