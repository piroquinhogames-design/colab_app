"""ModelLab Studio — servidor para execução integral no Google Colab.

Variáveis obrigatórias: STUDIO_PASSWORD, MEGA_EMAIL, MEGA_PASSWORD.
Variáveis opcionais: CIVITAI_TOKEN, MODEL_URL, MODEL_REPO, MODEL_PATH,
MODELS_CONFIG, MODEL_ID, MEGA_FOLDER, STUDIO_SECRET e PORT.
"""

from __future__ import annotations

import base64
import asyncio
import io
import atexit
import math
import hashlib
import zipfile
import signal
from collections import defaultdict, deque
import functools
import hmac
import json
import os
import queue
import random
import re
import shutil
import threading
import time
import types
import uuid
from dataclasses import dataclass, field, asdict, replace
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
from PIL import Image, PngImagePlugin
from studio_storage import artifact_path, atomic_json, read_json, JOB_ID
from civitai_resources import version_metadata, model_file
from studio_downloads import digest_file
from studio_security import LoginLimiter, bounded_civitai_image
login_limiter = LoginLimiter()


ROOT = Path(os.environ.get("STUDIO_ROOT", "/content/modellab-studio")).resolve()
# O cache fica fora do diretório do checkpoint para permitir trocar o perfil sem
# duplicar os shards. Se houver um cache da execução anterior, ele é reaproveitado.
legacy_hf_home = Path.home() / ".cache" / "huggingface"
default_hf_home = ROOT / "huggingface-cache"
HF_HOME = Path(os.environ.get("HF_HOME") or (legacy_hf_home if legacy_hf_home.exists() else default_hf_home)).resolve()
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE") or (HF_HOME / "hub")).resolve()
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# O Nova EXAnime AM é um modelo Anima bf16. O carregador correto é o
# UNETLoader nativo do ComfyUI; o arquivo fica no diretório diffusion_models.
MODELS = ROOT / "models"
LORAS = ROOT / "loras"
OUTPUTS = ROOT / "outputs"
UPLOADS = ROOT / "uploads"
for directory in (MODELS, LORAS, OUTPUTS, UPLOADS, HF_HUB_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_LORAS = int(os.environ.get("MODELLAB_MAX_LORAS", "8"))
MODEL_URL = os.environ.get(
    "MODEL_URL", "https://civitai.com/api/download/models/2983680?fileId=2863158"
)
MODEL_REPO = os.environ.get("MODEL_REPO", "")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS / "diffusion_models" / "WAI-ANIMA1.safetensors"))
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "wai-anima")
MEGA_FOLDER = os.environ.get("MEGA_FOLDER", "ModelLabStudio")
CIVITAI_BASE = "https://civitai.com/api/v1"
LAST_SETTINGS_NAME = "last_settings.json"
MODEL_PROFILE_CACHE = ROOT / "model_profiles.json"

# A família controla tanto a busca de LoRAs quanto os defaults enviados ao motor.

MODEL_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "anima": {
        "base": "Anima", "engine": "comfyui", "lora_base": "Anima",
        "defaults": {
            "steps": 24, "guidance": 5.0, "strength": 0.75, "sampler": "euler_a",
            "positive_prefix": "masterpiece, best quality, score_9, score_8, score_7, year 2025, newest, highres, absurdres, very aesthetic",
            "negative_prompt": "worst quality, low quality, early, old, score_1, score_2, score_3, cartoon, graphic, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, bad anatomy, bad hands, missing fingers, extra fingers, extra digits, fewer digits, cropped, very displeasing, artist name, blurry, jpeg artifacts, lowres, censor",
        },
        "notes": "Nova EXAnime AM; Anima B1 + A11. Usa o workflow nativo do ComfyUI, sem Diffusers/SDXL.",
    },
    "sdxl-illustrious": {
        "base": "Illustrious", "engine": "sdxl", "lora_base": "Illustrious",
        "defaults": {"steps": 28, "guidance": 6.5, "strength": 0.65, "sampler": "euler_a"},
        "notes": "SDXL derivado de Illustrious; compatível com a maioria das LoRAs Illustrious quando a variante coincide.",
    },
    "pony": {
        "base": "Pony", "engine": "sdxl", "lora_base": "Pony",
        "defaults": {"steps": 30, "guidance": 5.5, "strength": 0.65, "sampler": "euler_a"},
        "notes": "Prefect Pony XL V6 é um checkpoint SDXL fp16; use LoRAs Pony/SDXL compatíveis.",
    },
    "sdxl": {
        "base": "SDXL 1.0", "engine": "sdxl", "lora_base": "SDXL 1.0",
        "defaults": {"steps": 28, "guidance": 6.5, "strength": 0.65, "sampler": "euler_a"},
        "notes": "SDXL convencional.",
    },
    "flux": {
        "base": "Flux", "engine": "unsupported", "lora_base": "Flux",
        "defaults": {"steps": 28, "guidance": 3.5, "strength": 0.65, "sampler": "euler_a"},
        "notes": "Catalogável, mas exige um engine Flux separado antes de gerar.",
    },
    "sd3": {
        "base": "SD 3", "engine": "unsupported", "lora_base": "SD 3",
        "defaults": {"steps": 28, "guidance": 5.0, "strength": 0.65, "sampler": "euler_a"},
        "notes": "Catalogável, mas exige um engine SD3 separado antes de gerar.",
    },
}
SUPPORTED_MODEL_FAMILIES = set(MODEL_FAMILY_PROFILES)
SUPPORTED_SAMPLERS = {"euler_a", "euler", "dpmpp_2m", "dpmpp_2m_sde_gpu"}


def normalize_model_family(base_model: str | None) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", str(base_model or "").lower()).strip()
    if "anima" in value:
        return "anima"
    if "pony" in value:
        return "pony"
    if "illustrious" in value or "noobai" in value or "noob ai" in value:
        return "sdxl-illustrious"
    if "flux" in value:
        return "flux"
    if "sd 3" in value or "sd3" in value:
        return "sd3"
    if "sdxl" in value:
        return "sdxl"
    return "sdxl"


def family_profile(family: str | None) -> dict[str, Any]:
    normalized = str(family or "sdxl").strip().lower()
    return MODEL_FAMILY_PROFILES.get(normalized, MODEL_FAMILY_PROFILES["sdxl"])


def civitai_base_for_family(family: str | None) -> str:
    """Retorna o rótulo local da base, sem assumir um enum mutável do Civitai."""
    return str(family_profile(family).get("lora_base", "SDXL 1.0"))


def version_matches_family(version: dict[str, Any], family: str | None, model_name: str = "") -> bool:
    expected = civitai_base_for_family(family).lower()
    actual = str(version.get("baseModel") or model_name or "").lower()
    if not actual:
        return False
    if family == "anima":
        # O Civitai já publicou versões com "Anima", "Anima B1 + A11"
        # e nomes equivalentes. O filtro remoto é omitido para essa família
        # e esta verificação local evita perder resultados legítimos.
        return "anima" in actual
    if expected == "sdxl 1.0":
        return actual in {"sdxl 1.0", "sdxl"}
    return expected in actual or actual in expected


def _filter_day(value: str | None) -> Any:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _published_day(value: Any) -> Any:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError, OverflowError):
        return None


def _matches_date_range(value: Any, start_day: Any, end_day: Any) -> bool:
    if start_day is None and end_day is None:
        return True
    day = _published_day(value)
    if day is None:
        return False
    return (start_day is None or day >= start_day) and (end_day is None or day <= end_day)


def _request_date_range(args: Any) -> tuple[Any, Any]:
    start_raw = args.get("date_from", "").strip()
    end_raw = args.get("date_to", "").strip()
    start_day = _filter_day(start_raw)
    end_day = _filter_day(end_raw)
    if start_raw and start_day is None:
        raise ValueError("A data inicial deve estar no formato AAAA-MM-DD.")
    if end_raw and end_day is None:
        raise ValueError("A data final deve estar no formato AAAA-MM-DD.")
    if start_day and end_day and start_day > end_day:
        raise ValueError("A data inicial não pode ser posterior à data final.")
    return start_day, end_day


def _load_model_specs() -> dict[str, dict[str, Any]]:
    """Carrega checkpoints com perfil de família e defaults adaptativos."""
    default_family = os.environ.get("MODEL_FAMILY", "anima").strip().lower()
    base_profile = family_profile(default_family)
    default = {
        "id": DEFAULT_MODEL_ID,
        "name": "WAI-ANIMA v1.0" if DEFAULT_MODEL_ID == "wai-anima" else ("Nova EXAnime AM" if DEFAULT_MODEL_ID == "nova-exanime-am" else DEFAULT_MODEL_ID),
        "url": MODEL_URL,
        "repo": MODEL_REPO,
        "path": str(MODEL_PATH),
        "family": default_family,
        "base": base_profile["base"],
        "engine": base_profile["engine"],
        "lora_base": base_profile["lora_base"],
        "defaults": dict(base_profile["defaults"]),
        "notes": base_profile["notes"],
        "civitai_model_id": 2544636 if DEFAULT_MODEL_ID == "wai-anima" else (2856434 if DEFAULT_MODEL_ID == "nova-exanime-am" else None),
        "file_id": 2863158 if DEFAULT_MODEL_ID == "wai-anima" else (3108312 if DEFAULT_MODEL_ID == "nova-exanime-am" else None),
        "version_id": 2983680 if DEFAULT_MODEL_ID == "wai-anima" else (3226184 if DEFAULT_MODEL_ID == "nova-exanime-am" else None),
    }
    specs: dict[str, dict[str, Any]] = {DEFAULT_MODEL_ID: default}
    if DEFAULT_MODEL_ID == "wai-anima":
        default["defaults"].update({
            "steps": 24, "guidance": 5.0, "sampler": "euler_a",
            "positive_prefix": "masterpiece, best quality, score_7",
            "negative_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, lowres, censor",
        })
        default["notes"] = "WAI-ANIMA v1.0 (base Anima 1.0), FP16; encoder Qwen e Qwen Image VAE via ComfyUI."
        specs["nova-exanime-am"] = {
            **default, "id": "nova-exanime-am", "name": "Nova EXAnime AM",
            "url": "https://civitai.com/api/download/models/3226184?fileId=3108312",
            "path": str(MODELS / "diffusion_models" / "novaExanimeAM_v10.safetensors"),
            "defaults": dict(base_profile["defaults"]), "notes": base_profile["notes"],
            "civitai_model_id": 2856434, "version_id": 3226184, "file_id": 3108312,
        }
    raw = os.environ.get("MODELS_CONFIG", "").strip()
    try:
        decoded = json.loads(raw or "[]")
        candidates = decoded.values() if isinstance(decoded, dict) else decoded
        if not isinstance(candidates, list) and not isinstance(decoded, dict):
            candidates = []
        if isinstance(decoded, dict):
            candidates = list(decoded.values())
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            model_id = re.sub(r"[^a-z0-9._-]+", "-", str(candidate.get("id", "")).lower()).strip("-")
            if not model_id:
                continue
            candidate_family = str(candidate.get("family") or normalize_model_family(candidate.get("base"))).strip().lower()
            if candidate_family not in SUPPORTED_MODEL_FAMILIES:
                continue
            inherited = family_profile(candidate_family)
            profile = {**default, **inherited, **candidate}
            if model_id != DEFAULT_MODEL_ID:
                for key in ("civitai_model_id", "version_id", "file_id", "sha256", "repo"):
                    if key not in candidate: profile.pop(key, None)
            profile["id"] = model_id
            profile["name"] = str(candidate.get("name") or model_id)[:120]
            profile["family"] = candidate_family
            profile["path"] = str(candidate.get("path") or MODELS / "diffusion_models" / f"{model_id}.safetensors")
            profile["defaults"] = {**inherited["defaults"], **(candidate.get("defaults") or {})}
            if candidate_family == "pony" and "engine" not in candidate:
                # Checkpoints Pony vindos do Civitai usam o pipeline SDXL.
                profile["engine"] = "sdxl"
                profile.pop("repo", None)
            specs[model_id] = profile
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        cached = json.loads(MODEL_PROFILE_CACHE.read_text(encoding="utf-8")) if MODEL_PROFILE_CACHE.exists() else []
        candidates = list(cached.values()) if isinstance(cached, dict) else cached
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            model_id = re.sub(r"[^a-z0-9._-]+", "-", str(candidate.get("id", "")).lower()).strip("-")
            if not model_id or model_id == DEFAULT_MODEL_ID:
                continue
            candidate_family = str(candidate.get("family") or normalize_model_family(candidate.get("base"))).strip().lower()
            if candidate_family not in SUPPORTED_MODEL_FAMILIES:
                continue
            inherited = family_profile(candidate_family)
            profile = {**default, **inherited, **candidate}
            if model_id != DEFAULT_MODEL_ID:
                for key in ("civitai_model_id", "version_id", "file_id", "sha256", "repo"):
                    if key not in candidate: profile.pop(key, None)
            profile["id"] = model_id
            profile["name"] = str(candidate.get("name") or model_id)[:120]
            profile["family"] = candidate_family
            profile["path"] = str(candidate.get("path") or MODELS / "diffusion_models" / f"{model_id}.safetensors")
            profile["defaults"] = {**inherited["defaults"], **(candidate.get("defaults") or {})}
            if candidate_family == "pony" and "engine" not in candidate:
                # Checkpoints Pony vindos do Civitai usam o pipeline SDXL.
                profile["engine"] = "sdxl"
                profile.pop("repo", None)
            specs[model_id] = profile
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return specs


MODEL_SPECS = _load_model_specs()


def get_model_spec(model_id: str | None = None) -> dict[str, Any]:
    selected = str(model_id or DEFAULT_MODEL_ID).strip().lower()
    return MODEL_SPECS.get(selected) or MODEL_SPECS[DEFAULT_MODEL_ID]


def public_model_spec(spec: dict[str, Any]) -> dict[str, Any]:
    family = str(spec.get("family", "sdxl"))
    engine = str(spec.get("engine") or family_profile(family).get("engine", "unsupported"))
    ready = engine == "comfyui"
    return {
        "id": spec["id"], "name": spec.get("name", spec["id"]),
        "family": family, "base": spec.get("base", family_profile(family)["base"]),
        "lora_base": spec.get("lora_base", civitai_base_for_family(family)),
        "engine": engine, "ready": ready, "cached": Path(spec["path"]).exists(),
        "defaults": spec.get("defaults", family_profile(family)["defaults"]),
        "notes": spec.get("notes", ""), "repo": spec.get("repo"),
        "civitai_model_id": spec.get("civitai_model_id"),
        "version_id": spec.get("version_id"), "file_id": spec.get("file_id"),
        "capabilities": {"text2img": ready, "img2img": ready, "tiled_decode": ready},
    }

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config.update(
    SECRET_KEY=os.environ.get("STUDIO_SECRET") or os.urandom(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("STUDIO_COOKIE_SECURE", "1") == "1",
    PERMANENT_SESSION_LIFETIME=__import__("datetime").timedelta(hours=12),
    TRUSTED_HOSTS=os.environ.get("STUDIO_TRUSTED_HOSTS", "localhost,127.0.0.1,.trycloudflare.com").split(","),
    MAX_FORM_PARTS=16,
    MAX_FORM_MEMORY_SIZE=128 * 1024,
)


@app.after_request
def prevent_stale_app_assets(response):
    """Keep the HTML shell and executable assets in sync after a Colab restart."""
    path = request.path or ""
    cache_sensitive = path == "/" or path in {
        "/static/index.html", "/static/login.html", "/static/login.js", "/static/app.js", "/static/settings.js", "/static/style.css",
    }
    if cache_sensitive:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' blob: data: https://image.civitai.com https://images.civitai.com; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def preference_list(name: str) -> list:
    raw = read_json(OUTPUTS / name, [])
    if not isinstance(raw, list): return []
    if name == "favorites.json": return [item for item in raw[:10000] if isinstance(item, str) and JOB_ID.fullmatch(item)]
    return [item for item in raw[:50] if isinstance(item, dict) and isinstance(item.get("id"), str)
        and JOB_ID.fullmatch(item["id"]) and isinstance(item.get("name"), str) and len(item["name"]) <= 80
        and isinstance(item.get("settings"), dict) and isinstance(item["settings"].get("prompt"), str)]


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
    headers = {"User-Agent": "ModelLab-Studio/1.0"}
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
        if not token or not hmac.compare_digest(token.encode(), session.get("csrf", "").encode()):
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
    model_id: str = DEFAULT_MODEL_ID
    sampler: str = "euler_a"
    upscale: float = 1.0


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
        "model": params.model_id,
        "sampler": params.sampler, "upscale": params.upscale,
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
    sync_status: str = "pending"
    sync_error: str | None = None
    sync_attempts: int = 0
    model_snapshot: dict[str, Any] = field(default_factory=dict)
    prompt_id: str | None = None
    cancel_requested: bool = False
    download_progress: int = 0
    pipeline_progress: int = 0
    progress_phase: str = "queued"

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["params"]["loras"] = [asdict(item) for item in self.params.loras]
        data["params"]["model"] = data["params"].pop("model_id", DEFAULT_MODEL_ID)
        data["image_url"] = f"/api/history/{self.id}/image" if self.filename else None
        return data


class MegaArchive:
    """Adaptador mínimo que mantém imagens e metadados no mesmo diretório MEGA."""

    def __init__(self) -> None:
        self.client = None
        self.folder = None
        self.available = False
        self.error: str | None = None
        self.lock = threading.RLock()

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
        if isinstance(value, dict): return value.get("h")
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

    def _folder_nodes(self) -> list[Any]:
        if not self.client or not self.folder:
            return []
        nodes = self._file_nodes(self.client.get_files())
        return [node for node in nodes if isinstance(node, tuple) and node[1].get("p") == self.folder]

    def _find_file(self, name: str) -> Any:
        nodes = [node for node in self._folder_nodes() if self._node_name(node) == name]
        return sorted(nodes, key=lambda node: (node[1].get("ts", 0), str(node[0])), reverse=True)[0] if nodes else None

    def restore_preferences(self) -> None:
        for name in ("presets.json", "favorites.json"):
            destination = OUTPUTS / name
            if destination.exists(): continue
            with self.lock:
                node = self._find_file(name)
                if node: self.client.download(node, str(OUTPUTS))

    def _upload(self, path: Path) -> None:
        if not self.available or not self.client or not self.folder:
            raise RuntimeError("MEGA não está conectado a uma pasta de destino válida")
        with self.lock:
            previous = [node for node in self._folder_nodes() if self._node_name(node) == path.name]
            uploaded = self.client.upload(str(path), self.folder)
            if uploaded is None:
                raise RuntimeError(f"O cliente MEGA não confirmou o upload de {path.name}")
            new_handle = self._first_node(uploaded)
            for node in previous:
                if node[0] != new_handle:
                    self.client.destroy(node[0])

    def save_job(self, job: Job, image_path: Path | None) -> bool:
        metadata_path = artifact_path(OUTPUTS, job.id, ".json")
        synced = False
        if self.available:
            try:
                # O PNG é enviado primeiro; somente depois o manifesto confirma
                # a sincronização. Isso mantém o retry local seguro e evita um
                # terceiro upload do mesmo JSON.
                if image_path and image_path.exists():
                    self._upload(image_path)
                synced = True
            except Exception as exc:
                self.error = f"Falha ao enviar ao MEGA: {str(exc)[:180]}"
        job.mega_synced = synced
        # O manifesto local é sempre persistido, mesmo sem MEGA, para que o
        # histórico de imagens possa ser restaurado de forma automática e rápida
        # na próxima recarga, sem depender da reconexão do arquivo remoto.
        atomic_json(metadata_path, job.public())
        if not synced:
            return False
        try:
            self._upload(metadata_path)
            return True
        except Exception as exc:
            job.mega_synced = False
            atomic_json(metadata_path, job.public())
            self.error = f"Falha ao enviar ao MEGA: {str(exc)[:180]}"
            return False

    def save_last_settings(self, settings: dict[str, Any], *, sync: bool = True) -> bool:
        """Substitui o manifesto único do último sinal renderizado no arquivo MEGA."""
        settings_path = OUTPUTS / LAST_SETTINGS_NAME
        payload = {"updated_at": now_iso(), "settings": settings}
        atomic_json(settings_path, payload)
        if not sync or not self.available:
            return False
        try:
            self._upload(settings_path)
            return True
        except Exception as exc:
            self.error = f"Falha ao salvar preferências no MEGA: {str(exc)[:180]}"
            return False

    def load_last_settings(self, *, allow_remote: bool = True) -> dict[str, Any] | None:
        """Retorna primeiro o cache local e acessa o MEGA somente quando permitido."""
        cache = ROOT / "mega-cache"
        cache.mkdir(exist_ok=True)
        local = OUTPUTS / LAST_SETTINGS_NAME
        cached_remote = cache / LAST_SETTINGS_NAME
        for candidate in (local, cached_remote):
            try:
                if candidate.exists():
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    settings = payload.get("settings") if isinstance(payload, dict) else None
                    if isinstance(settings, dict):
                        return settings
            except (OSError, json.JSONDecodeError):
                continue
        if not allow_remote or not self.available or not self.client:
            return None
        try:
            node = self._download_node(self._find_file(LAST_SETTINGS_NAME))
            if not node:
                return None
            downloaded = self.client.download(node, str(cache))
            local = Path(downloaded) if downloaded else cached_remote
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
            for node in self._folder_nodes():
                name = self._node_name(node)
                if name in {LAST_SETTINGS_NAME, "presets.json", "favorites.json"} or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}\.json", name, re.I):
                    continue
                try:
                    cached = cache / name
                    stamp = cache / (name + ".remote.json")
                    identity = {"handle": node[0], "ts": node[1].get("ts"), "size": node[1].get("s")}
                    if cached.exists() and read_json(stamp) == identity:
                        downloaded = str(cached)
                    else:
                        downloaded = self.client.download(node, str(cache))
                        atomic_json(stamp, identity)
                    local = Path(downloaded) if downloaded else cache / name
                    if local.exists():
                        payload = json.loads(local.read_text(encoding="utf-8"))
                        if isinstance(payload, dict) and payload.get("id"):
                            jobs.append(payload)
                except Exception:
                    continue
        except Exception as exc:
            self.error = f"Falha ao ler o histórico MEGA: {str(exc)[:180]}"
            raise RuntimeError(self.error) from exc
        return jobs

    def delete_job(self, job_id: str) -> bool:
        """Remove PNG e manifesto do arquivo MEGA sem falhar se um deles já não existir."""
        if not self.available or not self.client:
            return False
        try:
            with self.lock:
                for name in (f"{job_id}.png", f"{job_id}.json"):
                    nodes = [node for node in self._folder_nodes() if self._node_name(node) == name]
                    for node in nodes:
                        self.client.destroy(node[0])
            return True
        except Exception as exc:
            self.error = f"Falha ao excluir o job do MEGA: {str(exc)[:180]}"
            return False

    def restore_image(self, job_id: str) -> Path | None:
        destination = artifact_path(OUTPUTS, job_id, ".png")
        if destination.exists():
            return destination
        if not self.available or not self.client:
            return None
        try:
            node = self._download_node(self._find_file(destination.name))
            if node:
                downloaded = self.client.download(node, str(OUTPUTS))
                result = Path(downloaded) if downloaded else destination
                return result if result.exists() else None
        except Exception as exc:
            self.error = f"Falha ao recuperar imagem do MEGA: {str(exc)[:180]}"
        return None


class GeneratorEngine:
    """Executa o workflow Anima no backend ComfyUI, sem abrir a UI ou descarregar o modelo."""

    def __init__(self) -> None:
        from comfy_backend import ComfyBackend

        self.pipe = None
        self.img_pipe = None
        self.device = "cuda"
        self.loaded_model_id: str | None = None
        self.load_lock = threading.Lock()
        comfy_root = Path(os.environ.get("COMFY_ROOT", ROOT / "comfyui-runtime"))
        # O custom node vive junto ao código do projeto; o ComfyUI/modelos vivem
        # no diretório persistente STUDIO_ROOT/COMFY_ROOT.
        project_root = Path(__file__).resolve().parent
        self.comfy = ComfyBackend(project_root, comfy_root, int(os.environ.get("COMFY_PORT", "8188")))

    def _vram(self) -> float | None:
        return self.comfy.gpu_memory().get("used_gb")

    @staticmethod
    def _extract_first_image(result: Any) -> Image.Image:
        """Mantém o extrator de contrato para resultados de pipelines e dicionários."""
        images = getattr(result, "images", None)
        if images is None and isinstance(result, dict):
            images = result.get("images")
        if not images:
            raise RuntimeError("O backend não retornou nenhuma imagem.")
        image = images[0]
        if not isinstance(image, Image.Image):
            raise RuntimeError("O backend retornou uma imagem em formato não suportado.")
        return image.convert("RGB")

    def ensure_checkpoint(self, spec: dict[str, Any], progress: Callable[[int], None] | None = None) -> Path:
        report = progress or (lambda _value: None)
        model_path = Path(spec["path"]).expanduser()
        url = str(spec.get("url") or "").strip()
        if not url:
            raise RuntimeError(f"O checkpoint {spec.get('name', spec['id'])} não possui URL de download.")
        return self.comfy.ensure_file(url, model_path, 1, report, sha256=spec.get("sha256"))

    @staticmethod
    def _is_unsupported_lora_key(key: str) -> bool:
        return False

    @classmethod
    def _prepare_lora_file(cls, source: Path) -> Path:
        return source

    def _download_lora(self, version_id: int, cancelled=None, model_id=None) -> Path:
        data = version_metadata(version_id)
        resource = model_file(data, kind="LORA", model_id=model_id)
        destination = LORAS / f"civitai_{version_id}_{resource['file_id']}.safetensors"
        return self.comfy.ensure_file(resource["url"], destination, 1,
            sha256=resource["sha256"], expected_bytes=resource["size_bytes"], cancelled=cancelled)

    def _resolve_checkpoint(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec.get("version_id"):
            data = version_metadata(int(spec["version_id"]))
            spec.update(model_file(data, kind="Checkpoint", model_id=spec.get("civitai_model_id"), file_id=spec.get("file_id")))
        return spec

    def _load_pipeline(self, spec: dict[str, Any], update: Callable[..., None] | None = None) -> None:
        report = update or (lambda *_args, **_kwargs: None)
        engine = str(spec.get("engine") or family_profile(spec.get("family")).get("engine", "unsupported"))
        if engine != "comfyui":
            raise RuntimeError(f"O perfil {spec.get('name', spec['id'])} não usa o engine ComfyUI Anima.")
        if not self.comfy.comfy_dir.joinpath("main.py").exists():
            raise RuntimeError("ComfyUI não está instalado. Execute launch_colab.py novamente.")
        if not __import__("torch").cuda.is_available():
            raise RuntimeError("Nenhuma GPU CUDA foi detectada. Ative uma sessão T4 no Colab.")
        report(0, self._vram(), download=0, pipeline=0, phase="checking_model")
        self._resolve_checkpoint(spec)
        model_path = Path(spec["path"]).expanduser()
        self.comfy.ensure_file(
            str(spec["url"]), model_path, 1,
            lambda value: report(0, self._vram(), download=value, pipeline=0, phase="downloading_model"),
            sha256=spec.get("sha256"), expected_bytes=spec.get("size_bytes", 0),
            cancelled=self.comfy.cancel_event.is_set,
        )
        report(0, self._vram(), download=100, pipeline=15, phase="downloading_anima_components")
        self.comfy.ensure_anima_dependencies(
            lambda _progress, _vram, _pipeline, phase: report(0, self._vram(), download=100, pipeline=20, phase=phase)
        )
        report(0, self._vram(), download=100, pipeline=70, phase="starting_comfy_backend")
        self.comfy.ensure_running()
        report(0, self._vram(), download=100, pipeline=70, phase="backend_ready")

    def generate(self, job: Job, update: Callable[..., None]) -> Path:
        with self.load_lock:
            spec = get_model_spec(job.params.model_id)
            self.comfy.cancel_event.clear()
            if job.cancel_requested: raise InterruptedError("Geração cancelada.")
            self._load_pipeline(spec, update)
            job.model_snapshot = {key: spec.get(key) for key in ("id", "name", "family", "version_id", "file_id", "sha256", "defaults", "engine")}
            runtime = read_json(ROOT / "runtime_versions.json", {})
            job.model_snapshot["runtime"] = {"comfyui_commit": runtime.get("comfyui_commit"), "python": runtime.get("python"),
                "packages": {key: runtime.get("packages", {}).get(key) for key in ("torch", "transformers", "safetensors", "pydantic", "pydantic_core")}}
            job.model_snapshot["loras"] = []
            update(0, self._vram(), download=100, pipeline=100, phase="preparing_loras")
            lora_names: list[tuple[str, float]] = []
            for selected in job.params.loras:
                downloaded = self._prepare_lora_file(self._download_lora(selected.version_id, self.comfy.cancel_event.is_set, selected.model_id))
                job.model_snapshot["loras"].append({"version_id": selected.version_id, "weight": selected.weight,
                    "sha256": read_json(downloaded.with_suffix(downloaded.suffix + ".verified.json"), {}).get("sha256")})
                lora_names.append((self.comfy.copy_lora(downloaded), selected.weight))
            model_name = self.comfy.register_checkpoint(Path(spec["path"]))
            if job.params.source_image:
                job.comfy_source = self.comfy.upload_source(Path(job.params.source_image))
            workflow = self.comfy.build_workflow(job, spec, model_name, lora_names)
            atomic_json(artifact_path(OUTPUTS, job.id, ".workflow.json"), workflow)
            update(0, self._vram(), download=100, pipeline=100, phase="generating")
            image = self.comfy.submit_and_wait(
                workflow,
                lambda progress, vram, _pipeline, phase: update(
                    progress, vram, download=100, pipeline=100, phase=phase
                ),
                on_submitted=lambda prompt_id: setattr(job, "prompt_id", prompt_id),
            )
            if job.cancel_requested: raise InterruptedError("Geração cancelada.")
            self.loaded_model_id = spec["id"]
            output = artifact_path(OUTPUTS, job.id, ".png")
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("prompt", json.dumps(workflow, ensure_ascii=False))
            metadata.add_text("modellab", json.dumps({"settings": saved_settings(job.params), "model": job.model_snapshot}, ensure_ascii=False))
            image.save(output, format="PNG", pnginfo=metadata)
            thumb = image.copy(); thumb.thumbnail((384, 384))
            thumb.save(artifact_path(OUTPUTS, job.id, ".thumb.jpg"), "JPEG", quality=82)
            update(100, self._vram(), download=100, pipeline=100, phase="completed")
            return output


class JobManager:
    def __init__(self, archive: MegaArchive, engine: GeneratorEngine) -> None:
        self.archive = archive
        self.engine = engine
        self.jobs: dict[str, Job] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.sync_queue = queue.Queue()
        self.sync_scheduled: set[str] = set()
        self.job_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
        self.idempotency: dict[str, tuple[str, str]] = {}
        self.tombstones_path = ROOT / "deleted_jobs.json"
        self.deleted = set(read_json(self.tombstones_path, []))
        self.sync_worker = threading.Thread(target=self._sync_run, name="archive-worker", daemon=True)
        if os.environ.get("STUDIO_START_WORKERS", "1") == "1": self.sync_worker.start()
        self.worker = threading.Thread(target=self._run, name="generation-worker", daemon=True)
        if os.environ.get("STUDIO_START_WORKERS", "1") == "1": self.worker.start()

    @staticmethod
    def _job_from_data(data: dict[str, Any]) -> Job:
        if not isinstance(data, dict) or not JOB_ID.fullmatch(str(data.get("id", ""))):
            raise ValueError("Manifesto inválido.")
        params = data["params"]
        if not isinstance(params, dict): raise ValueError("Parâmetros inválidos no manifesto.")
        if data.get("filename") and data["filename"] != f"{data['id']}.png":
            raise ValueError("Caminho de imagem inválido no manifesto.")
        if not isinstance(data.get("created_at"), str) or not isinstance(params.get("prompt"), str) or len(params["prompt"]) > 4000:
            raise ValueError("Manifesto inválido.")
        if data.get("status", "completed") not in {"queued", "running", "completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("Estado inválido no manifesto.")
        for key in ("seed", "steps", "width", "height"):
            if not isinstance(params.get(key), int) or isinstance(params.get(key), bool): raise ValueError("Número inválido no manifesto.")
        if not 0 <= params["seed"] <= 2**53 - 1 or not 10 <= params["steps"] <= 60 or any(params[k] not in range(512, 1025, 64) for k in ("width", "height")):
            raise ValueError("Parâmetros fora dos limites no manifesto.")
        if not isinstance(params.get("guidance"), (int, float)) or not math.isfinite(params["guidance"]) or not 1 <= params["guidance"] <= 15:
            raise ValueError("Guidance inválido no manifesto.")
        if not isinstance(params.get("loras", []), list) or len(params.get("loras", [])) > MAX_LORAS: raise ValueError("LoRAs inválidas no manifesto.")
        loras = [LoRASelection(**item) for item in params.get("loras", [])]
        if any(not isinstance(item.version_id, int) or item.version_id <= 0 or not isinstance(item.weight, (int, float)) or not 0 <= item.weight <= 1.5 for item in loras): raise ValueError("LoRA inválida no manifesto.")
        completed = data.get("status", "completed") == "completed"
        return Job(
            id=data["id"], created_at=data["created_at"], status=data.get("status", "completed"),
            progress=data.get("progress", 100),
            download_progress=data.get("download_progress", 100 if completed else 0),
            pipeline_progress=data.get("pipeline_progress", 100 if completed else 0),
            progress_phase=data.get("progress_phase", "completed" if completed else "queued"),
            params=GenerationParams(
                prompt=params["prompt"], negative_prompt=params.get("negative_prompt", ""), seed=params["seed"],
                steps=params["steps"], guidance=params["guidance"], width=params["width"], height=params["height"],
                strength=params.get("strength", 0.65), mode=params["mode"], upscale=params.get("upscale", 1.0),
                model_id=str(params.get("model") or DEFAULT_MODEL_ID), sampler=params.get("sampler", "euler_a"),
                loras=loras, edit_level=params.get("edit_level", "medium"),
            ), updated_at=data.get("updated_at"), completed_at=data.get("completed_at"),
            filename=f"{data['id']}.png" if data.get("status", "completed") == "completed" else None, mega_synced=bool(data.get("mega_synced", False)),
            sync_status=data.get("sync_status", "synced" if data.get("mega_synced") else "pending"),
            sync_error=data.get("sync_error"), sync_attempts=int(data.get("sync_attempts", 0)),
            model_snapshot=data.get("model_snapshot") or {}, prompt_id=data.get("prompt_id"),
            error=data.get("error"), vram_gb=data.get("vram_gb"),
        )

    def restore(self) -> None:
        """Restaura manifestos do MEGA sem sobrescrever registros já presentes
        no cache local; marca como sincronizados os que já existiam localmente."""
        for data in self.archive.list_remote_metadata():
            try:
                job = self._job_from_data(data)
                with self.lock:
                    if job.id in self.deleted: continue
                    existing = self.jobs.get(job.id)
                    if existing is not None:
                        continue
                    self.jobs[job.id] = job
            except (KeyError, TypeError, ValueError):
                continue

    def restore_local(self) -> None:
        """Carrega os manifestos persistidos em OUTPUTS para reexibir o histórico
        de imagens imediatamente na inicialização, sem esperar pela reconexão do MEGA."""
        for path in sorted((p for p in OUTPUTS.glob("*.json") if not p.name.endswith(".workflow.json")), key=lambda item: item.stat().st_mtime, reverse=True):
            if path.name == LAST_SETTINGS_NAME:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not data.get("id"):
                    continue
                job = self._job_from_data(data)
                with self.lock:
                    if job.id in self.jobs or job.id in self.deleted:
                        continue
                    if job.status in {"queued", "running"}:
                        job.status, job.progress_phase = "interrupted", "interrupted"
                        job.error = "A sessão anterior foi interrompida. Reenvie os parâmetros para continuar."
                    self.jobs[job.id] = job
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def enqueue(self, params: GenerationParams, idempotency_key: str | None = None, request_fingerprint: str | None = None) -> Job:
        fingerprint = request_fingerprint or hashlib.sha256(json.dumps(saved_settings(params), sort_keys=True).encode()).hexdigest()
        with self.lock:
            if idempotency_key and len(idempotency_key) > 128: raise ValueError("Chave de envio inválida.")
            if idempotency_key and idempotency_key in self.idempotency:
                old_fingerprint, job_id = self.idempotency[idempotency_key]
                if old_fingerprint != fingerprint:
                    raise ValueError("Esta chave já foi usada com outros parâmetros.")
                return self.jobs[job_id]
            active = sum(job.status in {"queued", "running"} for job in self.jobs.values())
            if active >= int(os.environ.get("STUDIO_MAX_QUEUE", "8")):
                raise ValueError("A fila está cheia. Aguarde ou cancele um job.")
            job = Job(id=str(uuid.uuid4()), created_at=now_iso(), status="queued", progress=0, params=params, updated_at=now_iso())
            self.jobs[job.id] = job
            if idempotency_key:
                if len(self.idempotency) > 512: self.idempotency.pop(next(iter(self.idempotency)))
                self.idempotency[idempotency_key] = (fingerprint, job.id)
            atomic_json(artifact_path(OUTPUTS, job.id, ".json"), job.public())
            self.pending.put(job.id)
            return job

    def _persist_local(self, job: Job) -> None:
        if job.id not in self.deleted:
            atomic_json(artifact_path(OUTPUTS, job.id, ".json"), job.public())

    def schedule_sync(self, job_id: str) -> None:
        with self.lock:
            if job_id not in self.sync_scheduled and job_id not in self.deleted:
                self.sync_scheduled.add(job_id)
                self.sync_queue.put(job_id)

    def _sync_run(self) -> None:
        while not self.stop.is_set():
            try: job_id = self.sync_queue.get(timeout=1)
            except queue.Empty: continue
            try:
                if job_id == "preferences":
                    self.archive.sync_last_settings()
                    if self.archive.available:
                        for name in ("presets.json", "favorites.json"):
                            path = OUTPUTS / name
                            if path.exists(): self.archive._upload(path)
                    continue
                job = self.get(job_id)
                if job and job.id not in self.deleted:
                    self._persist_job(job, artifact_path(OUTPUTS, job.id, ".png") if job.filename else None)
            except Exception as exc:
                job = self.get(job_id)
                if job:
                    job.sync_status, job.sync_error = "failed", str(exc)[:250]
                    self._persist_local(job)
            finally:
                with self.lock: self.sync_scheduled.discard(job_id)
                self.sync_queue.task_done()
                retry = self.get(job_id)
                if retry and retry.sync_status == "failed" and retry.sync_attempts < 3:
                    timer = threading.Timer(5 * 2 ** retry.sync_attempts, lambda ident=job_id: None if self.stop.is_set() else self.schedule_sync(ident))
                    timer.daemon = True; timer.start()

    def cancel(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job: raise ValueError("Job não encontrado.")
            if job.status not in {"queued", "running"}: return job
            job.cancel_requested = True
            if job.status == "queued":
                job.status, job.progress_phase = "cancelled", "cancelled"
            else:
                self.engine.comfy.cancel_event.set()
            self._persist_local(job)
            return job

    def close(self) -> None:
        self.stop.set()
        self.engine.comfy.cancel_event.set()
        self.engine.comfy.close()
        if self.sync_worker.is_alive(): self.sync_worker.join(timeout=3)

    def _update(
        self,
        job: Job,
        progress: int,
        vram: float | None,
        *,
        download: int | None = None,
        pipeline: int | None = None,
        phase: str | None = None,
    ) -> None:
        with self.lock:
            job.progress, job.vram_gb, job.updated_at = progress, vram, now_iso()
            if download is not None:
                job.download_progress = max(0, min(100, int(download)))
            if pipeline is not None:
                job.pipeline_progress = max(0, min(100, int(pipeline)))
            if phase is not None:
                job.progress_phase = phase

    def _persist_job(self, job: Job, image: Path | None) -> None:
        with self.job_locks[job.id]:
            if job.id in self.deleted: return
            with self.lock:
                job.sync_status, job.sync_error = "uploading", None
                job.sync_attempts += 1
                snapshot = replace(job, sync_status="synced", mega_synced=True)
                self._persist_local(job)
            synced = self.archive.save_job(snapshot, image)
            with self.lock:
                if job.id in self.deleted: return
                job.mega_synced = synced
                job.sync_status = "synced" if synced else "failed"
                job.sync_error = None if synced else (self.archive.error or "MEGA indisponível; arquivo local preservado.")
                job.updated_at = now_iso()
                self._persist_local(job)
            if synced:
                self.archive.sync_last_settings()

    def _run(self) -> None:
        while not self.stop.is_set():
            try: job_id = self.pending.get(timeout=1)
            except queue.Empty: continue
            with self.lock:
                job = self.jobs.get(job_id)
                if not job or job.cancel_requested:
                    self.pending.task_done()
                    continue
                job.status, job.updated_at = "running", now_iso()
                job.progress_phase = "starting"
                self._persist_local(job)
            try:
                image = self.engine.generate(
                    job,
                    lambda progress, vram, **stages: self._update(job, progress, vram, **stages),
                )
                with self.lock:
                    job.filename = image.name
                    job.status, job.progress, job.download_progress, job.pipeline_progress, job.progress_phase, job.completed_at, job.updated_at = "completed", 100, 100, 100, "completed", now_iso(), now_iso()
                    job.mega_synced = False
                self._persist_local(job)
                self.schedule_sync(job.id)
            except Exception as exc:
                with self.lock:
                    error_text = str(exc)
                    if len(error_text) > 1_500:
                        error_text = error_text[:300] + "\n... [log truncado] ...\n" + error_text[-1_150:]
                    job.status, job.error, job.progress_phase, job.updated_at = ("cancelled", None, "cancelled", now_iso()) if job.cancel_requested or isinstance(exc, InterruptedError) else ("failed", error_text, "failed", now_iso())
                    job.mega_synced = False
                self._persist_local(job)
            finally:
                if job.params.source_image: Path(job.params.source_image).unlink(missing_ok=True)
                self.pending.task_done()

    def sync_pending(self) -> tuple[int, int]:
        """Reenvia PNGs e manifestos locais de jobs concluídos ainda pendentes."""
        with self.lock:
            pending = [
                job for job in self.jobs.values()
                if job.status == "completed" and not job.mega_synced
            ]
        for job in pending:
            if artifact_path(OUTPUTS, job.id, ".png").exists():
                self.schedule_sync(job.id)
        return len(pending), 0

    def public_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
            queued = [job.id for job in reversed(jobs) if job.status == "queued"]
            result = []
            for job in jobs:
                payload = job.public()
                payload["queue_position"] = queued.index(job.id) + 1 if job.id in queued else 0
                result.append(payload)
            return result

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def remove(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.pop(job_id, None)


archive = MegaArchive()
engine = GeneratorEngine()
manager = JobManager(archive, engine)
# Restaura o histórico de imagens do disco na própria inicialização, para que o
# /api/bootstrap já entregue a galeria sem esperar pela reconexão do MEGA.
manager.restore_local()
atexit.register(manager.close)
archive_ready = threading.Event()
archive_restore_lock = threading.Lock()


def restore_archive() -> None:
    """Tenta restaurar o histórico do MEGA com retéries, para não falhar de forma
    definitiva por um erro transitório da API. Serializado para não competir com a
    preparação inicial do arquivo."""
    with archive_restore_lock:
        for attempt in range(1, 4):
            try:
                if not archive.available: raise RuntimeError("MEGA indisponível")
                manager.restore()
                return
            except Exception as exc:
                archive.error = f"Falha ao restaurar do MEGA (tentativa {attempt}): {str(exc)[:160]}"
                if attempt < 3:
                    time.sleep(min(2 * attempt, 6))
                else: raise


_archive_heal_lock = threading.Lock()

def _heal_archive_restore() -> None:
    if not _archive_heal_lock.acquire(blocking=False): return
    try:
        _heal_archive_restore_once()
    finally:
        _archive_heal_lock.release()

def _heal_archive_restore_once() -> None:
    """Continua tentando restaurar em segundo plano até obter sucesso ou esgotar
    as tentativas, cobrindo indisponibilidade/erro transitório do MEGA e mesclando
    registros que existem apenas no arquivo remoto."""
    for attempt in range(1, 7):
        if attempt > 1:
            time.sleep(15)
        if not archive.available:
            try:
                archive.connect()
            except Exception:
                continue
        if not archive.available:
            continue
        try:
            restore_archive()
            return
        except Exception:
            continue


def initialize_archive() -> None:
    """Prepara o MEGA em segundo plano para não bloquear a abertura do servidor."""
    try:
        archive.connect()
        if archive.available:
            restore_archive()
            archive.load_last_settings()
            archive.restore_preferences()
    except Exception as exc:
        archive.error = str(exc)[:200]
    finally:
        archive_ready.set()
    # Se a restauração inicial falhou por erro transitório, segue tentando em
    # segundo plano para que o histórico apareça sozinho assim que o MEGA responder.
    threading.Thread(target=_heal_archive_restore, name="archive-restore-healer", daemon=True).start()


if os.environ.get("STUDIO_START_WORKERS", "1") == "1":
    threading.Thread(target=initialize_archive, name="archive-initializer", daemon=True).start()


def validate_params(raw: dict[str, Any], source_image: str | None) -> GenerationParams:
    if not isinstance(raw, dict): raise ValueError("Envie os parâmetros como um objeto JSON.")
    mode = raw.get("mode", "text2img")
    if mode not in {"text2img", "img2img"}:
        raise ValueError("Modo de geração inválido.")
    edit_level = str(raw.get("edit_level", "medium")).strip().lower()
    if edit_level not in {"low", "medium", "high"}:
        raise ValueError("Nível de edição inválido. Use baixo, médio ou alto.")
    requested_model = str(raw.get("model") or DEFAULT_MODEL_ID).strip().lower()
    if requested_model not in MODEL_SPECS:
        raise ValueError("Checkpoint não reconhecido. Selecione um perfil disponível no ModelLab.")
    model_spec = get_model_spec(requested_model)
    family = str(model_spec.get("family", "sdxl")).strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"A família {family!r} não está habilitada neste motor.")
    engine = str(model_spec.get("engine") or family_profile(family).get("engine", "unsupported"))
    if engine != "comfyui":
        raise ValueError(f"O modelo selecionado pertence à família {family}, mas esse engine ainda não está configurado no ModelLab.")
    defaults = model_spec.get("defaults", {})
    sampler = str(raw.get("sampler") or defaults.get("sampler", "euler_a")).strip().lower()
    if sampler not in SUPPORTED_SAMPLERS:
        raise ValueError("Sampler não suportado pelo perfil atual.")
    prompt = str(raw.get("prompt", "")).strip()
    if not prompt or len(prompt) > 4000:
        raise ValueError("Informe um prompt entre 1 e 4000 caracteres.")
    try:
        upscale = float(raw.get("upscale", 1))
        if upscale not in {1.0, 1.5, 2.0}: raise ValueError("Ampliação inválida.")
        seed = int(raw.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        steps = int(raw.get("steps", defaults.get("steps", 28)))
        width = int(raw.get("width", 1024))
        height = int(raw.get("height", 1024))
        guidance = float(raw.get("guidance", defaults.get("guidance", 6.5)))
        strength = float(raw.get("strength", defaults.get("strength", 0.65)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Há um parâmetro numérico inválido.") from exc
    if not 10 <= steps <= 60 or not 1 <= guidance <= 15 or not 0.05 <= strength <= 1:
        raise ValueError("Steps, guidance ou strength estão fora dos limites aceitos.")
    size_min, size_max = 512, 1024
    if width not in range(size_min, size_max + 1, 64) or height not in range(size_min, size_max + 1, 64):
        raise ValueError(f"Largura e altura devem ser múltiplos de 64 entre {size_min} e {size_max}.")
    if mode == "img2img" and not source_image:
        raise ValueError("Envie uma imagem-base para usar img2img.")
    if not 0 <= seed <= 2**53 - 1: raise ValueError("Seed deve estar entre 0 e 2^53−1 para preservar precisão no navegador.")
    if not math.isfinite(guidance) or not math.isfinite(strength): raise ValueError("Os números devem ser finitos.")
    candidates = raw.get("loras", [])
    if not isinstance(candidates, list) or len(candidates) > MAX_LORAS:
        raise ValueError(f"Envie uma lista com até {MAX_LORAS} LoRAs.")
    parsed_loras: list[LoRASelection] = []
    for candidate in candidates:
        if not isinstance(candidate, dict): raise ValueError("A seleção de LoRA é inválida.")
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
        mode=mode, model_id=requested_model, sampler=sampler, loras=parsed_loras,
        source_image=source_image, edit_level=edit_level, upscale=upscale,
    )


@app.route("/")
def index():
    page = "index.html" if session.get("authenticated") else "login.html"
    response = send_file(Path(app.static_folder) / page)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict): return jsonify({"error": "Login inválido."}), 400
    # Only trust the tunnel's forwarded IP when explicitly configured.
    key = request.headers.get("CF-Connecting-IP", request.remote_addr or "local") if os.environ.get("STUDIO_TRUST_TUNNEL") == "1" else (request.remote_addr or "local")
    if not login_limiter.allow(key):
        return jsonify({"error": "Muitas tentativas. Aguarde cinco minutos."}), 429
    submitted = str(payload.get("password", ""))
    configured = os.environ.get("STUDIO_PASSWORD", "")
    if not configured or not hmac.compare_digest(submitted.encode(), configured.encode()):
        return jsonify({"error": "Senha inválida."}), 401
    session.clear()
    session.permanent = True
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
    # Jobs e preferências já são restaurados pelo inicializador em segundo plano.
    # O helper usa cache local primeiro, portanto a recarga nunca espera pelo MEGA.
    last_settings = archive.load_last_settings(allow_remote=False)
    return jsonify({
        "csrf": session.get("csrf"), "jobs": manager.public_jobs()[:40],
        "archive": {
            "available": archive.available, "ready": archive_ready.is_set(),
            "error": archive.error, "folder": MEGA_FOLDER,
        },
        "last_settings": last_settings,
        "last_settings_source": "local" if last_settings else None,
        "presets": preference_list("presets.json"), "favorites": preference_list("favorites.json"),
        "limits": {"maxLoras": MAX_LORAS, "sizes": list(range(512, 1025, 64))},
        "models": [public_model_spec(spec) for spec in MODEL_SPECS.values()],
        "model": public_model_spec(get_model_spec()),
    })


@app.route("/api/comfy-health")
@authentication_required
def comfy_health():
    return jsonify({
        "backend": "comfyui-headless",
        "model": public_model_spec(get_model_spec()),
        "comfy": engine.comfy.status(),
    })


@app.route("/api/model-catalog")
@authentication_required
def model_catalog():
    include_adult = request.args.get("include_adult", "").strip().lower() in {"1", "true", "yes"}
    if include_adult and not os.environ.get("CIVITAI_TOKEN", "").strip():
        return jsonify({"error": "Defina CIVITAI_TOKEN no servidor para consultar conteúdo adulto autorizado."}), 400
    try:
        limit = min(max(int(request.args.get("limit", 24)), 1), 48)
    except (TypeError, ValueError):
        limit = 24
    params: dict[str, Any] = {
        "limit": limit, "types": "Checkpoint", "sort": request.args.get("sort", "Most Downloaded"),
        "period": request.args.get("period", "AllTime"), "primaryFileOnly": "true",
        "nsfw": "true" if include_adult else "false",
    }
    if request.args.get("cursor"):
        params["cursor"] = request.args["cursor"]
    if request.args.get("query"):
        params["query"] = request.args["query"][:120]
    if request.args.get("tag"):
        params["tag"] = request.args["tag"][:80]
    family_filter = request.args.get("family", "").strip().lower()
    if family_filter in SUPPORTED_MODEL_FAMILIES:
        params["baseModels"] = civitai_base_for_family(family_filter)
    try:
        response = requests.get(f"{CIVITAI_BASE}/models", params=params, headers=civitai_headers(), timeout=25)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return jsonify({"error": f"Não foi possível consultar a loja de modelos Civitai: {str(exc)[:160]}"}), 502

    items: list[dict[str, Any]] = []
    for model in payload.get("items", []):
        versions = [version for version in model.get("modelVersions", []) if version.get("modelType", "Checkpoint") == "Checkpoint"]
        if family_filter in SUPPORTED_MODEL_FAMILIES:
            versions = [version for version in versions if normalize_model_family(version.get("baseModel") or model.get("name")) == family_filter]
        if not versions:
            continue
        version = versions[0]
        files = version.get("files") or []
        checkpoint_file = next((item for item in files if str(item.get("type", "")).lower() == "model" and str(item.get("name", "")).lower().endswith((".safetensors", ".ckpt"))), None)
        inferred_family = normalize_model_family(version.get("baseModel") or model.get("name"))
        profile = family_profile(inferred_family)
        version_id = int(version.get("id") or 0)
        model_numeric_id = int(model.get("id") or 0)
        internal_id = re.sub(r"[^a-z0-9._-]+", "-", f"civitai-{model_numeric_id}-{version_id}".lower()).strip("-")
        preview = next((image.get("url") for image in version.get("images", []) if image.get("url")), None)
        items.append({
            "id": internal_id, "civitai_model_id": model_numeric_id, "version_id": version_id,
            "name": model.get("name") or internal_id, "version": version.get("name") or f"Versão {version_id}",
            "creator": (model.get("creator") or {}).get("username"), "base_model": version.get("baseModel"),
            "family": inferred_family, "engine": profile.get("engine"), "image": preview,
            "downloads": (version.get("stats") or {}).get("downloadCount", 0), "mature": bool(model.get("nsfw")),
            "cached": bool(checkpoint_file and (MODELS / f"{internal_id}.safetensors").exists()),
            "file": checkpoint_file.get("name") if checkpoint_file else None,
            "notes": profile.get("notes", ""), "defaults": profile.get("defaults", {}),
        })
    return jsonify({
        "items": items, "next_cursor": (payload.get("metadata") or {}).get("nextCursor"),
        "family": family_filter or "all", "base_model": civitai_base_for_family(family_filter) if family_filter else "all",
        "includes_adult": include_adult, "catalog_query": {"authenticated": bool(os.environ.get("CIVITAI_TOKEN", "").strip())},
    })


@app.route("/api/model-profile", methods=["POST"])
@authentication_required
@csrf_required
def model_profile():
    payload = request.get_json(silent=True) or {}
    try:
        if not isinstance(payload, dict): raise ValueError("Perfil inválido.")
        version_id = int(payload.get("version_id"))
        civitai_model_id = int(payload.get("civitai_model_id"))
        data = version_metadata(version_id)
        resource = model_file(data, kind="Checkpoint", model_id=civitai_model_id,
            file_id=int(payload["file_id"]) if payload.get("file_id") else None)
        family = "anima"
        profile = family_profile(family)
        model_id = f"civitai-{civitai_model_id}-{version_id}"
        spec = {
            "id": model_id, "name": str((data.get("model") or {}).get("name") or data.get("name") or model_id)[:120],
            **resource, "path": str(MODELS / "diffusion_models" / f"{model_id}.safetensors"),
            "family": family, "base": str(data.get("baseModel")), "engine": "comfyui",
            "lora_base": "Anima", "defaults": dict(profile["defaults"]), "notes": profile["notes"],
            "civitai_model_id": civitai_model_id, "version_id": version_id,
        }
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except requests.RequestException:
        return jsonify({"error": "Não foi possível confirmar a versão no Civitai. Tente novamente."}), 502
    if model_id not in MODEL_SPECS and len(MODEL_SPECS) >= 100:
        return jsonify({"error": "Limite de 100 perfis atingido."}), 400
    MODEL_SPECS[model_id] = spec
    try:
        cached_profiles = [candidate for key, candidate in MODEL_SPECS.items() if key != DEFAULT_MODEL_ID]
        atomic_json(MODEL_PROFILE_CACHE, cached_profiles)
    except OSError:
        pass
    return jsonify(public_model_spec(spec))


@app.route("/api/catalog")
@authentication_required
def catalog():
    include_adult = request.args.get("include_adult", "").strip().lower() in {"1", "true", "yes"}
    if include_adult and not os.environ.get("CIVITAI_TOKEN", "").strip():
        return jsonify({"error": "Defina CIVITAI_TOKEN no servidor para consultar conteúdo adulto autorizado."}), 400
    family = request.args.get("family", "anima").strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        family = "anima"
    try:
        start_day, end_day = _request_date_range(request.args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    sort = request.args.get("sort", "Most Downloaded")
    if sort not in {"Most Downloaded", "Highest Rated", "Newest", "Oldest"}:
        sort = "Most Downloaded"
    period = request.args.get("period", "AllTime")
    if period not in {"AllTime", "Year", "Month", "Week", "Day"}:
        period = "AllTime"
    base_filter = request.args.get("base_filter", "compatible").strip().lower()
    compatible_only = base_filter not in {"0", "false", "no", "all"}
    try:
        requested_limit = int(request.args.get("limit", 48))
    except (TypeError, ValueError):
        requested_limit = 48
    params: dict[str, Any] = {
        "limit": min(max(requested_limit, 1), 100), "types": "LORA",
        "sort": sort, "period": period, "primaryFileOnly": "true",
        "nsfw": "true" if include_adult else "false",
    }
    # "Anima" não é garantidamente um valor do enum BaseModel em todas as
    # versões da API. Para não transformar uma base nova em resposta vazia,
    # buscamos todos os LoRAs e filtramos a compatibilidade localmente.
    if compatible_only and family != "anima":
        params["baseModels"] = civitai_base_for_family(family)
    if request.args.get("cursor"):
        params["cursor"] = request.args["cursor"]
    query = request.args.get("query", "").strip()[:120]
    if query:
        # Tenta interpretar como ID ou URL de modelo antes de busca textual.
        match = re.search(r"(?:models/|^)(\d+)", query)
        if match:
            params["ids"] = match.group(1)
        else:
            params["query"] = query
    if request.args.get("tag"):
        params["tag"] = request.args["tag"][:80]

    fallback_used = False
    try:
        def get_models(page_params: dict[str, Any]) -> dict[str, Any]:
            response = requests.get(f"{CIVITAI_BASE}/models", params=page_params, headers=civitai_headers(), timeout=25)
            response.raise_for_status()
            return response.json()

        active_params = dict(params)
        payload = get_models(active_params)
        # Se uma busca textual (sem ID) retornar vazia e não for paginação,
        # tenta relaxar o filtro de família, pois o criador pode ter marcado a
        # base de forma genérica no Civitai.
        if not payload.get("items") and "query" in params and not params.get("cursor"):
            fallback_params = dict(params)
            fallback_params.pop("baseModels", None)
            fallback_payload = get_models(fallback_params)
            if fallback_payload.get("items"):
                payload = fallback_payload
                active_params = fallback_params
                fallback_used = True

        # O filtro de família/data é parcialmente local. Para Anima, a API
        # não oferece um baseModel confiável; por isso o primeiro lote bruto
        # pode conter somente poucos resultados compatíveis. Continue lendo
        # páginas até preencher a página visual solicitada.
        model_items = list(payload.get("items", []))
        next_cursor = (payload.get("metadata") or {}).get("nextCursor")
        seen_cursors = {str(next_cursor)} if next_cursor else set()
        local_filtering = compatible_only or start_day is not None or end_day is not None

        def visible_model_count(models: list[dict[str, Any]]) -> int:
            count = 0
            for candidate_model in models:
                candidate_versions = [
                    item for item in candidate_model.get("modelVersions", []) if isinstance(item, dict)
                ]
                if compatible_only:
                    candidate_versions = [
                        item for item in candidate_versions
                        if version_matches_family(item, family, str(candidate_model.get("name") or ""))
                    ]
                if start_day is not None or end_day is not None:
                    candidate_versions = [
                        item for item in candidate_versions
                        if _matches_date_range(item.get("publishedAt") or item.get("createdAt"), start_day, end_day)
                    ]
                if not candidate_versions and ("query" in params or "ids" in params) and not (start_day or end_day):
                    candidate_versions = [
                        item for item in candidate_model.get("modelVersions", []) if isinstance(item, dict)
                    ]
                if candidate_versions:
                    count += 1
            return count

        max_pages = 8 if local_filtering else 3
        for _ in range(max_pages):
            enough_results = visible_model_count(model_items) >= requested_limit if local_filtering else len(model_items) >= requested_limit
            if not next_cursor or enough_results:
                break
            page_params = dict(active_params)
            page_params["cursor"] = next_cursor
            page = get_models(page_params)
            model_items.extend(page.get("items", []))
            next_cursor = (page.get("metadata") or {}).get("nextCursor")
            if not next_cursor or str(next_cursor) in seen_cursors:
                break
            seen_cursors.add(str(next_cursor))
        payload = {"items": model_items, "metadata": {"nextCursor": next_cursor}}
    except requests.RequestException as exc:
        return jsonify({"error": f"Não foi possível consultar o catálogo Civitai: {str(exc)[:160]}"}), 502
    items = []
    for model in payload.get("items", []):
        all_versions = [item for item in model.get("modelVersions", []) if isinstance(item, dict)]
        versions = all_versions if not compatible_only else [
            item for item in all_versions if version_matches_family(item, family, str(model.get("name") or ""))
        ]
        if start_day is not None or end_day is not None:
            versions = [
                item for item in versions
                if _matches_date_range(item.get("publishedAt") or item.get("createdAt"), start_day, end_day)
            ]
        # Uma busca textual/por ID deve priorizar o resultado oficial solicitado.
        # Se a API não trouxe baseModel ou usou um rótulo novo, não escondemos o
        # modelo: exibimos as versões publicadas e sinalizamos a busca ampliada.
        if not versions and ("query" in params or "ids" in params) and all_versions and not (start_day or end_day):
            versions = all_versions
            fallback_used = True
        if not versions:
            continue
        if sort == "Oldest":
            versions = sorted(versions, key=lambda item: _published_day(item.get("publishedAt") or item.get("createdAt")) or datetime.min.date())
        else:
            versions = sorted(versions, key=lambda item: _published_day(item.get("publishedAt") or item.get("createdAt")) or datetime.min.date(), reverse=True)
        version = versions[0]
        image = next((item.get("url") for item in version.get("images", []) if item.get("url")), None)
        version_items = []
        for candidate in versions:
            candidate_image = next((item.get("url") for item in candidate.get("images", []) if item.get("url")), None)
            version_items.append({
                "id": candidate.get("id"), "name": candidate.get("name") or f"Versão {candidate.get('id')}",
                "base_model": candidate.get("baseModel") or "não informado",
                "image": candidate_image, "downloads": candidate.get("stats", {}).get("downloadCount", 0),
                "created_at": candidate.get("createdAt"), "updated_at": candidate.get("updatedAt"),
            })
        items.append({
            "id": model.get("id"), "name": model.get("name"), "creator": model.get("creator", {}).get("username"),
            "tags": model.get("tags", [])[:10], "version_id": version.get("id"), "version": version.get("name"),
            "versions": version_items, "image": image, "downloads": version.get("stats", {}).get("downloadCount", 0), "mature": bool(model.get("nsfw")),
        })
    # O backend pode ter lido várias páginas para compensar o filtro local,
    # mas a página visual continua limitada ao lote solicitado pelo cliente.
    items = items[:requested_limit]
    return jsonify({
        "items": items, "next_cursor": payload.get("metadata", {}).get("nextCursor"),
        "family": family, "base_model": civitai_base_for_family(family),
        "includes_adult": include_adult,
        "catalog_query": {"nsfw": params["nsfw"], "authenticated": bool(os.environ.get("CIVITAI_TOKEN", "").strip())},
        "fallback_used": fallback_used,
        "compatible_only": compatible_only,
        "date_from": request.args.get("date_from", ""), "date_to": request.args.get("date_to", ""),
    })


@app.route("/api/prompt-store")
@authentication_required
def prompt_store():
    include_adult = request.args.get("include_adult", "").strip().lower() in {"1", "true", "yes"}
    if include_adult and not os.environ.get("CIVITAI_TOKEN", "").strip():
        return jsonify({"error": "Defina CIVITAI_TOKEN no servidor para consultar conteúdo adulto autorizado."}), 400
    sort = request.args.get("sort", "Most Reactions")
    if sort not in {"Most Reactions", "Random", "Newest", "Oldest"}:
        sort = "Most Reactions"
    period = request.args.get("period", "AllTime")
    if period not in {"AllTime", "Year", "Month", "Week", "Day"}:
        period = "AllTime"
    try:
        start_day, end_day = _request_date_range(request.args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    family = request.args.get("family", "anima").strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        family = "anima"
    compatible_only = request.args.get("base_filter", "compatible").strip().lower() not in {"0", "false", "no", "all"}
    search = request.args.get("query", "").strip().lower()[:120]
    tag_search = request.args.get("tag", "").strip().lower()[:80]
    filters = [term.strip().lower() for term in request.args.get("filters", "").split(",") if term.strip()][:8]
    try:
        limit = min(max(int(request.args.get("limit", 24)), 1), 100)
    except (TypeError, ValueError):
        limit = 24

    params: dict[str, Any] = {
        "limit": limit,
        "sort": sort,
        "period": period,
        "type": "image",
        "withMeta": "true",
        "nsfw": "true" if include_adult else "false",
    }
    if compatible_only:
        params["baseModels"] = civitai_base_for_family(family)
    username = request.args.get("username", "").strip()[:80]
    if username:
        params["username"] = username
    if request.args.get("cursor"):
        params["cursor"] = request.args["cursor"]

    try:
        def get_images(page_params: dict[str, Any]) -> dict[str, Any]:
            response = requests.get(
                f"{CIVITAI_BASE}/images",
                params=page_params,
                headers=civitai_headers(),
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}

        def prompt_item(image: dict[str, Any]) -> dict[str, Any] | None:
            raw_meta = image.get("meta")
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            prompt = str(meta.get("prompt") or meta.get("Prompt") or "").strip()
            negative_prompt = str(meta.get("negativePrompt") or meta.get("Negative prompt") or "").strip()

            resource_candidates: list[Any] = []
            for resource_key in ("civitaiResources", "resources"):
                candidate_resources = meta.get(resource_key) or []
                if isinstance(candidate_resources, dict):
                    candidate_resources = [candidate_resources]
                if isinstance(candidate_resources, list):
                    resource_candidates.extend(candidate_resources)
            resources: list[dict[str, Any]] = []
            seen_resource_keys: set[tuple[str, str, str]] = set()
            for resource in resource_candidates:
                if not isinstance(resource, dict):
                    continue
                version_raw = resource.get("modelVersionId") or resource.get("versionId") or resource.get("model_version_id") or resource.get("version_id")
                if version_raw:
                    resource_key = ("version", str(version_raw), str(resource.get("type", "")).lower())
                else:
                    resource_key = ("raw", json.dumps(resource, sort_keys=True, default=str), "")
                if resource_key in seen_resource_keys:
                    continue
                seen_resource_keys.add(resource_key)
                resources.append(resource)

            loras: list[dict[str, Any]] = []
            seen_lora_versions: set[int] = set()
            for resource in resources:
                if str(resource.get("type", "")).lower() != "lora":
                    continue
                try:
                    version_id = int(resource.get("modelVersionId") or resource.get("versionId") or resource.get("model_version_id") or resource.get("version_id"))
                except (TypeError, ValueError):
                    continue
                if version_id <= 0 or version_id in seen_lora_versions:
                    continue
                seen_lora_versions.add(version_id)
                try:
                    model_id = int(resource.get("modelId") or resource.get("model_id")) if (resource.get("modelId") or resource.get("model_id")) else None
                except (TypeError, ValueError):
                    model_id = None
                raw_weight = resource.get("weight", resource.get("strength", 0.8))
                try:
                    weight = float(raw_weight if raw_weight is not None else 0.8)
                except (TypeError, ValueError):
                    weight = 0.8
                if weight != weight:
                    weight = 0.8
                weight = max(0.0, min(1.5, weight))
                name = str(resource.get("modelName") or resource.get("modelVersionName") or resource.get("name") or f"LoRA // {version_id}").strip()[:120]
                loras.append({
                    "version_id": version_id,
                    "model_id": model_id,
                    "name": name or f"LoRA // {version_id}",
                    "weight": weight,
                })

            raw_tags = image.get("tags") or []
            tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
            family_hints = [
                image.get("baseModel"),
                meta.get("Model type"),
                meta.get("modelType"),
                meta.get("baseModel"),
            ]
            family_hints.extend(
                resource.get(key)
                for resource in resources
                for key in ("baseModel", "modelName", "modelVersionName")
            )
            resource_families = {
                normalize_model_family(str(hint))
                for hint in family_hints
                if hint is not None and str(hint).strip()
            }
            if compatible_only and resource_families and family not in resource_families:
                return None

            created_at = image.get("createdAt")
            if not _matches_date_range(created_at, start_day, end_day):
                return None
            resource_text = " ".join(
                str(resource.get(key) or "")
                for resource in resources
                for key in ("modelName", "modelVersionName", "baseModel")
            )
            haystack = " ".join([
                prompt,
                negative_prompt,
                str(image.get("username", "")),
                " ".join(tags),
                str(image.get("baseModel", "")),
                resource_text,
            ]).lower()
            if search and search not in haystack:
                return None
            if tag_search and tag_search not in " ".join(tags).lower():
                return None
            if filters and not all(term in haystack for term in filters):
                return None

            size = str(meta.get("Size", ""))
            size_parts = re.split(r"[xX×]", size, maxsplit=1)
            width = image.get("width") or (size_parts[0] if len(size_parts) == 2 else None)
            height = image.get("height") or (size_parts[1] if len(size_parts) == 2 else None)
            stats = image.get("stats") if isinstance(image.get("stats"), dict) else {}
            return {
                "id": image.get("id"), "image": image.get("url"), "prompt": prompt,
                "negative_prompt": negative_prompt, "seed": meta.get("seed"),
                "steps": meta.get("steps"), "guidance": meta.get("cfgScale") or meta.get("guidanceScale"),
                "width": width, "height": height,
                "username": image.get("username"), "created_at": created_at,
                "nsfw": bool(image.get("nsfw")), "nsfw_level": image.get("nsfwLevel"),
                "tags": tags[:12], "loras": loras[:MAX_LORAS],
                "reactions": stats.get("heartCount", 0) or 0,
            }

        active_params = dict(params)
        next_cursor = None
        matched_items: list[dict[str, Any]] = []
        seen_cursors: set[str] = set()
        seen_images: set[str] = set()
        local_filtering = compatible_only or start_day is not None or end_day is not None or bool(search or tag_search or filters)
        max_pages = 8 if local_filtering else 1
        for page_number in range(max_pages):
            page = get_images(active_params)
            raw_items = page.get("items") or []
            if not isinstance(raw_items, list):
                raw_items = []
            for image in raw_items:
                if not isinstance(image, dict):
                    continue
                image_key = str(image.get("id") or image.get("url") or json.dumps(image, sort_keys=True, default=str))
                if image_key in seen_images:
                    continue
                seen_images.add(image_key)
                item = prompt_item(image)
                if item is not None:
                    matched_items.append(item)
            next_cursor = (page.get("metadata") or {}).get("nextCursor")
            if not next_cursor or len(matched_items) >= limit or page_number + 1 >= max_pages:
                break
            next_cursor_key = str(next_cursor)
            if next_cursor_key in seen_cursors:
                break
            seen_cursors.add(next_cursor_key)
            active_params = dict(params)
            active_params["cursor"] = next_cursor
    except (requests.RequestException, ValueError, TypeError) as exc:
        return jsonify({"error": f"Não foi possível consultar a Loja de Prompts no Civitai: {str(exc)[:160]}"}), 502

    items = matched_items[:limit]
    if sort == "Random":
        random.shuffle(items)
    elif sort == "Oldest":
        items.sort(key=lambda item: _published_day(item.get("created_at")) or datetime.min.date())
    elif sort == "Newest":
        items.sort(key=lambda item: _published_day(item.get("created_at")) or datetime.min.date(), reverse=True)
    else:
        items.sort(key=lambda item: item.get("reactions", 0) or 0, reverse=True)
    return jsonify({
        "items": items, "next_cursor": next_cursor,
        "family": family, "base_model": civitai_base_for_family(family),
        "includes_adult": include_adult,
        "catalog_query": {"sort": sort, "period": period, "authenticated": bool(os.environ.get("CIVITAI_TOKEN", "").strip()), "randomized": sort == "Random"},
        "compatible_only": compatible_only, "date_from": request.args.get("date_from", ""), "date_to": request.args.get("date_to", ""),
    })


@app.route("/api/prompt-store/image")
@authentication_required
def prompt_store_image():
    try:
        raw = bounded_civitai_image(request.args.get("url", "").strip(), MAX_UPLOAD_BYTES)
        return send_file(io.BytesIO(raw), mimetype="image/jpeg", download_name="civitai-remix.jpg")
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    except requests.RequestException:
        return jsonify({"error": "Não foi possível baixar a imagem para remix."}), 502


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
                if image.width * image.height > 40_000_000: raise ValueError("Imagem grande demais.")
                image.verify()
            source = str(source_path)
        except Exception:
            source_path.unlink(missing_ok=True)
            return jsonify({"error": "Não foi possível validar a imagem-base enviada."}), 400
    try:
        params = validate_params(raw, source)
    except (ValueError, TypeError) as exc:
        if source: Path(source).unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    preferences_persisted = archive.save_last_settings(saved_settings(params), sync=False)
    try:
        job = manager.enqueue(params, request.headers.get("Idempotency-Key"), hashlib.sha256((json.dumps(raw, sort_keys=True) + (digest_file(Path(source)) if source else "")).encode()).hexdigest())
        if source and job.params.source_image != source: Path(source).unlink(missing_ok=True)
    except ValueError as exc:
        if source: Path(source).unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 409
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
    items = manager.public_jobs()
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(100, max(1, int(request.args.get("limit", 40))))
    except ValueError:
        return jsonify({"error": "Paginação inválida."}), 400
    return jsonify({"items": items[offset:offset + limit], "total": len(items), "next_offset": offset + limit if offset + limit < len(items) else None})


@app.route("/api/history/sync", methods=["POST"])
@authentication_required
@csrf_required
def history_sync():
    if not archive.available:
        threading.Thread(target=_heal_archive_restore, daemon=True).start()
    pending, synced = manager.sync_pending()
    last_settings_synced = False
    before = len(manager.jobs)
    return jsonify({
        "items": manager.public_jobs(),
        "restored": max(0, len(manager.jobs) - before),
        "pending": pending,
        "synced": synced,
        "last_settings_synced": last_settings_synced,
        "archive": {"available": archive.available, "error": archive.error, "folder": MEGA_FOLDER},
    })


@app.route("/api/history/<job_id>", methods=["DELETE"])
@authentication_required
@csrf_required
def delete_history(job_id: str):
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "Registro de histórico não encontrado."}), 404
    if job.status in {"queued", "running"}:
        return jsonify({"error": "Não é possível excluir um job enquanto ele está em execução."}), 409
    with manager.job_locks[job.id]:
        remote_required = bool(job.mega_synced or job.sync_attempts)
        if remote_required:
            if not archive.available:
                return jsonify({"error": "Conecte o MEGA para confirmar a exclusão remota."}), 503
            if not archive.delete_job(job.id):
                return jsonify({"error": archive.error or "A exclusão remota falhou."}), 502
        with manager.lock:
            manager.deleted.add(job.id)
            atomic_json(manager.tombstones_path, sorted(manager.deleted))
            manager.remove(job.id)
        for suffix in (".png", ".json", ".workflow.json", ".thumb.jpg"):
            artifact_path(OUTPUTS, job.id, suffix).unlink(missing_ok=True)
        # Delete the secondary ComfyUI output as well.
        for output in engine.comfy.output_dir.glob(f"modellab_{job.id}_*.png"):
            output.unlink(missing_ok=True)
        return jsonify({"ok": True, "id": job.id, "remote_deleted": remote_required})


@app.route("/api/history/<job_id>/image")
@authentication_required
def history_image(job_id: str):
    job = manager.get(job_id)
    if not job or not job.filename:
        return jsonify({"error": "Imagem não encontrada."}), 404
    image = artifact_path(OUTPUTS, job.id, ".png")
    if not image.exists():
        image = archive.restore_image(job_id) or image
    if not image.exists():
        return jsonify({"error": "A imagem não está disponível no cache nem no MEGA."}), 404
    return send_file(image, mimetype="image/png", as_attachment=request.args.get("download") == "1", download_name=image.name)


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
@authentication_required
@csrf_required
def cancel_job(job_id):
    try: return jsonify(manager.cancel(job_id).public())
    except ValueError as exc: return jsonify({"error": str(exc)}), 404


@app.route("/api/history/<job_id>/retry-sync", methods=["POST"])
@authentication_required
@csrf_required
def retry_sync(job_id):
    job = manager.get(job_id)
    if not job or job.status != "completed": return jsonify({"error": "Resultado não encontrado."}), 404
    manager.schedule_sync(job_id)
    return jsonify({"ok": True}), 202


@app.route("/api/history/<job_id>/thumbnail")
@authentication_required
def thumbnail(job_id):
    job = manager.get(job_id)
    if not job or job.status != "completed": return jsonify({"error": "Resultado não encontrado."}), 404
    path = artifact_path(OUTPUTS, job_id, ".thumb.jpg")
    if not path.exists():
        image_path = artifact_path(OUTPUTS, job_id, ".png")
        if not image_path.exists(): archive.restore_image(job_id)
        if not image_path.exists(): return jsonify({"error": "Imagem indisponível."}), 404
        with Image.open(image_path) as image:
            image.thumbnail((384, 384)); image.convert("RGB").save(path, "JPEG", quality=82)
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/history/<job_id>/export")
@authentication_required
def export_job(job_id):
    job = manager.get(job_id)
    if not job or job.status != "completed": return jsonify({"error": "Resultado não encontrado."}), 404
    image_path = artifact_path(OUTPUTS, job_id, ".png")
    if not image_path.exists(): archive.restore_image(job_id)
    if not image_path.exists(): return jsonify({"error": "Imagem indisponível."}), 404
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_STORED) as bundle:
        bundle.write(image_path, f"{job_id}.png")
        bundle.writestr("manifest.json", json.dumps(job.public(), ensure_ascii=False, indent=2))
        workflow = read_json(artifact_path(OUTPUTS, job_id, ".workflow.json"))
        if not workflow:
            with Image.open(image_path) as image: workflow = parse_json(image.info.get("prompt"), None)
        if workflow: bundle.writestr("workflow.json", json.dumps(workflow, ensure_ascii=False, indent=2))
    content.seek(0)
    return send_file(content, mimetype="application/zip", as_attachment=True, download_name=f"modellab-{job_id}.zip")


presets_lock = threading.RLock()
PRESETS_PATH = OUTPUTS / "presets.json"
FAVORITES_PATH = OUTPUTS / "favorites.json"

@app.route("/api/presets", methods=["GET", "POST"])
@authentication_required
def presets():
    if request.method == "GET": return jsonify({"items": preference_list("presets.json")})
    if not hmac.compare_digest(request.headers.get("X-CSRF-Token", "").encode(), session.get("csrf", "").encode()) or not session.get("csrf"):
        return jsonify({"error": "Token de segurança inválido."}), 403
    payload = request.get_json(silent=True) or {}
    try:
        if not isinstance(payload, dict): raise ValueError("Preset inválido.")
        name = str(payload.get("name", "")).strip()[:80]
        if not name: raise ValueError("Dê um nome ao preset.")
        settings = payload.get("settings")
        params = validate_params(settings, "preset-validation" if isinstance(settings, dict) and settings.get("mode") == "img2img" else None)
        with presets_lock:
            items = preference_list("presets.json")
            if len(items) >= 50: raise ValueError("Limite de 50 presets atingido.")
            item = {"id": str(uuid.uuid4()), "name": name, "settings": saved_settings(params)}
            items.append(item); atomic_json(PRESETS_PATH, items)
        manager.schedule_sync("preferences")
        return jsonify(item), 201
    except (TypeError, ValueError) as exc: return jsonify({"error": str(exc)}), 400


@app.route("/api/presets/<preset_id>", methods=["DELETE"])
@authentication_required
@csrf_required
def delete_preset(preset_id):
    with presets_lock:
        items = [item for item in preference_list("presets.json") if item.get("id") != preset_id]
        atomic_json(PRESETS_PATH, items)
    manager.schedule_sync("preferences")
    return jsonify({"ok": True})


@app.route("/api/history/<job_id>/favorite", methods=["POST"])
@authentication_required
@csrf_required
def favorite(job_id):
    if not manager.get(job_id): return jsonify({"error": "Resultado não encontrado."}), 404
    with presets_lock:
        favorites = set(preference_list("favorites.json"))
        if job_id in favorites: favorites.remove(job_id)
        else: favorites.add(job_id)
        atomic_json(FAVORITES_PATH, sorted(favorites))
    manager.schedule_sync("preferences")
    return jsonify({"favorite": job_id in favorites})


@app.route("/api/jobs/batch", methods=["POST"])
@authentication_required
@csrf_required
def batch_jobs():
    payload = request.get_json(silent=True) or {}
    try:
        if not isinstance(payload, dict): raise ValueError("Lote inválido.")
        count = int(payload.get("count", 3))
        if not 1 <= count <= 4: raise ValueError("O lote deve conter de 1 a 4 variações.")
        params = validate_params(payload.get("settings"), None)
        if params.mode != "text2img": raise ValueError("Variações em lote usam TXT→IMG.")
        with manager.lock:
            active = sum(job.status in {"queued", "running"} for job in manager.jobs.values())
            if active + count > int(os.environ.get("STUDIO_MAX_QUEUE", "8")): raise ValueError("Não há espaço para o lote na fila.")
            jobs = [manager.enqueue(replace(params, seed=(params.seed + i) % 2**53)).public() for i in range(count)]
        return jsonify({"items": jobs}), 202
    except (TypeError, ValueError) as exc: return jsonify({"error": str(exc)}), 400


@app.route("/api/models/cache", methods=["DELETE"])
@authentication_required
@csrf_required
def clear_model_cache():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("model") not in MODEL_SPECS:
        return jsonify({"error": "Modelo inválido."}), 400
    model_id = payload["model"]
    if not engine.load_lock.acquire(blocking=False):
        return jsonify({"error": "Aguarde o fim da geração para limpar o cache."}), 409
    try:
        with manager.lock:
            if engine.loaded_model_id == model_id or any(job.params.model_id == model_id and job.status in {"queued", "running"} for job in manager.jobs.values()):
                return jsonify({"error": "Esse modelo está carregado ou reservado na fila. Selecione um modelo sem uso."}), 409
            path = Path(MODEL_SPECS[model_id]["path"])
            if not path.resolve().is_relative_to(MODELS.resolve()) or path.is_symlink():
                return jsonify({"error": "Cache externo não pode ser removido pelo painel."}), 400
            size = path.stat().st_size if path.exists() else 0
            for suffix in ("", ".verified.json", ".part", ".part.json"):
                path.with_name(path.name + suffix).unlink(missing_ok=True)
        return jsonify({"freed_mb": round(size / 1024**2, 2)})
    finally:
        engine.load_lock.release()


@app.route("/api/diagnostics")
@authentication_required
def diagnostics():
    import importlib.metadata
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "pydantic", "pydantic-core", "aiohttp", "safetensors"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    disk = shutil.disk_usage(ROOT)
    return jsonify({"python": __import__("sys").version.split()[0], "packages": versions,
        "backend": engine.comfy.status(), "disk_free_gb": round(disk.free / 1024**3, 2),
        "queue": manager.pending.qsize(), "archive": {"available": archive.available, "error": archive.error}})


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
    signal.signal(signal.SIGTERM, lambda *_: __import__("sys").exit(0))
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
