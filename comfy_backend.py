"""Backend headless do ComfyUI para modelos Anima.

O módulo conversa com um processo local do ComfyUI pela API HTTP. A interface
web não é aberta. O processo é iniciado com ``--gpu-only`` e o workflow usa o
loader nativo do Anima, em vez de tentar carregar o checkpoint com Diffusers.

A limpeza executada pelo custom node só coleta temporários e esvazia o cache
ocioso do allocator CUDA; ela não chama ``unload_all_models`` nem o endpoint
``/free`` e, portanto, não descarrega o Anima residente.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image
from studio_storage import atomic_json, read_json
from studio_downloads import ensure_download


ProgressCallback = Callable[[int, float | None, int, str], None]


class ComfyBackend:
    """Cliente e supervisor de um ComfyUI local sem frontend."""

    def __init__(self, root: Path, comfy_root: Path, port: int = 8188) -> None:
        self.root = root.resolve()
        self.comfy_root = comfy_root.resolve()
        self.comfy_dir = Path(os.environ.get("COMFYUI_DIR", "/content/ComfyUI")).resolve()
        self.port = int(port)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.output_dir = self.comfy_root / "output"
        self.process: subprocess.Popen[str] | None = None
        self.start_lock = threading.Lock()
        self.session = requests.Session()
        self.client_id = str(uuid.uuid4())
        self.timeout = float(os.environ.get("COMFY_JOB_TIMEOUT", "3600"))
        self.cancel_event = threading.Event()
        self.current_prompt_id = None
        self.gpu_cache = ({}, 0.0)
        self.identity_path = self.comfy_root / "backend_identity.json"
        self.memory_node_available: bool | None = None
        self.log_path = self.comfy_root / "logs" / "comfyui.log"
        self.log_handle = None

    @property
    def model_dir(self) -> Path:
        return self.comfy_root / "models" / "diffusion_models"

    @property
    def text_encoder_dir(self) -> Path:
        return self.comfy_root / "models" / "text_encoders"

    @property
    def vae_dir(self) -> Path:
        return self.comfy_root / "models" / "vae"

    @property
    def lora_dir(self) -> Path:
        return self.comfy_root / "models" / "loras"

    def _ensure_directories(self) -> None:
        for directory in (self.model_dir, self.text_encoder_dir, self.vae_dir, self.lora_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _ensure_cleanup_node(self) -> None:
        source = self.root / "comfy_memory_node.py"
        if not source.exists():
            raise RuntimeError("O custom node de manutenção de memória não está presente no projeto.")
        # O ComfyUI resolve custom_nodes relativo a --base-directory, que é
        # comfy_root; não relativo ao diretório onde o código foi clonado.
        destination_dir = self.comfy_root / "custom_nodes"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "modellab_memory.py"
        if not destination.exists() or destination.read_bytes() != source.read_bytes():
            shutil.copy2(source, destination)

    def _command(self) -> list[str]:
        command = [
            sys.executable,
            "main.py",
            "--listen", "127.0.0.1",
            "--port", str(self.port),
            "--base-directory", str(self.comfy_root),
            "--disable-auto-launch",
            "--preview-method", "none",
            # A T4 não executa BF16 nativo de forma eficiente. O ComfyUI faz o
            # cast interno para FP16 quando esta opção está ativa.
            "--force-fp16",
            "--fp16-intermediates",
            # O parser do ComfyUI trata gpu-only/highvram como opções
            # mutuamente exclusivas. gpu-only é a política desejada: não
            # permitir offload de encoders/modelo para a RAM entre jobs.
            os.environ.get("COMFY_MEMORY_MODE", "--gpu-only"),
            # Evita manter resultados intermediários de nodes na RAM. O cache
            # interno de modelos do ComfyUI continua separado e residente.
            "--cache-none",
        ]
        extra = os.environ.get("COMFYUI_EXTRA_ARGS", "").strip()
        if command[command.index("--fp16-intermediates") + 1] not in {"--gpu-only", "--normalvram", "--lowvram"}:
            raise ValueError("COMFY_MEMORY_MODE deve ser --gpu-only, --normalvram ou --lowvram.")
        if extra:
            tokens = shlex.split(extra)
            allowed = {"--disable-metadata", "--use-pytorch-cross-attention", "--dont-upcast-attention"}
            if any(token not in allowed for token in tokens):
                raise ValueError("COMFYUI_EXTRA_ARGS contém uma opção não permitida; use COMFY_MEMORY_MODE para memória.")
            command.extend(tokens)
        return command

    def _open_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        handle.write(f"\n\n=== início do ComfyUI {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        handle.write("Comando: " + " ".join(self._command()) + "\n")
        handle.flush()
        return handle

    def _log_tail(self, limit: int = 8_000) -> str:
        try:
            if not self.log_path.exists():
                return ""
            with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit), os.SEEK_SET)
                tail = handle.read().strip()
            for key in ("CIVITAI_TOKEN", "MEGA_PASSWORD", "STUDIO_PASSWORD", "STUDIO_SECRET"):
                secret = os.environ.get(key, "")
                if len(secret) >= 4: tail = tail.replace(secret, "[redigido]")
            return tail
        except OSError:
            return ""

    def _startup_error(self, message: str) -> RuntimeError:
        tail = self._log_tail()
        if tail:
            return RuntimeError(f"{message}\nLog do ComfyUI ({self.log_path}):\n{tail[-8_000:]}")
        return RuntimeError(f"{message}\nLog do ComfyUI: {self.log_path} (vazio ou indisponível)")

    def _memory_node_loaded(self) -> bool | None:
        """Consulta object_info sem iniciar ou descarregar o ComfyUI."""
        try:
            response = self.session.get(f"{self.base_url}/object_info", timeout=5)
            if not response.ok:
                return None
            objects = response.json()
            return "ModelLabMemoryCleanup" in objects
        except (requests.RequestException, ValueError):
            return None

    def _reachable(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=2)
            return response.ok
        except requests.RequestException:
            return False

    def ensure_running(self) -> None:
        with self.start_lock:
            # O node precisa ser copiado mesmo se houver um processo antigo
            # residente; nesse caso ele só ficará disponível após reinício.
            self._ensure_directories()
            self._ensure_cleanup_node()
            if self._reachable():
                identity = read_json(self.identity_path, {})
                signature = hashlib.sha256(" ".join(self._command()).encode()).hexdigest()
                if identity.get("signature") != signature or identity.get("pid") != (self.process.pid if self.process else None):
                    raise RuntimeError("A porta ComfyUI está ocupada por outro processo. Pare a sessão anterior ou escolha COMFY_PORT.")
                self.memory_node_available = self._memory_node_loaded()
                return
            if not (self.comfy_dir / "main.py").exists():
                raise RuntimeError(
                    f"ComfyUI não encontrado em {self.comfy_dir}. Execute o launcher para instalar o backend."
                )
            if self.process is not None and self.process.poll() is None:
                self._wait_until_ready()
                self.memory_node_available = self._memory_node_loaded()
                return
            if self.log_handle is not None: self.log_handle.close()
            self.log_handle = self._open_log()
            try:
                self.process = subprocess.Popen(
                    self._command(),
                    cwd=self.comfy_dir,
                    text=True,
                    stdout=self.log_handle,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
            except Exception:
                self.log_handle.close()
                self.log_handle = None
                raise
            try:
                self._wait_until_ready()
                response = self.session.get(f"{self.base_url}/object_info", timeout=10)
                response.raise_for_status()
                required = {"UNETLoader", "CLIPLoader", "VAELoader", "KSampler", "EmptySD3LatentImage", "VAEEncode", "VAEDecodeTiled", "LoadImage", "ImageScale", "SaveImage"}
                if missing := required - set(response.json()):
                    raise RuntimeError("Nodes obrigatórios ausentes: " + ", ".join(sorted(missing)))
                atomic_json(self.identity_path, {"pid": self.process.pid, "signature": hashlib.sha256(" ".join(self._command()).encode()).hexdigest()})
                self.memory_node_available = self._memory_node_loaded()
            except Exception:
                self.close_process()
                raise

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + float(os.environ.get("COMFY_START_TIMEOUT", "180"))
        while time.monotonic() < deadline:
            if self._reachable():
                return
            if self.process is not None and self.process.poll() is not None:
                code = self.process.returncode
                raise self._startup_error(f"O processo headless do ComfyUI encerrou antes de abrir a API (código {code}).")
            time.sleep(0.5)
        raise self._startup_error(f"ComfyUI não respondeu em {self.base_url} dentro do tempo configurado.")

    @staticmethod
    def _headers() -> dict[str, str]:
        token = os.environ.get("CIVITAI_TOKEN", "").strip()
        headers = {"User-Agent": "ModelLab-Studio/2.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def ensure_file(
        self,
        url: str,
        destination: Path,
        minimum_bytes: int,
        report: Callable[[int], None] | None = None,
        **options,
    ) -> Path:
        return ensure_download(url, destination, minimum_bytes, report, **options)

    def ensure_anima_dependencies(self, report: ProgressCallback | None = None) -> None:
        """Baixa os arquivos compartilhados exigidos pelo workflow Anima."""
        self._ensure_directories()
        files = [
            (
                "https://huggingface.co/circlestone-labs/Anima/resolve/f973fc41ec7545364ac9776c2440285f43ff2a30/split_files/text_encoders/qwen_3_06b_base.safetensors",
                self.text_encoder_dir / "qwen_3_06b_base.safetensors",
                100 * 1024 * 1024,
                10, "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba", 1192135096,
            ),
            (
                "https://huggingface.co/circlestone-labs/Anima/resolve/f973fc41ec7545364ac9776c2440285f43ff2a30/split_files/vae/qwen_image_vae.safetensors",
                self.vae_dir / "qwen_image_vae.safetensors",
                100 * 1024 * 1024,
                20, "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", 253806246,
            ),
        ]
        for url, destination, minimum, phase_value, sha256, size in files:
            if report:
                report(0, None, phase_value, "downloading_anima_components")
            self.ensure_file(
                url,
                destination,
                minimum,
                lambda value, base=phase_value: report(value, None, base, "downloading_anima_components") if report else None,
                cancelled=self.cancel_event.is_set, sha256=sha256, expected_bytes=size,
            )

    def register_checkpoint(self, source: Path) -> str:
        self._ensure_directories()
        if source.resolve().parent == self.model_dir.resolve(): return source.name
        # Use a unique registered name for custom paths, without copying gigabytes.
        name = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12] + "_" + source.name
        destination = self.model_dir / name
        if destination.is_symlink() and destination.resolve() != source.resolve(): destination.unlink()
        if not destination.exists(): destination.symlink_to(source.resolve())
        return name

    def upload_source(self, path: Path) -> str:
        with path.open("rb") as handle:
            response = self.session.post(f"{self.base_url}/upload/image", files={"image": (path.name, handle)},
                data={"type": "input", "overwrite": "true"}, timeout=60)
        response.raise_for_status()
        info = response.json()
        return "/".join(filter(None, [info.get("subfolder"), info["name"]]))

    def gpu_memory(self) -> dict[str, Any]:
        cached, stamp = self.gpu_cache
        if time.monotonic() - stamp < 2: return cached
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=1)
            response.raise_for_status()
            device = (response.json().get("devices") or [{}])[0]
            total, free = device.get("vram_total", 0), device.get("vram_free", 0)
            cached = {"used_gb": round((total - free) / 1024**3, 2), "free_gb": round(free / 1024**3, 2),
                "total_gb": round(total / 1024**3, 2), "source": "ComfyUI system_stats"}
        except (requests.RequestException, ValueError, IndexError):
            cached = {}
        self.gpu_cache = (cached, time.monotonic())
        return cached

    def copy_lora(self, source: Path) -> str:
        self._ensure_directories()
        safe_name = source.name.replace("/", "_").replace("\\", "_")
        destination = self.lora_dir / safe_name
        if source.resolve() != destination.resolve():
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
        return destination.name

    @staticmethod
    def _sampler(sampler: str) -> tuple[str, str]:
        mapping = {
            "euler_a": ("euler_ancestral", "normal"),
            "euler": ("euler", "normal"),
            "dpmpp_2m": ("dpmpp_2m", "normal"),
            "dpmpp_2m_sde_gpu": ("dpmpp_2m_sde_gpu", "karras"),
        }
        return mapping.get(sampler, mapping["euler_a"])

    def build_workflow(
        self,
        job: Any,
        spec: dict[str, Any],
        model_filename: str,
        lora_names: list[tuple[str, float]],
    ) -> dict[str, dict[str, Any]]:
        defaults = spec.get("defaults", {})
        positive_prefix = str(defaults.get("positive_prefix", "")).strip()
        positive = f"{positive_prefix}, {job.params.prompt}" if positive_prefix else job.params.prompt
        negative = job.params.negative_prompt or str(defaults.get("negative_prompt", ""))
        sampler_name, scheduler = self._sampler(job.params.sampler)
        model_node = "1"
        use_memory_node = self.memory_node_available is not False
        workflow: dict[str, dict[str, Any]] = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": model_filename, "weight_dtype": "default"},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_06b_base.safetensors",
                    "type": "qwen_image",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_vae.safetensors"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": positive},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": negative},
            },
            "6": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": job.params.width, "height": job.params.height, "batch_size": 1},
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": [model_node, 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["6", 0],
                    "seed": job.params.seed,
                    "steps": job.params.steps,
                    "cfg": job.params.guidance,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": job.params.strength if job.params.mode == "img2img" else 1.0,
                },
            },
            "8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["7", 0], "vae": ["3", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
            "10": {
                "class_type": "SaveImage",
                "inputs": {"images": ["9" if use_memory_node else "8", 0], "filename_prefix": f"modellab_{job.id}"},
            },
        }
        if job.params.mode == "img2img":
            workflow["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}}
            workflow["12"] = {"class_type": "LoadImage", "inputs": {"image": job.comfy_source}}
            workflow["13"] = {"class_type": "ImageScale", "inputs": {"image": ["12", 0], "upscale_method": "lanczos",
                "width": job.params.width, "height": job.params.height, "crop": "center"}}
        if use_memory_node:
            workflow["9"] = {
                "class_type": "ModelLabMemoryCleanup",
                "inputs": {"image": ["8", 0]},
            }
        upscale = getattr(job.params, "upscale", 1.0)
        if upscale > 1:
            workflow["11"] = {"class_type": "ImageScale", "inputs": {"image": ["8", 0], "upscale_method": "lanczos",
                "width": int(job.params.width * upscale), "height": int(job.params.height * upscale), "crop": "disabled"}}
            if use_memory_node: workflow["9"]["inputs"]["image"] = ["11", 0]
            else: workflow["10"]["inputs"]["images"] = ["11", 0]
        for index, (lora_name, weight) in enumerate(lora_names, start=1):
            node_id = str(100 + index)
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [model_node, 0],
                    "lora_name": lora_name,
                    "strength_model": weight,
                },
            }
            model_node = node_id
        if lora_names:
            workflow["7"]["inputs"]["model"] = [model_node, 0]
        return workflow

    def _download_output(self, image_info: dict[str, Any]) -> bytes:
        params = {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
        response = self.session.get(f"{self.base_url}/view", params=params, timeout=120)
        response.raise_for_status()
        return response.content

    def _cancel_prompt(self, prompt_id: str) -> None:
        # Interrupt is global: verify that our own prompt is running first.
        response = self.session.get(f"{self.base_url}/queue", timeout=5)
        response.raise_for_status()
        data = response.json()
        if any(item[1] == prompt_id for item in data.get("queue_running", [])):
            self.session.post(f"{self.base_url}/interrupt", timeout=5).raise_for_status()
        elif any(item[1] == prompt_id for item in data.get("queue_pending", [])):
            self.session.post(f"{self.base_url}/queue", json={"delete": [prompt_id]}, timeout=5).raise_for_status()

    def submit_and_wait(self, workflow, update=None, on_submitted=None) -> Image.Image:
        self.ensure_running()
        ws = None
        try:
            try:
                import websocket
                ws = websocket.create_connection(f"ws://127.0.0.1:{self.port}/ws?clientId={self.client_id}", timeout=2)
                ws.settimeout(0.2)
            except Exception:
                ws = None
            response = self.session.post(f"{self.base_url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=30)
            if not response.ok:
                raise RuntimeError(f"ComfyUI rejeitou o workflow: {response.text[:2000]}")
            payload = response.json()
            if payload.get("error") or not payload.get("prompt_id"):
                raise RuntimeError(f"ComfyUI rejeitou o workflow: {payload}")
            prompt_id = self.current_prompt_id = payload["prompt_id"]
            if on_submitted: on_submitted(prompt_id)
            deadline = time.monotonic() + self.timeout
            next_history = 0.0
            active_prompt = None
            while time.monotonic() < deadline:
                if self.cancel_event.is_set():
                    try: self._cancel_prompt(prompt_id)
                    except requests.RequestException:
                        self.close_process()
                    raise InterruptedError("Geração cancelada.")
                if ws:
                    try:
                        message = ws.recv()
                        if isinstance(message, str):
                            event = json.loads(message); data = event.get("data", {})
                            event_prompt = data.get("prompt_id")
                            if event_prompt: active_prompt = event_prompt
                            if (event_prompt or active_prompt) == prompt_id and update:
                                if event.get("type") == "progress":
                                    update(min(95, int(95 * data.get("value", 0) / max(data.get("max", 1), 1))), self.gpu_memory().get("used_gb"), 100, "generating")
                                elif event.get("type") == "executing":
                                    node = str(data.get("node"))
                                    phase = "decoding" if node == "8" else "saving" if node == "10" else "loading_weights"
                                    update(98 if node == "10" else 96 if node == "8" else 0, self.gpu_memory().get("used_gb"), 100, phase)
                    except Exception as exc:
                        # Socket timeouts are normal; reconnect fallback uses history.
                        if exc.__class__.__name__ not in {"WebSocketTimeoutException", "TimeoutError"}:
                            ws.close(); ws = None
                elif update:
                    update(0, self.gpu_memory().get("used_gb"), 100, "generating_indeterminate")
                if time.monotonic() >= next_history:
                    next_history = time.monotonic() + 1
                    try:
                        response = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=5)
                        response.raise_for_status(); history = response.json().get(prompt_id)
                    except requests.RequestException: history = None
                    if history:
                        status = history.get("status") or {}
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI falhou: {status.get('messages', [])}")
                        output = (history.get("outputs") or {}).get("10", {})
                        for image_info in output.get("images", []):
                            with Image.open(io.BytesIO(self._download_output(image_info))) as image:
                                return image.convert("RGB")
                        if status.get("completed"):
                            raise RuntimeError("O workflow terminou sem a imagem final.")
                if self.process is not None and self.process.poll() is not None:
                    raise self._startup_error("ComfyUI encerrou durante a geração.")
                if not ws: time.sleep(0.5)
            self._cancel_prompt(prompt_id)
            raise TimeoutError(f"Workflow cancelado após exceder {self.timeout:g}s.")
        finally:
            self.current_prompt_id = None
            if ws: ws.close()

    def status(self) -> dict[str, Any]:
        """Retorna saúde/estado sem iniciar, limpar ou descarregar o backend."""
        payload: dict[str, Any] = {
            "url": self.base_url,
            "process_alive": bool(self.process is not None and self.process.poll() is None),
            "reachable": False,
            "memory_node_available": self.memory_node_available,
            "log_path": str(self.log_path),
            "log_tail": self._log_tail(4_000),
        }
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=3)
            payload["reachable"] = response.ok
            if response.ok:
                payload["system"] = response.json()
                self.memory_node_available = self._memory_node_loaded()
                payload["memory_node_available"] = self.memory_node_available
            queue_response = self.session.get(f"{self.base_url}/queue", timeout=3)
            if queue_response.ok:
                queue_payload = queue_response.json()
                payload["queue_running"] = len(queue_payload.get("queue_running", []))
                payload["queue_pending"] = len(queue_payload.get("queue_pending", []))
        except (requests.RequestException, ValueError) as exc:
            payload["error"] = str(exc)[:180]
        return payload

    def close_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=5)
        self.process = None
        self.identity_path.unlink(missing_ok=True)
        if self.log_handle is not None:
            self.log_handle.close(); self.log_handle = None

    def close(self) -> None:
        self.cancel_event.set()
        with self.start_lock:
            self.close_process()
        self.session.close()


__all__ = ["ComfyBackend"]
