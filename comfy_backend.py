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
        self.model_lock = threading.Lock()
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
            "--gpu-only",
            # Evita manter resultados intermediários de nodes na RAM. O cache
            # interno de modelos do ComfyUI continua separado e residente.
            "--cache-none",
        ]
        extra = os.environ.get("COMFYUI_EXTRA_ARGS", "").strip()
        if extra:
            command.extend(shlex.split(extra))
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
                return handle.read().strip()
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
                self.memory_node_available = self._memory_node_loaded()
            except Exception:
                if self.process.poll() is None:
                    self.process.terminate()
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
    ) -> Path:
        """Baixa um arquivo grande com retomada e sem carregá-lo na RAM."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size >= minimum_bytes:
            if report:
                report(100)
            return destination
        temporary = destination.with_suffix(destination.suffix + ".part")
        resume_at = temporary.stat().st_size if temporary.exists() else 0
        headers = self._headers()
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"
        with requests.get(url, headers=headers, stream=True, timeout=(20, 300)) as response:
            response.raise_for_status()
            append = bool(resume_at and response.status_code == 206)
            if not append:
                resume_at = 0
            try:
                content_length = int(response.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                content_length = 0
            total = content_length + resume_at if append and content_length else content_length
            downloaded = resume_at
            last = -1
            with temporary.open("ab" if append else "wb") as handle:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if report and total:
                        value = min(99, int(downloaded * 100 / total))
                        if value != last:
                            report(value)
                            last = value
        if temporary.stat().st_size < minimum_bytes:
            raise RuntimeError(f"Download incompleto: {destination.name}. O arquivo parcial será retomado.")
        temporary.replace(destination)
        if report:
            report(100)
        return destination

    def ensure_anima_dependencies(self, report: ProgressCallback | None = None) -> None:
        """Baixa os arquivos compartilhados exigidos pelo workflow Anima."""
        self._ensure_directories()
        files = [
            (
                "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
                self.text_encoder_dir / "qwen_3_06b_base.safetensors",
                100 * 1024 * 1024,
                10,
            ),
            (
                "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                self.vae_dir / "qwen_image_vae.safetensors",
                100 * 1024 * 1024,
                20,
            ),
        ]
        for url, destination, minimum, phase_value in files:
            if report:
                report(0, None, phase_value, "downloading_anima_components")
            self.ensure_file(
                url,
                destination,
                minimum,
                lambda value, base=phase_value: report(value, None, base, "downloading_anima_components") if report else None,
            )

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
                "class_type": "EmptyLatentImage",
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
                    "denoise": 1.0,
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "10": {
                "class_type": "SaveImage",
                "inputs": {"images": ["9" if use_memory_node else "8", 0], "filename_prefix": f"modellab_{job.id}"},
            },
        }
        if use_memory_node:
            workflow["9"] = {
                "class_type": "ModelLabMemoryCleanup",
                "inputs": {"image": ["8", 0]},
            }
        for index, (lora_name, weight) in enumerate(lora_names, start=1):
            node_id = str(10 + index)
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

    def submit_and_wait(self, workflow: dict[str, dict[str, Any]], update: ProgressCallback | None = None) -> Image.Image:
        self.ensure_running()
        response = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"ComfyUI rejeitou o workflow: {payload['error']}")
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Resposta inesperada do ComfyUI: {payload}")
        deadline = time.monotonic() + self.timeout
        last_progress = -1
        while time.monotonic() < deadline:
            try:
                history_response = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
                history_response.raise_for_status()
                history = history_response.json().get(prompt_id)
            except requests.RequestException:
                history = None
            if history:
                status = history.get("status") or {}
                status_string = str(status.get("status_str", ""))
                if status_string == "error" or status.get("completed") is False and status.get("messages"):
                    raise RuntimeError(f"ComfyUI falhou ao executar o workflow: {status.get('messages', [])}")
                outputs = history.get("outputs") or {}
                for output in outputs.values():
                    for image_info in output.get("images", []) if isinstance(output, dict) else []:
                        raw = self._download_output(image_info)
                        with Image.open(io.BytesIO(raw)) as image:
                            return image.convert("RGB")
                if status.get("completed") and not outputs:
                    raise RuntimeError("ComfyUI concluiu o workflow sem produzir uma imagem.")
            if update:
                last_progress = min(98, last_progress + 1)
                update(last_progress, None, 100, "generating")
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("O backend do ComfyUI encerrou durante a geração.")
            time.sleep(0.35)
        raise TimeoutError(f"O workflow do ComfyUI excedeu {self.timeout:g}s.")

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

    def close(self) -> None:
        with self.start_lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None


__all__ = ["ComfyBackend"]
