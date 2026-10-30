"""Anima LoRA training worker for the ModelLab Studio Colab runtime.

Training runs in an isolated virtualenv so its PyTorch/dependency pins never
replace the ComfyUI inference stack used by the main application.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from studio_storage import atomic_json, read_json

ROOT = Path(os.environ.get("STUDIO_ROOT", "/content/modellab-studio")).resolve()
TRAIN_ROOT = ROOT / "lora-training"
DATASETS = TRAIN_ROOT / "datasets"
OUTPUTS = TRAIN_ROOT / "output"
CONFIGS = TRAIN_ROOT / "configs"
LOGS = TRAIN_ROOT / "logs"
TRAINER_DIR = TRAIN_ROOT / "sd-scripts"
VENV_DIR = TRAIN_ROOT / "venv"
STATE_FILE = TRAIN_ROOT / "state.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_FILES = 5000
REPO = "https://github.com/kohya-ss/sd-scripts.git"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip()).strip("-_")
    return name[:48] or "anima-lora"


class AnimaLoRATrainer:
    def __init__(self, model_path: Path, qwen_path: Path, vae_path: Path, comfy_lora_dir: Path):
        self.model_path = model_path
        self.qwen_path = qwen_path
        self.vae_path = vae_path
        self.comfy_lora_dir = comfy_lora_dir
        self.lock = threading.RLock()
        self.setup_thread: threading.Thread | None = None
        self.train_thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.state: dict[str, Any] = read_json(STATE_FILE, {}) or {}
        self.state.setdefault("setup", "not_started")
        self.state.setdefault("status", "idle")
        self.state.setdefault("logs", [])
        self.state.setdefault("dataset", {"name": "", "images": 0, "captions": 0})
        self._persist()

    def _persist(self) -> None:
        with self.lock:
            atomic_json(STATE_FILE, self.state)

    def _log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {str(message)[:1200]}"
        with self.lock:
            logs = list(self.state.get("logs", []))
            logs.append(line)
            self.state["logs"] = logs[-250:]
            self.state["updated_at"] = _now()
            self._persist()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.state)
            result["logs"] = list(self.state.get("logs", []))[-100:]
            result["busy"] = bool(
                (self.setup_thread and self.setup_thread.is_alive())
                or (self.train_thread and self.train_thread.is_alive())
            )
            return result

    def _run(self, command: list[str], cwd: Path | None = None) -> None:
        self._log("$ " + " ".join(command))
        process = subprocess.Popen(
            command, cwd=str(cwd or TRAIN_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        with self.lock:
            self.process = process
        assert process.stdout is not None
        for line in process.stdout:
            if line.strip():
                self._log(line.rstrip())
        code = process.wait()
        with self.lock:
            self.process = None
        if code:
            raise RuntimeError(f"Comando encerrou com código {code}.")

    def prepare(self) -> None:
        with self.lock:
            if self.setup_thread and self.setup_thread.is_alive():
                return
            if self.state.get("setup") == "ready":
                return
            self.state["setup"] = "preparing"
            self.state["error"] = None
            self._persist()
            self.setup_thread = threading.Thread(target=self._prepare_worker, daemon=True)
            self.setup_thread.start()

    def _prepare_worker(self) -> None:
        try:
            TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
            if not (TRAINER_DIR / ".git").exists():
                self._run(["git", "clone", "--depth", "1", REPO, str(TRAINER_DIR)])
            if not (VENV_DIR / "bin" / "python").exists():
                self._run(["python", "-m", "venv", str(VENV_DIR)])
            python = str(VENV_DIR / "bin" / "python")
            pip = [python, "-m", "pip"]
            # T4 (Turing, SM 7.5): use fp16 and CUDA 12.4 wheels. This venv is
            # isolated from the active Colab/ComfyUI torch installation.
            self._run(pip + ["install", "--upgrade", "pip"])
            self._run(pip + ["install", "torch==2.5.1", "torchvision==0.20.1",
                             "--index-url", "https://download.pytorch.org/whl/cu124"])
            self._run(pip + ["install", "-r", str(TRAINER_DIR / "requirements.txt")], cwd=TRAINER_DIR)
            self._run(pip + ["install", "toml", "bitsandbytes>=0.45,<1", "accelerate>=1.2"])
            check = subprocess.run(
                [python, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"],
                capture_output=True, text=True,
            )
            if check.returncode:
                raise RuntimeError("O ambiente de treino não detectou CUDA: " + check.stderr[-1200:])
            self._log(check.stdout.strip())
            if not (TRAINER_DIR / "anima_train_network.py").is_file():
                raise RuntimeError("A revisão instalada de sd-scripts não contém anima_train_network.py.")
            with self.lock:
                self.state["setup"] = "ready"
                self.state["error"] = None
                self._persist()
            self._log("Ambiente Anima preparado. O PyTorch do ComfyUI permaneceu isolado.")
        except Exception as exc:
            with self.lock:
                self.state["setup"] = "failed"
                self.state["error"] = str(exc)[:2000]
                self._persist()
            self._log("Falha ao preparar ambiente: " + str(exc))

    def import_dataset(self, archive_path: Path, dataset_name: str) -> dict[str, Any]:
        name = _safe_name(dataset_name or archive_path.stem)
        destination = DATASETS / name
        DATASETS.mkdir(parents=True, exist_ok=True)
        temporary = DATASETS / (name + ".upload-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                if len(members) > MAX_ARCHIVE_FILES:
                    raise ValueError("O ZIP contém arquivos demais (limite: 5000).")
                total = sum(item.file_size for item in members)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("O conteúdo descompactado excede 512 MB.")
                for item in members:
                    path = Path(item.filename)
                    if path.is_absolute() or ".." in path.parts or not path.parts:
                        raise ValueError("O ZIP contém um caminho inválido.")
                    if item.is_dir():
                        continue
                    suffix = path.suffix.lower()
                    if suffix not in IMAGE_EXTENSIONS and suffix != ".txt":
                        continue
                    target = temporary / path.name
                    if target.exists():
                        raise ValueError("Há nomes de arquivo duplicados no ZIP.")
                    with archive.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            images = [p for p in temporary.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            if not images:
                raise ValueError("Nenhuma imagem encontrada no ZIP.")
            captions = sum(1 for image in images if image.with_suffix(".txt").is_file())
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
            result = {"name": name, "images": len(images), "captions": captions, "path": str(destination)}
            with self.lock:
                self.state["dataset"] = {k: result[k] for k in ("name", "images", "captions")}
                self._persist()
            self._log(f"Dataset {name}: {len(images)} imagens, {captions} legendas .txt.")
            return result
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            archive_path.unlink(missing_ok=True)

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.state.get("setup") != "ready":
                raise ValueError("Prepare primeiro o ambiente de treinamento.")
            if self.train_thread and self.train_thread.is_alive():
                raise ValueError("Já existe um treinamento em andamento.")
            dataset_name = _safe_name(payload.get("dataset") or self.state.get("dataset", {}).get("name"))
            dataset = DATASETS / dataset_name
            if not dataset.is_dir():
                raise ValueError("Envie um ZIP de dataset antes de iniciar.")
            images = [p for p in dataset.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            if not images:
                raise ValueError("O dataset não contém imagens.")
            missing = [p.name for p in images if not p.with_suffix(".txt").is_file()]
            if missing:
                raise ValueError(f"{len(missing)} imagem(ns) sem legenda .txt correspondente.")
            rank = int(payload.get("rank", 8))
            resolution = int(payload.get("resolution", 512))
            epochs = int(payload.get("epochs", 4))
            repeats = int(payload.get("repeats", 2))
            lr = float(payload.get("learning_rate", 0.0001))
            if rank not in (4, 8, 16, 32) or resolution not in (512, 640, 768):
                raise ValueError("Rank ou resolução fora das opções permitidas.")
            if not 1 <= epochs <= 20 or not 1 <= repeats <= 10 or not 1e-6 <= lr <= 5e-4:
                raise ValueError("Parâmetros de treinamento fora dos limites.")
            total_steps = math.ceil(len(images) * repeats) * epochs
            if total_steps > 1000 and not payload.get("confirm_long_run"):
                raise ValueError(f"Estimativa de {total_steps} passos. Reduza épocas/repeats ou confirme a execução longa.")
            project = _safe_name(payload.get("project_name") or "anima-lora")
            output = OUTPUTS / project
            output.mkdir(parents=True, exist_ok=True)
            CONFIGS.mkdir(parents=True, exist_ok=True)
            config_path = CONFIGS / f"{project}.toml"
            import tomllib
            import tomli_w
            training = {
                "pretrained_model_name_or_path": str(self.model_path),
                "qwen3": str(self.qwen_path), "vae": str(self.vae_path),
                "network_module": "networks.lora_anima", "network_dim": rank, "network_alpha": rank,
                "network_train_unet_only": True, "learning_rate": lr,
                "optimizer_type": "AdamW8bit", "optimizer_args": ["weight_decay=0.1", "betas=[0.9, 0.99]"],
                "lr_scheduler": "cosine_with_restarts", "lr_scheduler_num_cycles": 1,
                "lr_warmup_steps": min(50, max(0, total_steps // 10)), "max_train_epochs": epochs,
                "train_batch_size": 1, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
                "seed": 42, "timestep_sampling": "sigmoid", "discrete_flow_shift": 1.0,
                "qwen3_max_token_length": 512, "t5_max_token_length": 512,
                "mixed_precision": "fp16", "gradient_checkpointing": True,
                "cache_latents": True, "cache_text_encoder_outputs": True,
                "vae_chunk_size": 32, "vae_disable_cache": True,
                "output_dir": str(output), "output_name": project, "save_model_as": "safetensors",
                "save_precision": "fp16", "save_every_n_epochs": 1, "save_last_n_epochs": 4,
                "shuffle_caption": False, "caption_extension": ".txt",
                "noise_offset": 0.03, "multires_noise_discount": 0.3,
            }
            dataset_config = {
                "general": {"resolution": resolution, "enable_bucket": True, "bucket_no_upscale": False,
                            "bucket_reso_steps": 64, "min_bucket_reso": 256, "max_bucket_reso": 2048},
                "datasets": [{"resolution": resolution, "subsets": [{"num_repeats": repeats,
                    "image_dir": str(dataset), "caption_extension": ".txt", "caption_dropout_rate": 0.1}]}],
            }
            import toml
            config_path.write_text(toml.dumps(training), encoding="utf-8")
            dataset_config_path = CONFIGS / f"{project}-dataset.toml"
            dataset_config_path.write_text(toml.dumps(dataset_config), encoding="utf-8")
            run_id = uuid.uuid4().hex[:12]
            self.state.update({"status": "queued", "run_id": run_id, "project_name": project,
                               "progress": 0, "total_steps": total_steps, "error": None,
                               "output": str(output), "started_at": _now()})
            self._persist()
            self.train_thread = threading.Thread(
                target=self._train_worker,
                args=(project, config_path, dataset_config_path, output, total_steps),
                daemon=True,
            )
            self.train_thread.start()
            return self.snapshot()

    def _train_worker(self, project: str, config: Path, dataset_config: Path, output: Path, total_steps: int) -> None:
        log_path = LOGS / f"{project}.log"
        LOGS.mkdir(parents=True, exist_ok=True)
        python = str(VENV_DIR / "bin" / "python")
        command = [python, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", "1",
                   str(TRAINER_DIR / "anima_train_network.py"), "--config_file", str(config),
                   "--dataset_config", str(dataset_config)]
        try:
            with self.lock:
                self.state["status"] = "running"
                self._persist()
            self._log(f"Iniciando {project}: {total_steps} passos estimados, FP16, rank {config.stem}.")
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(TRAINER_DIR), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                with self.lock:
                    self.process = process
                assert process.stdout is not None
                for line in process.stdout:
                    text = line.rstrip()
                    log.write(text + "\n")
                    log.flush()
                    match = re.search(r"(?:step|steps)\s*[:=]?\s*(\d+)\s*/\s*(\d+)", text, re.I)
                    if match:
                        current, total = int(match.group(1)), max(1, int(match.group(2)))
                        with self.lock:
                            self.state["progress"] = min(99, int(current * 100 / total))
                            self.state["current_step"] = current
                            self.state["total_steps"] = total
                            self._persist()
                    if text:
                        self._log(text)
                code = process.wait()
                with self.lock:
                    self.process = None
                if code:
                    raise RuntimeError(f"Treinamento terminou com código {code}. Consulte o log.")
            candidates = sorted(output.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise RuntimeError("O treinamento terminou sem gerar um arquivo .safetensors.")
            self.comfy_lora_dir.mkdir(parents=True, exist_ok=True)
            target = self.comfy_lora_dir / f"{project}.safetensors"
            shutil.copy2(candidates[0], target)
            with self.lock:
                self.state.update({"status": "completed", "progress": 100, "result": str(target),
                                   "completed_at": _now(), "error": None})
                self._persist()
            self._log(f"LoRA pronta e instalada no ComfyUI: {target.name}.")
        except Exception as exc:
            with self.lock:
                self.state.update({"status": "failed", "error": str(exc)[:2000]})
                self._persist()
            self._log("Treinamento falhou: " + str(exc))

    def cancel(self) -> None:
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            with self.lock:
                self.state.update({"status": "cancelled", "error": "Treinamento cancelado pelo painel."})
                self._persist()
            self._log("Treinamento cancelado.")
