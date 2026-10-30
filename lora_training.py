"""Isolated Anima LoRA training service for the ModelLab Colab panel."""
from __future__ import annotations

import json
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
from typing import Any, Callable

from studio_storage import atomic_json, read_json

ROOT = Path(os.environ.get("STUDIO_ROOT", "/content/modellab-studio")).resolve()
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", ROOT)).resolve()
TRAIN_ROOT = ROOT / "lora-training"
DATASETS = TRAIN_ROOT / "datasets"
OUTPUTS = TRAIN_ROOT / "output"
CONFIGS = TRAIN_ROOT / "configs"
LOGS = TRAIN_ROOT / "logs"
SCRIPTS = TRAIN_ROOT / "sd-scripts"
VENV = TRAIN_ROOT / "venv"
STATE = TRAIN_ROOT / "state.json"
MAX_ARCHIVE = 200 * 1024 * 1024
MAX_UNPACKED = 768 * 1024 * 1024
MAX_FILES = 3000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,47}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip()).strip("-_")
    if not SAFE_NAME.fullmatch(cleaned):
        raise ValueError("Use um nome de 2 a 48 caracteres: letras, números, hífen ou sublinhado.")
    return cleaned


class AnimaLoRATrainer:
    def __init__(self, on_finish: Callable[[], None] | None = None):
        self.on_finish = on_finish
        self.lock = threading.RLock()
        self.setup_thread: threading.Thread | None = None
        self.train_thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self.state: dict[str, Any] = read_json(STATE, {}) or {}
        self.state.setdefault("setup", "not_started")
        self.state.setdefault("status", "idle")
        self.state.setdefault("logs", [])
        self.state.setdefault("datasets", [])
        self.state.setdefault("loras", [])
        self._persist()

    def _persist(self) -> None:
        with self.lock:
            atomic_json(STATE, self.state)

    def _log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {str(message)[:900]}"
        with self.lock:
            self.state["logs"] = (self.state.get("logs") or [])[-179:] + [line]
            self.state["updated_at"] = _now()
            self._persist()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            data = dict(self.state)
            data["logs"] = list(self.state.get("logs") or [])[-100:]
            data["busy"] = bool(
                (self.setup_thread and self.setup_thread.is_alive())
                or (self.train_thread and self.train_thread.is_alive())
            )
            return data

    def _run_setup_command(self, command: list[str], cwd: Path | None = None) -> None:
        self._log("$ " + " ".join(command))
        process = subprocess.Popen(command, cwd=str(cwd or TRAIN_ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            raise RuntimeError(f"Comando de preparação terminou com código {code}.")

    def prepare(self) -> dict[str, Any]:
        with self.lock:
            if self.setup_thread and self.setup_thread.is_alive():
                return self.snapshot()
            if self.state.get("setup") == "ready":
                return self.snapshot()
            self.state.update({"setup": "preparing", "error": None})
            self._persist()
            self.setup_thread = threading.Thread(target=self._prepare_worker, name="anima-lora-setup", daemon=True)
            self.setup_thread.start()
        return self.snapshot()

    def _prepare_worker(self) -> None:
        try:
            TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
            if not (SCRIPTS / ".git").is_dir():
                self._run_setup_command(["git", "clone", "--depth", "1",
                                         "https://github.com/kohya-ss/sd-scripts.git", str(SCRIPTS)])
            if not (VENV / "bin" / "python").is_file():
                self._run_setup_command(["python", "-m", "venv", str(VENV)])
            python = str(VENV / "bin" / "python")
            pip = [python, "-m", "pip"]
            self._run_setup_command(pip + ["install", "--upgrade", "pip"])
            # Separate environment: never modify Colab/ComfyUI's active torch stack.
            self._run_setup_command(pip + ["install", "torch==2.5.1", "torchvision==0.20.1",
                                           "--index-url", "https://download.pytorch.org/whl/cu124"])
            constraints = TRAIN_ROOT / "constraints.txt"
            constraints.write_text("torch==2.5.1\ntorchvision==0.20.1\n", encoding="utf-8")
            self._run_setup_command(pip + ["install", "-r", str(SCRIPTS / "requirements.txt"),
                                           "-c", str(constraints)], cwd=SCRIPTS)
            self._run_setup_command(pip + ["install", "bitsandbytes>=0.45,<1", "accelerate>=1.2", "toml"])
            check = subprocess.run(
                [python, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"],
                capture_output=True, text=True, timeout=45,
            )
            if check.returncode:
                raise RuntimeError("CUDA não ficou disponível no ambiente isolado: " + check.stderr[-1200:])
            if not (SCRIPTS / "anima_train_network.py").is_file():
                raise RuntimeError("A revisão de sd-scripts instalada não inclui anima_train_network.py.")
            self._log(check.stdout.strip())
            with self.lock:
                self.state.update({"setup": "ready", "error": None, "trainer_revision": self._git_revision()})
                self._persist()
            self._log("Treinador Anima pronto. O ambiente de inferência do ModelLab não foi alterado.")
        except Exception as exc:
            with self.lock:
                self.state.update({"setup": "failed", "error": str(exc)[:1800]})
                self._persist()
            self._log("Falha na preparação: " + str(exc))

    def _git_revision(self) -> str:
        result = subprocess.run(["git", "-C", str(SCRIPTS), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def import_dataset(self, archive_path: Path, dataset_name: str, shared_caption: str = "") -> dict[str, Any]:
        if archive_path.stat().st_size > MAX_ARCHIVE:
            archive_path.unlink(missing_ok=True)
            raise ValueError("O ZIP excede o limite de 200 MB.")
        name = _name(dataset_name or archive_path.stem)
        DATASETS.mkdir(parents=True, exist_ok=True)
        destination = DATASETS / name
        temporary = DATASETS / (name + ".tmp-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_FILES:
                    raise ValueError("O ZIP excede o limite de 3000 arquivos.")
                if sum(item.file_size for item in entries) > MAX_UNPACKED:
                    raise ValueError("O conteúdo descompactado excede 768 MB.")
                seen: set[str] = set()
                for item in entries:
                    parts = Path(item.filename).parts
                    mode = (item.external_attr >> 16) & 0o170000
                    if Path(item.filename).is_absolute() or ".." in parts or mode == 0o120000:
                        raise ValueError("O ZIP contém um caminho ou link simbólico inválido.")
                    if item.is_dir():
                        continue
                    ext = Path(item.filename).suffix.lower()
                    if ext not in IMAGE_EXTS | {".txt"}:
                        continue
                    target_name = Path(item.filename).name
                    if not target_name or target_name in seen:
                        raise ValueError("O ZIP contém nomes repetidos após a organização do dataset.")
                    seen.add(target_name)
                    with archive.open(item) as source, (temporary / target_name).open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            images = sorted(p for p in temporary.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if not images:
                raise ValueError("O ZIP não contém imagens PNG, JPG ou WEBP.")
            caption = str(shared_caption or "").strip()[:1500]
            missing = [p for p in images if not p.with_suffix(".txt").exists()]
            if missing and not caption:
                raise ValueError(f"{len(missing)} imagem(ns) estão sem legenda .txt. Inclua as legendas no ZIP ou informe uma legenda comum.")
            for image in missing:
                image.with_suffix(".txt").write_text(caption, encoding="utf-8")
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
            result = {"name": name, "images": len(images), "captions": len(images), "path": str(destination)}
            with self.lock:
                datasets = [item for item in self.state.get("datasets", []) if item.get("name") != name]
                datasets.append({k: result[k] for k in ("name", "images", "captions")})
                self.state["datasets"] = datasets[-30:]
                self.state["selected_dataset"] = name
                self._persist()
            self._log(f"Dataset {name} importado: {len(images)} imagens e legendas correspondentes.")
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
            dataset_name = _name(payload.get("dataset") or self.state.get("selected_dataset") or "")
            dataset = DATASETS / dataset_name
            if not dataset.is_dir():
                raise ValueError("Importe um dataset antes de iniciar.")
            images = [p for p in dataset.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if not images or any(not p.with_suffix(".txt").is_file() for p in images):
                raise ValueError("Todas as imagens precisam de uma legenda .txt com o mesmo nome.")
            project = _name(payload.get("project_name") or "anima-lora")
            rank = int(payload.get("rank", 8))
            resolution = int(payload.get("resolution", 512))
            epochs = int(payload.get("epochs", 4))
            repeats = int(payload.get("repeats", 2))
            lr = float(payload.get("learning_rate", 0.0001))
            if rank not in (4, 8, 16) or resolution not in (512, 640, 768):
                raise ValueError("Rank ou resolução fora das opções disponíveis.")
            if not 1 <= epochs <= 20 or not 1 <= repeats <= 10 or not 0.000001 <= lr <= 0.0005:
                raise ValueError("Épocas, repeats ou learning rate fora dos limites.")
            steps = len(images) * repeats * epochs
            if steps > 1200 and not payload.get("confirm_long_run"):
                raise ValueError(f"Estimativa de {steps} passos. Confirme a execução longa ou reduza épocas/repeats.")
            output = OUTPUTS / project
            output.mkdir(parents=True, exist_ok=True)
            config_path = CONFIGS / f"{project}-dataset.toml"
            CONFIGS.mkdir(parents=True, exist_ok=True)
            import toml
            dataset_config = {
                "general": {"resolution": resolution, "enable_bucket": True, "bucket_no_upscale": False,
                            "bucket_reso_steps": 64, "min_bucket_reso": 256, "max_bucket_reso": 1536},
                "datasets": [{"resolution": resolution, "subsets": [{"image_dir": str(dataset),
                    "num_repeats": repeats, "caption_extension": ".txt", "shuffle_caption": False}]}],
            }
            config_path.write_text(toml.dumps(dataset_config), encoding="utf-8")
            model_path = Path(os.environ.get("MODEL_PATH", ROOT / "models/diffusion_models/WAI-ANIMA1.safetensors"))
            qwen_path = COMFY_ROOT / "models/text_encoders/qwen_3_06b_base.safetensors"
            vae_path = COMFY_ROOT / "models/vae/qwen_image_vae.safetensors"
            for label, path in (("WAI-ANIMA", model_path), ("Qwen3", qwen_path), ("VAE", vae_path)):
                if not path.is_file():
                    raise ValueError(f"Arquivo necessário não encontrado: {label} ({path.name}). Execute a preparação do modelo no painel primeiro.")
            args = [
                str(VENV / "bin" / "python"), "-m", "accelerate.commands.launch",
                "--num_cpu_threads_per_process", "1", "--mixed_precision", "fp16",
                str(SCRIPTS / "anima_train_network.py"),
                "--pretrained_model_name_or_path", str(model_path),
                "--qwen3", str(qwen_path), "--vae", str(vae_path),
                "--dataset_config", str(config_path), "--output_dir", str(output),
                "--output_name", project, "--save_model_as", "safetensors",
                "--network_module", "networks.lora_anima", "--network_dim", str(rank),
                "--network_alpha", str(rank), "--network_train_unet_only",
                "--learning_rate", str(lr), "--optimizer_type", "AdamW8bit",
                "--lr_scheduler", "constant", "--timestep_sampling", "sigmoid",
                "--discrete_flow_shift", "1.0", "--max_train_epochs", str(epochs),
                "--save_every_n_epochs", "1", "--save_last_n_epochs", "3",
                "--mixed_precision", "fp16", "--gradient_checkpointing",
                "--cache_latents", "--cache_text_encoder_outputs",
                "--qwen_image_vae_2d", "--blocks_to_swap", "16", "--block_swap_h2d_only",
                "--max_data_loader_n_workers", "0",
            ]
            run_id = uuid.uuid4().hex[:12]
            self.state.update({"status": "queued", "run_id": run_id, "project_name": project,
                               "dataset": dataset_name, "progress": 0, "current_step": 0,
                               "total_steps": steps, "loss": None, "error": None,
                               "started_at": _now(), "result": None, "command": args[3:]})
            self._persist()
            self.train_thread = threading.Thread(target=self._train_worker, args=(args, output, project),
                                                 name="anima-lora-training", daemon=True)
            self.train_thread.start()
            return self.snapshot()

    def _train_worker(self, command: list[str], output: Path, project: str) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS / f"{project}.log"
        try:
            with self.lock:
                self.state["status"] = "running"
                self._persist()
            self._log(f"Treinamento {project} iniciado em FP16 para T4.")
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(SCRIPTS), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                with self.lock:
                    self.process = process
                assert process.stdout is not None
                for raw in process.stdout:
                    line = raw.rstrip()
                    log.write(line + "\n")
                    log.flush()
                    step_match = re.search(r"(\d+)\s*/\s*(\d+)", line)
                    loss_match = re.search(r"(?:loss|train_loss)\s*[:=]\s*([0-9.eE+-]+)", line, re.I)
                    if step_match:
                        current, total = int(step_match.group(1)), max(1, int(step_match.group(2)))
                        with self.lock:
                            self.state["current_step"] = current
                            self.state["total_steps"] = total
                            self.state["progress"] = min(99, int(current * 100 / total))
                            self._persist()
                    if loss_match:
                        try:
                            with self.lock:
                                self.state["loss"] = float(loss_match.group(1))
                                self._persist()
                        except ValueError:
                            pass
                    if line:
                        self._log(line)
                code = process.wait()
                with self.lock:
                    self.process = None
                if code:
                    raise RuntimeError(f"Treinador terminou com código {code}. Consulte os logs.")
            files = sorted(output.glob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
            if not files:
                raise RuntimeError("O treinamento terminou sem gerar um arquivo .safetensors.")
            comfy_loras = COMFY_ROOT / "models" / "loras"
            comfy_loras.mkdir(parents=True, exist_ok=True)
            target = comfy_loras / f"{project}.safetensors"
            shutil.copy2(files[0], target)
            with self.lock:
                loras = [item for item in self.state.get("loras", []) if item.get("file") != target.name]
                loras.append({"name": project, "file": target.name, "size_mb": round(target.stat().st_size / 1024**2, 2),
                              "created_at": _now()})
                self.state.update({"loras": loras[-50:], "status": "completed", "progress": 100,
                                   "result": target.name, "completed_at": _now(), "error": None})
                self._persist()
            self._log(f"LoRA instalada no diretório do ComfyUI: {target.name}")
        except Exception as exc:
            with self.lock:
                self.state.update({"status": "failed", "error": str(exc)[:1800]})
                self._persist()
            self._log("Treinamento falhou: " + str(exc))
        finally:
            with self.lock:
                self.process = None
            if self.on_finish:
                self.on_finish()

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
