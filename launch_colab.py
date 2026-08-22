"""Execute este arquivo em uma célula Colab depois de enviar a pasta colab_app.

Exemplo: !python /content/colab_app/launch_colab.py
Interrompa a célula para encerrar servidor e túnel.
"""

from __future__ import annotations

import getpass
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests


APP_DIR = Path(__file__).resolve().parent
PORT = os.environ.get("PORT", "7860")
SERVER_START_TIMEOUT = float(os.environ.get("SERVER_START_TIMEOUT", "180"))
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/content/ComfyUI"))
COMFYUI_REPO = os.environ.get("COMFYUI_REPO", "https://github.com/comfyanonymous/ComfyUI.git")
COMFYUI_COMMIT = os.environ.get("COMFYUI_COMMIT", "c1739380c6fab78e7e263cb665d04aafbfe24593")
TUNNEL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", re.I)


def ask_secret(name: str, prompt: str, required: bool = True) -> None:
    if os.environ.get(name):
        return
    value = getpass.getpass(prompt)
    if required and not value:
        raise RuntimeError(f"{name} é obrigatório.")
    if value:
        os.environ[name] = value


def install_requirements() -> None:
    print("[setup] Atualizando as dependências diretas do estúdio sem tocar no PyTorch/CUDA…")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps",
        "-r", str(APP_DIR / "requirements.txt"),
    ], check=True)
    print("[setup] Instalando dependências do ComfyUI sem substituir torch, torchvision ou torchaudio…")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps",
        "-r", str(APP_DIR / "comfy_requirements.txt"),
    ], check=True)


def ensure_comfyui() -> None:
    """Instala uma revisão conhecida do backend sem abrir o frontend."""
    COMFYUI_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (COMFYUI_DIR / ".git").exists():
        print(f"[setup] Clonando ComfyUI em {COMFYUI_DIR}…")
        subprocess.run(["git", "clone", "--filter=blob:none", COMFYUI_REPO, str(COMFYUI_DIR)], check=True)
    current = subprocess.check_output(["git", "-C", str(COMFYUI_DIR), "rev-parse", "HEAD"], text=True).strip()
    if current != COMFYUI_COMMIT:
        print(f"[setup] Fixando ComfyUI em {COMFYUI_COMMIT}…")
        subprocess.run(["git", "-C", str(COMFYUI_DIR), "fetch", "--depth", "1", "origin", COMFYUI_COMMIT], check=True)
        subprocess.run(["git", "-C", str(COMFYUI_DIR), "checkout", "--detach", COMFYUI_COMMIT], check=True)
    print(f"[setup] ComfyUI headless fixado em {COMFYUI_COMMIT[:12]}.")


def validate_runtime() -> None:
    """Valida o runtime necessário ao loader nativo Anima/ComfyUI."""
    import torch
    try:
        from Crypto.Cipher import AES
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyCryptodome não foi carregado; reinicie o runtime Colab e execute a célula novamente."
        ) from error
    try:
        import trampoline  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "A dependência trampoline não foi instalada; execute novamente o launcher para corrigir o ambiente."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não encontrada. Selecione T4 no Colab e reinicie o ambiente.")
    from packaging.version import Version
    transformers_version = Version(importlib.metadata.version("transformers"))
    if transformers_version < Version("4.51.0"):
        raise RuntimeError(
            f"Transformers incompatível: encontrado {transformers_version}, mínimo 4.51.0 para o encoder Qwen do Anima. "
            "Reinicie o ambiente Colab e execute esta célula novamente."
        )
    safetensors_version = Version(importlib.metadata.version("safetensors"))
    if safetensors_version < Version("0.8.0"):
        raise RuntimeError(
            f"Safetensors incompatível: encontrado {safetensors_version}, mínimo 0.8.0 para o loader Anima. "
            "Reinicie o ambiente Colab e execute esta célula novamente."
        )
    print(
        f"[setup] Runtime validado: torch={torch.__version__}, transformers={transformers_version}, "
        f"safetensors={safetensors_version}, crypto=AES"
    )


def ensure_cloudflared() -> str:
    existing = shutil.which("cloudflared") or os.environ.get("CLOUDFLARED_BIN")
    if existing and Path(existing).exists():
        return existing
    print("[setup] Instalando o cliente cloudflared…")
    binary = Path("/usr/local/bin/cloudflared")
    subprocess.run([
        "curl", "-L", "--fail", "--silent", "--show-error",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "-o", str(binary),
    ], check=True)
    binary.chmod(0o755)
    return str(binary)


def pipe_output(process: subprocess.Popen[str], label: str, on_line=None) -> None:
    assert process.stdout is not None
    for raw in iter(process.stdout.readline, ""):
        line = raw.rstrip()
        if line:
            print(f"[{label}] {line}")
            if on_line:
                on_line(line)


def wait_for_server(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    last_error = "a porta ainda não aceitou conexões"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("O servidor encerrou antes de responder. Leia as linhas [server] acima.")
        try:
            response = requests.get(f"http://127.0.0.1:{PORT}/api/health", timeout=1.5)
            if response.ok:
                return
            last_error = f"health HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)[:160]
        time.sleep(0.5)
    raise RuntimeError(
        f"O servidor não respondeu em {SERVER_START_TIMEOUT:g}s ({last_error}). "
        "A inicialização do modelo/MEGA pode continuar; verifique as linhas [server] acima."
    )


def main() -> None:
    if not os.path.exists("/content"):
        print("Aviso: este inicializador foi desenhado para Google Colab.")
    ask_secret("STUDIO_PASSWORD", "Defina a senha de acesso ao painel: ")
    ask_secret("MEGA_EMAIL", "E-mail da conta MEGA: ")
    ask_secret("MEGA_PASSWORD", "Senha da conta MEGA: ")
    ask_secret("CIVITAI_TOKEN", "Token Civitai (Enter para continuar sem token): ", required=False)

    # Configuração técnica padronizada: Nova EXAnime AM é Anima bf16;
    # o loader nativo do ComfyUI faz o cast FP16 compatível com a T4.
    os.environ.setdefault("STUDIO_ROOT", "/content/modellab-studio")
    if "HF_HOME" not in os.environ:
        legacy_hf_home = Path.home() / ".cache" / "huggingface"
        default_hf_home = Path(os.environ["STUDIO_ROOT"]) / "huggingface-cache"
        # Reaproveita downloads feitos pela execução anterior antes de criar um cache novo.
        os.environ["HF_HOME"] = str(legacy_hf_home if legacy_hf_home.exists() else default_hf_home)
    os.environ.setdefault("HF_HUB_CACHE", f"{os.environ['HF_HOME']}/hub")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("MEGA_FOLDER", "ModelLabStudio")
    # Substitui defaults antigos persistidos no runtime; perfis customizados ainda
    # podem ser fornecidos por MODELS_CONFIG.
    os.environ["MODEL_ID"] = "nova-exanime-am"
    os.environ["MODEL_URL"] = "https://civitai.com/api/download/models/3226184?fileId=3108312"
    os.environ["MODEL_REPO"] = ""
    os.environ["MODEL_PATH"] = f"{os.environ['STUDIO_ROOT']}/models/diffusion_models/novaExanimeAM_v10.safetensors"
    os.environ["MODEL_FAMILY"] = "anima"
    os.environ["COMFYUI_DIR"] = str(COMFYUI_DIR)
    # O base-directory do ComfyUI coincide com STUDIO_ROOT para compartilhar
    # models/diffusion_models, models/text_encoders e models/vae.
    os.environ["COMFY_ROOT"] = os.environ["STUDIO_ROOT"]
    print("[setup] Perfil Nova EXAnime AM padronizado; backend ComfyUI headless e modelo residente na GPU configurados.")

    install_requirements()
    ensure_comfyui()
    validate_runtime()
    cloudflared = ensure_cloudflared()
    print(f"[setup] Iniciando ModelLab Studio na GPU atual (aguardando até {SERVER_START_TIMEOUT:g}s)…")
    server = subprocess.Popen(
        [sys.executable, "server.py"], cwd=APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    threading.Thread(target=pipe_output, args=(server, "server"), daemon=True).start()
    wait_for_server(server)

    current_url = APP_DIR / "current_tunnel_url.txt"
    tunnel: subprocess.Popen[str] | None = None
    shutdown_requested = False

    def start_tunnel() -> subprocess.Popen[str]:
        nonlocal tunnel
        current_url.unlink(missing_ok=True)
        public_url: list[str] = []

        def capture_url(line: str) -> None:
            match = TUNNEL_PATTERN.search(line)
            if match and not public_url:
                public_url.append(match.group(0))
                current_url.write_text(public_url[0] + "\n", encoding="utf-8")
                print("\n" + "=" * 72)
                print(f"URL ATUAL DO PAINEL: {public_url[0]}")
                print("Abra exatamente esta URL; ela muda quando o túnel é recriado.")
                print("Use a senha definida nesta célula. Pare a célula para encerrar o nó.")
                print("=" * 72 + "\n")

        tunnel = subprocess.Popen(
            [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{PORT}"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(target=pipe_output, args=(tunnel, "tunnel", capture_url), daemon=True).start()
        return tunnel

    try:
        tunnel = start_tunnel()
        while server.poll() is None:
            if tunnel.poll() is not None:
                print("[tunnel] O cloudflared encerrou; criando um novo endereço público…")
                time.sleep(2)
                tunnel = start_tunnel()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_requested = True
        print("\n[shutdown] Encerrando túnel e servidor…")
    finally:
        for process in (tunnel, server):
            if process is not None and process.poll() is None:
                process.terminate()
        current_url.unlink(missing_ok=True)
        for process in (tunnel, server):
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
