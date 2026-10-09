"""Execute este arquivo em uma célula Colab depois de enviar a pasta colab_app.

Exemplo: !python /content/colab_app/launch_colab.py
Interrompa a célula para encerrar servidor e túnel.
"""

from __future__ import annotations

import getpass
import json
import signal
import tempfile
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
    print("[setup] Resolvendo dependências sem substituir o stack GPU do Colab…")
    protected = {}
    for name in ("torch", "torchvision", "torchaudio"):
        try: protected[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "torch": raise RuntimeError("PyTorch ausente. Use um runtime Colab com GPU.")
    with tempfile.TemporaryDirectory(prefix="modellab-deps-") as temporary:
        constraints = Path(temporary) / "constraints.txt"
        base = (APP_DIR / "runtime_constraints.txt").read_text()
        constraints.write_text(base + "\n" + "\n".join(f"{key}=={value}" for key, value in protected.items()))
        # mega.py is a vendored compatibility wrapper; its obsolete tenacity pin
        # is deliberately not imposed on the rest of the runtime.
        studio = Path(temporary) / "requirements.txt"
        studio.write_text("\n".join(line for line in (APP_DIR / "requirements.txt").read_text().splitlines() if not line.startswith("mega.py")))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
            "-c", str(constraints), "-r", str(studio), "-r", str(APP_DIR / "comfy_requirements.txt"), "tenacity>=8,<10"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "mega.py==1.0.8"], check=True)
    if any(importlib.metadata.version(key) != value for key, value in protected.items()):
        raise RuntimeError("O stack GPU foi alterado inesperadamente; reinicie o runtime.")
    validate_pydantic_runtime()
    subprocess.run([sys.executable, "-c", "import transformers, tokenizers, huggingface_hub, aiohttp, yarl, sqlalchemy, alembic, torchsde, trampoline, websocket; from Crypto.Cipher import AES"], check=True)
    versions = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions() if distribution.metadata.get("Name")}
    diagnostic = Path(os.environ.get("STUDIO_ROOT", "/content/modellab-studio")) / "runtime_versions.json"
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.write_text(json.dumps({"python": sys.version, "comfyui_commit": COMFYUI_COMMIT, "packages": versions}, indent=2))


def validate_pydantic_runtime() -> None:
    """Teste em processo novo para evitar módulos antigos no cache de imports."""
    result = subprocess.run([
        sys.executable, "-c",
        "import pydantic, pydantic_core; from pydantic_settings import BaseSettings; "
        "from pydantic import TypeAdapter; "
        "assert TypeAdapter(int).validate_python('7') == 7; "
        "print('pydantic=' + pydantic.__version__ + ', core=' + pydantic_core.__version__)",
    ], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Falha na validação das dependências Pydantic antes de iniciar ComfyUI: " + result.stderr[-3000:])
    print("[setup] " + result.stdout.strip())


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
    from studio_downloads import ensure_download
    custom = os.environ.get("CLOUDFLARED_BIN")
    if custom:
        if not Path(custom).is_file(): raise RuntimeError("CLOUDFLARED_BIN não existe.")
        return custom
    binary = Path(os.environ.get("STUDIO_ROOT", "/content/modellab-studio")) / "bin" / "cloudflared"
    ensure_download("https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/cloudflared-linux-amd64", binary, 1,
        sha256="d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db")
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

    # WAI-ANIMA FP16 é o default; configurações explícitas são preservadas.
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
    os.environ.setdefault("MODEL_ID", "wai-anima")
    os.environ.setdefault("MODEL_URL", "https://civitai.com/api/download/models/2983680?fileId=2863158")
    os.environ.setdefault("MODEL_REPO", "")
    os.environ.setdefault("MODEL_PATH", f"{os.environ['STUDIO_ROOT']}/models/diffusion_models/WAI-ANIMA1.safetensors")
    os.environ.setdefault("MODEL_FAMILY", "anima")
    os.environ["COMFYUI_DIR"] = str(COMFYUI_DIR)
    # O base-directory do ComfyUI coincide com STUDIO_ROOT para compartilhar
    # models/diffusion_models, models/text_encoders e models/vae.
    os.environ.setdefault("COMFY_ROOT", os.environ["STUDIO_ROOT"])
    os.environ.setdefault("STUDIO_TRUST_TUNNEL", "1")
    os.environ.setdefault("STUDIO_COOKIE_SECURE", "1")
    print("[setup] Perfil WAI-ANIMA v1.0 padronizado; backend ComfyUI headless e modelo residente na GPU configurados.")

    install_requirements()
    ensure_comfyui()
    validate_runtime()
    cloudflared = ensure_cloudflared()
    print(f"[setup] Iniciando ModelLab Studio na GPU atual (aguardando até {SERVER_START_TIMEOUT:g}s)…")
    server = subprocess.Popen(
        [sys.executable, "server.py"], cwd=APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True,
    )
    threading.Thread(target=pipe_output, args=(server, "server"), daemon=True).start()

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
        wait_for_server(server)
        tunnel = start_tunnel()
        tunnel_failures = 0
        while server.poll() is None:
            if tunnel.poll() is not None:
                print("[tunnel] O cloudflared encerrou; criando um novo endereço público…")
                tunnel_failures += 1
                time.sleep(min(2 ** min(tunnel_failures, 5), 30))
                tunnel = start_tunnel()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_requested = True
        print("\n[shutdown] Encerrando túnel e servidor…")
    finally:
        for process in (tunnel, server):
            if process is server and process is not None:
                try: os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            elif process is not None and process.poll() is None:
                process.terminate()
        current_url.unlink(missing_ok=True)
        for process in (tunnel, server):
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if process is server: os.killpg(server.pid, signal.SIGKILL)
                    else: process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
