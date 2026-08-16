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
import zipfile
from pathlib import Path

import requests


APP_DIR = Path(__file__).resolve().parent
PORT = os.environ.get("PORT", "7860")
SERVER_START_TIMEOUT = float(os.environ.get("SERVER_START_TIMEOUT", "180"))
TUNNEL_PATTERN = re.compile(r"https://[-a-z0-9]+\.(?:ngrok-free\.app|ngrok\.io)", re.I)


def ask_secret(name: str, prompt: str, required: bool = True) -> None:
    if os.environ.get(name):
        return
    value = getpass.getpass(prompt)
    if required and not value:
        raise RuntimeError(f"{name} é obrigatório.")
    if value:
        os.environ[name] = value


def install_requirements() -> None:
    print("[setup] Atualizando a matriz direta Diffusers/Transformers/PEFT do estúdio…")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps",
        "-r", str(APP_DIR / "requirements.txt"),
    ], check=True)


def validate_runtime() -> None:
    """Falha cedo quando a sessão Colab mantém uma combinação incompatível para LoRA."""
    import torch
    import diffusers
    import peft
    try:
        from Crypto.Cipher import AES
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyCryptodome não foi carregado; reinicie o runtime Colab e execute a célula novamente."
        ) from error

    expected_peft = "0.17.0"
    installed_peft = importlib.metadata.version("peft")
    if installed_peft != expected_peft:
        raise RuntimeError(
            f"PEFT incompatível: encontrado {installed_peft}, esperado {expected_peft}. "
            "Reinicie o ambiente Colab e execute esta célula novamente."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA não encontrada. Selecione T4 no Colab e reinicie o ambiente.")
    transformers_version = importlib.metadata.version("transformers")
    if transformers_version != "4.48.3":
        raise RuntimeError(
            f"Transformers incompatível: encontrado {transformers_version}, esperado 4.48.3. "
            "Reinicie o ambiente Colab e execute esta célula novamente."
        )
    print(
        f"[setup] Runtime validado: torch={torch.__version__}, diffusers={diffusers.__version__}, "
        f"transformers={transformers_version}, peft={peft.__version__}, crypto=AES"
    )


def ensure_ngrok() -> str:
    existing = shutil.which("ngrok") or os.environ.get("NGROK_BIN")
    if existing and Path(existing).exists():
        return existing
    print("[setup] Instalando o cliente ngrok…")
    archive = Path("/tmp/ngrok.zip")
    binary = Path("/usr/local/bin/ngrok")
    subprocess.run([
        "curl", "-L", "--fail", "--silent", "--show-error",
        "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip",
        "-o", str(archive),
    ], check=True)
    with zipfile.ZipFile(archive) as package:
        package.extract("ngrok", "/tmp")
    shutil.copy2("/tmp/ngrok", binary)
    binary.chmod(0o755)
    return str(binary)


def configure_ngrok(ngrok: str) -> None:
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "NGROK_AUTHTOKEN não configurado. Crie um token em https://dashboard.ngrok.com/get-started/your-authtoken "
            "e defina-o no Colab antes de executar o inicializador."
        )
    subprocess.run([ngrok, "config", "add-authtoken", token], check=True, capture_output=True, text=True)


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
    os.environ.setdefault("STUDIO_ROOT", "/content/modellab-studio")
    os.environ.setdefault("MEGA_FOLDER", "ModelLabStudio")

    install_requirements()
    validate_runtime()
    ngrok = ensure_ngrok()
    configure_ngrok(ngrok)
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
            [ngrok, "http", PORT, "--log=stdout", "--log-format=logfmt"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(target=pipe_output, args=(tunnel, "tunnel", capture_url), daemon=True).start()
        return tunnel

    try:
        tunnel = start_tunnel()
        while server.poll() is None:
            if tunnel.poll() is not None:
                print("[tunnel] O ngrok encerrou; criando um novo endereço público…")
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
