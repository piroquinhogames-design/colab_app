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
TUNNEL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com", re.I)


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


def ensure_cloudflared() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    print("[setup] Instalando o cliente de túnel temporário…")
    package = "/tmp/cloudflared.deb"
    subprocess.run([
        "bash", "-lc",
        "curl -L --fail --silent --show-error https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && dpkg -i /tmp/cloudflared.deb",
    ], check=True)
    return shutil.which("cloudflared") or "/usr/local/bin/cloudflared"


def pipe_output(process: subprocess.Popen[str], label: str, on_line=None) -> None:
    assert process.stdout is not None
    for raw in iter(process.stdout.readline, ""):
        line = raw.rstrip()
        if line:
            print(f"[{label}] {line}")
            if on_line:
                on_line(line)


def wait_for_server(process: subprocess.Popen[str]) -> None:
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError("O servidor encerrou antes de responder. Leia as linhas [server] acima.")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/api/health", timeout=1.5).ok:
                return
        except requests.RequestException:
            time.sleep(.5)
    raise RuntimeError("O servidor não respondeu a tempo.")


def main() -> None:
    if not os.path.exists("/content"):
        print("Aviso: este inicializador foi desenhado para Google Colab.")
    ask_secret("STUDIO_PASSWORD", "Defina a senha de acesso ao painel: ")
    ask_secret("MEGA_EMAIL", "E-mail da conta MEGA: ")
    ask_secret("MEGA_PASSWORD", "Senha da conta MEGA: ")
    ask_secret("CIVITAI_TOKEN", "Token Civitai (Enter para continuar sem token): ", required=False)
    os.environ.setdefault("STUDIO_ROOT", "/content/illustrious-studio")
    os.environ.setdefault("MEGA_FOLDER", "IllustriousStudio")

    install_requirements()
    validate_runtime()
    cloudflared = ensure_cloudflared()
    print("[setup] Iniciando Illustrious LoRA Studio na GPU atual…")
    server = subprocess.Popen(
        [sys.executable, "server.py"], cwd=APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    threading.Thread(target=pipe_output, args=(server, "server"), daemon=True).start()
    wait_for_server(server)

    public_url: list[str] = []
    def capture_url(line: str) -> None:
        match = TUNNEL_PATTERN.search(line)
        if match and not public_url:
            public_url.append(match.group(0))
            print("\n" + "=" * 72)
            print(f"PAINEL PUBLICADO: {public_url[0]}")
            print("Use a senha definida nesta célula. Pare a célula para encerrar o nó.")
            print("=" * 72 + "\n")

    tunnel = subprocess.Popen(
        [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{PORT}"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    threading.Thread(target=pipe_output, args=(tunnel, "tunnel", capture_url), daemon=True).start()
    try:
        while server.poll() is None and tunnel.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[shutdown] Encerrando túnel e servidor…")
    finally:
        for process in (tunnel, server):
            if process.poll() is None:
                process.terminate()
        for process in (tunnel, server):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
