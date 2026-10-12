"""Execute este arquivo em uma célula Colab depois de enviar a pasta colab_app.

Exemplo: !python /content/colab_app/launch_colab.py
Interrompa a célula para encerrar servidor e todos os túneis.
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


def ask_comfy_exposure() -> bool:
    """Pede consentimento novo em cada execução; ausência de resposta é não."""
    print("\n[setup] Integração externa com SillyTavern (opcional).")
    print("AVISO: esta opção cria uma URL pública do ComfyUI SEM autenticação.")
    print("Quem tiver o link poderá gerar/cancelar imagens, consumir sua GPU e acessar arquivos servidos pelo ComfyUI.")
    print("A senha do painel NÃO protege esta API. Ative apenas se aceitar esse acesso nesta sessão.")
    while True:
        try:
            answer = input("Expor a API ComfyUI publicamente nesta sessão? [sim/NÃO]: ").strip().casefold()
        except (EOFError, OSError):
            print("[setup] Entrada indisponível; API ComfyUI externa desativada.")
            return False
        if answer in {"sim", "s", "yes", "y"}:
            print("[setup] Exposição pública aceita para esta sessão.")
            return True
        if answer in {"", "não", "nao", "n", "no"}:
            print("[setup] API ComfyUI externa desativada.")
            return False
        print("Responda sim ou não. Enter mantém a API externa desativada.")


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
    try:
        for raw in iter(process.stdout.readline, ""):
            line = raw.rstrip()
            if line:
                print(f"[{label}] {line}")
                if on_line:
                    on_line(line)
    finally:
        process.stdout.close()


class QuickTunnel:
    """Supervisiona um túnel sem bloquear a recuperação dos demais serviços."""

    def __init__(self, binary: str, port: int, name: str, url_file: Path, on_url) -> None:
        self.binary, self.port, self.name = binary, port, name
        self.url_file, self.on_url = url_file, on_url
        self.process: subprocess.Popen[str] | None = None
        self.url = ""
        self.failures = 0
        self.retry_at = 0.0
        self.lock = threading.Lock()
        self.stopped = False
        self.url_file.unlink(missing_ok=True)

    def _capture_url(self, process, line: str) -> None:
        match = TUNNEL_PATTERN.search(line)
        if not match:
            return
        with self.lock:
            # Uma thread do processo anterior não pode publicar um link antigo.
            if self.stopped or process is not self.process or self.url or process.poll() is not None:
                return
            self.url = match.group(0)
            self.url_file.write_text(self.url + "\n", encoding="utf-8")
            self.on_url(self.url)

    def _retry(self) -> None:
        self.failures += 1
        self.retry_at = time.monotonic() + min(2 ** min(self.failures, 5), 30)
        print(f"[{self.name}] Túnel indisponível; um novo endereço será criado em instantes.", flush=True)

    def service(self) -> None:
        if self.stopped:
            return
        if self.process is not None:
            if self.process.poll() is None:
                return
            with self.lock:
                self.process = None
                self.url = ""
                self.url_file.unlink(missing_ok=True)
            self._retry()
            return
        if time.monotonic() < self.retry_at:
            return
        try:
            process = subprocess.Popen(
                [self.binary, "tunnel", "--url", f"http://127.0.0.1:{self.port}"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            print(f"[{self.name}] Falha ao iniciar cloudflared: {exc}", flush=True)
            self._retry()
            return
        with self.lock:
            self.process = process
        threading.Thread(
            target=pipe_output,
            args=(process, self.name, lambda line: self._capture_url(process, line)),
            daemon=True,
        ).start()

    def close(self) -> None:
        with self.lock:
            self.stopped = True
            process, self.process = self.process, None
            self.url = ""
            self.url_file.unlink(missing_ok=True)
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


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
    expose_comfy = ask_comfy_exposure()
    # Nunca reutiliza uma autorização de uma execução/ambiente anterior.
    os.environ["STUDIO_PREPARE_COMFY"] = "1" if expose_comfy else "0"
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
        [sys.executable, "-u", "server.py"], cwd=APP_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True,
    )
    threading.Thread(target=pipe_output, args=(server, "server"), daemon=True).start()

    def show_panel_url(url: str) -> None:
        print(f"\nURL ATUAL DO PAINEL: {url}", flush=True)
        print("Use a senha definida nesta célula. O endereço muda quando o túnel é recriado.", flush=True)
        if expose_comfy:
            print(f"WORKFLOW SILLYTAVERN (entre no painel antes de baixar): {url}/api/sillytavern/workflow", flush=True)
        print("Pare a célula para encerrar o servidor e todos os túneis.\n", flush=True)

    def show_comfy_url(url: str) -> None:
        print(f"\nURL DO COMFYUI PARA SILLYTAVERN: {url}", flush=True)
        print("Source: ComfyUI | Server Type: Standard Server | ComfyUI URL: o endereço acima.", flush=True)
        print("Use o workflow ModelLab_SillyTavern.json; comece com 768×768, 24 passos, CFG 5, euler_ancestral / normal.", flush=True)
        print("Este acesso à API é público, sem senha. Se o túnel reiniciar, atualize a URL no SillyTavern.\n", flush=True)

    panel_tunnel = QuickTunnel(cloudflared, int(PORT), "tunnel-painel", APP_DIR / "current_tunnel_url.txt", show_panel_url)
    comfy_url_file = APP_DIR / "current_comfy_tunnel_url.txt"
    comfy_url_file.unlink(missing_ok=True)
    comfy_tunnel = QuickTunnel(
        cloudflared, int(os.environ.get("COMFY_PORT", "8188")), "tunnel-comfyui", comfy_url_file, show_comfy_url,
    ) if expose_comfy else None
    comfy_ready = False
    comfy_failed = False

    try:
        wait_for_server(server)
        if expose_comfy:
            print("[setup] Preparando modelo, encoder e VAE; o link ComfyUI será publicado quando o backend estiver pronto.", flush=True)
        while server.poll() is None:
            panel_tunnel.service()
            if comfy_tunnel is not None and not comfy_ready and not comfy_failed:
                try:
                    response = requests.get(f"http://127.0.0.1:{PORT}/api/health", timeout=2)
                    response.raise_for_status()
                    status = response.json().get("public_comfy", {}).get("status")
                    comfy_ready = status == "ready"
                    if status == "failed":
                        comfy_failed = True
                        print("[setup] Falha ao preparar ComfyUI externo; nenhum túnel de API foi aberto. Consulte o erro [comfy-setup] acima.", flush=True)
                except (requests.RequestException, ValueError):
                    pass
            if comfy_tunnel is not None and comfy_ready:
                comfy_tunnel.service()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[shutdown] Encerrando túneis e servidor…", flush=True)
    finally:
        # Interrompe também downloads/ComfyUI iniciados no grupo do servidor.
        try: os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        panel_tunnel.close()
        if comfy_tunnel is not None:
            comfy_tunnel.close()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=5)


def handle_shutdown(_signum, _frame) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_shutdown)
    main()
