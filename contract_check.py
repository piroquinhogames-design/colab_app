"""Testes de contrato executados pelo Vitest contra rotas reais do Flask.

O módulo MEGA é substituído apenas neste teste para que nenhum serviço externo,
credencial ou download de modelo seja acionado no ambiente de CI.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import asyncio
from pathlib import Path


class _UnusedMega:
    def login(self, *_args, **_kwargs):
        raise AssertionError("O teste não deve tentar conectar ao MEGA")


sys.modules["mega"] = types.SimpleNamespace(Mega=_UnusedMega)
if hasattr(asyncio, "coroutine"):
    delattr(asyncio, "coroutine")
root = Path(tempfile.mkdtemp(prefix="illustrious-contract-"))
os.environ.update({
    "STUDIO_ROOT": str(root),
    "STUDIO_PASSWORD": "contract-password",
    "CIVITAI_TOKEN": "never-expose-this-token",
})
sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402
import launch_colab  # noqa: E402
if not hasattr(asyncio, "coroutine"):
    raise AssertionError("O adaptador de compatibilidade Python 3.12 não foi aplicado")


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: esperado {expected!r}, recebido {actual!r}")


def main() -> None:
    package_root = Path(__file__).resolve().parent
    requirements = (package_root / "requirements.txt").read_text(encoding="utf-8")
    if "peft==0.17.0" not in requirements:
        raise AssertionError("O pacote deve fixar a versão PEFT compatível com carregamento de LoRA")
    if "pycryptodome==3.21.0" not in requirements:
        raise AssertionError("O pacote deve instalar pycryptodome para o módulo Crypto exigido pelo MEGA")
    if any(line.strip().startswith("torch") for line in requirements.splitlines()):
        raise AssertionError("O pacote não deve atualizar o PyTorch/CUDA que já vem com o Colab")
    launcher_source = (package_root / "launch_colab.py").read_text(encoding="utf-8")
    if "def validate_runtime()" not in launcher_source or "PEFT incompatível" not in launcher_source:
        raise AssertionError("O inicializador deve validar PEFT antes de iniciar o servidor")
    if '"pip", "check"' in launcher_source:
        raise AssertionError("O inicializador não deve falhar por conflitos globais do Colab")
    if '"-m", "venv"' in launcher_source or "VENV_PYTHON" in launcher_source:
        raise AssertionError("O inicializador não pode depender de venv, pois ensurepip falha no Colab")
    if '"--upgrade", "--no-deps"' not in launcher_source:
        raise AssertionError("O instalador deve preservar dependências globais do Colab com --no-deps")
    if 'transformers_version != "4.48.3"' not in launcher_source:
        raise AssertionError("O runtime deve confirmar a versão de Transformers compatível com PEFT")
    if "from Crypto.Cipher import AES" not in launcher_source:
        raise AssertionError("O runtime deve validar o módulo Crypto exigido pelo cliente MEGA")
    server_source = (package_root / "server.py").read_text(encoding="utf-8")
    if ".enable_vae_slicing()" in server_source or ".vae.enable_slicing()" not in server_source:
        raise AssertionError("O servidor deve usar a API VAE atual, não o atalho obsoleto do Diffusers")

    isolated_root = Path(tempfile.mkdtemp(prefix="illustrious-install-contract-"))
    original_app_dir = launch_colab.APP_DIR
    calls: list[list[str]] = []
    original_run = launch_colab.subprocess.run
    try:
        (isolated_root / "requirements.txt").write_text("", encoding="utf-8")
        launch_colab.APP_DIR = isolated_root
        launch_colab.subprocess.run = lambda command, **_: calls.append(command)
        launch_colab.install_requirements()
        if len(calls) != 1 or calls[0][0] != sys.executable or "--no-deps" not in calls[0] or "--upgrade" not in calls[0]:
            raise AssertionError("O instalador deve atualizar somente os pacotes diretos via pip global")
    finally:
        launch_colab.APP_DIR = original_app_dir
        launch_colab.subprocess.run = original_run

    anonymous = server.app.test_client()
    assert_equal(anonymous.get("/api/history").status_code, 401, "Histórico deve exigir sessão")

    client = server.app.test_client()
    failed = client.post("/api/login", json={"password": "wrong"})
    assert_equal(failed.status_code, 401, "Senha inválida deve ser rejeitada")
    logged = client.post("/api/login", json={"password": "contract-password"})
    assert_equal(logged.status_code, 200, "Senha válida deve criar sessão")
    csrf = logged.get_json()["csrf"]

    boot = client.get("/api/bootstrap")
    assert_equal(boot.status_code, 200, "Bootstrap autenticado deve responder")
    boot_text = boot.get_data(as_text=True)
    if "never-expose-this-token" in boot_text:
        raise AssertionError("O token Civitai vazou no payload bootstrap")

    rejected = client.post("/api/jobs", json={"prompt": ""}, headers={"X-CSRF-Token": csrf})
    assert_equal(rejected.status_code, 400, "Prompt vazio deve ser rejeitado")
    blocked = client.post("/api/jobs", json={"prompt": "test"})
    assert_equal(blocked.status_code, 403, "Mutação sem CSRF deve ser bloqueada")

    called = {}
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return {"items": [{
                "id": 42, "name": "Adapter", "creator": {"username": "artist"}, "tags": ["style"],
                "modelVersions": [
                    {"id": 73, "name": "IFL", "baseModel": "Illustrious", "images": [], "stats": {"downloadCount": 8}},
                    {"id": 74, "name": "LMB v2", "baseModel": "Illustrious", "images": [], "stats": {"downloadCount": 5}},
                    {"id": 75, "name": "Outra base", "baseModel": "SDXL 1.0", "images": [], "stats": {"downloadCount": 99}},
                ],
            }], "metadata": {"nextCursor": "next-page"}}
    def fake_get(url, params, headers, timeout):
        called.update({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return Response()
    original_get = server.requests.get
    server.requests.get = fake_get
    try:
        catalog = client.get("/api/catalog?tag=portrait")
    finally:
        server.requests.get = original_get
    assert_equal(catalog.status_code, 200, "Catálogo deve responder")
    assert_equal(called["params"]["baseModels"], "Illustrious", "Catálogo deve filtrar Illustrious")
    catalog_json = catalog.get_json()
    assert_equal(catalog_json["items"][0]["version_id"], 73, "Versão padrão de LoRA deve ser exposta")
    assert_equal([item["id"] for item in catalog_json["items"][0]["versions"]], [73, 74], "Todas as versões Illustrious devem ser expostas")
    assert_equal(catalog_json["items"][0]["versions"][1]["name"], "LMB v2", "Nome da versão deve ser preservado")
    assert_equal(catalog_json["next_cursor"], "next-page", "Cursor deve ser preservado")
    assert_equal(called["params"]["nsfw"], "false", "Catálogo padrão deve declarar nsfw=false")
    server.requests.get = fake_get
    try:
        adult_catalog = client.get("/api/catalog?include_adult=true")
    finally:
        server.requests.get = original_get
    assert_equal(adult_catalog.status_code, 200, "Catálogo adulto deve responder")
    assert_equal(called["params"]["nsfw"], "true", "Catálogo adulto deve solicitar nsfw=true")
    assert_equal(adult_catalog.get_json()["catalog_query"]["nsfw"], "true", "Resposta deve expor o filtro adulto aplicado")

    empty_response = server.MegaArchive._connection_error(ValueError("Expecting value: line 1 column 1 (char 0)"))
    if "resposta vazia" not in empty_response or "MEGA_EMAIL/MEGA_PASSWORD" not in empty_response:
        raise AssertionError("Falha de autenticação vazia deve gerar diagnóstico acionável sem credenciais")
    missing_credentials = server.MegaArchive()
    missing_credentials.connect()
    if missing_credentials.available or "MEGA_EMAIL" not in (missing_credentials.error or ""):
        raise AssertionError("Credenciais ausentes devem manter MEGA indisponível com instrução segura")
    original_mega = server.Mega
    original_login_adapter = server.MegaArchive._login_with_http_adapter
    original_email = os.environ.get("MEGA_EMAIL")
    original_password = os.environ.get("MEGA_PASSWORD")
    try:
        server.MegaArchive._login_with_http_adapter = staticmethod(
            lambda *_args: (_ for _ in ()).throw(ValueError("Invalid credentials for MEGA"))
        )
        os.environ["MEGA_EMAIL"] = "user@example.invalid"
        os.environ["MEGA_PASSWORD"] = "not-a-real-password"
        invalid_credentials = server.MegaArchive()
        invalid_credentials.connect()
        if invalid_credentials.available or "Invalid credentials" not in (invalid_credentials.error or ""):
            raise AssertionError("Credenciais inválidas devem gerar erro explícito sem ativar o arquivo")
        if "not-a-real-password" in (invalid_credentials.error or ""):
            raise AssertionError("A mensagem MEGA não pode expor a senha")
    finally:
        server.Mega = original_mega
        server.MegaArchive._login_with_http_adapter = staticmethod(original_login_adapter)
        if original_email is None: os.environ.pop("MEGA_EMAIL", None)
        else: os.environ["MEGA_EMAIL"] = original_email
        if original_password is None: os.environ.pop("MEGA_PASSWORD", None)
        else: os.environ["MEGA_PASSWORD"] = original_password

    class ReadyMegaClient:
        def find(self, _name):
            return {"name": "IllustriousStudio", "h": "folder-node"}
    flaky_calls = {"count": 0}
    def flaky_login(*_args):
        flaky_calls["count"] += 1
        if flaky_calls["count"] == 1:
            raise server.MegaHttpError("HTTP 200; resposta vazia")
        return ReadyMegaClient()
    try:
        server.MegaArchive._login_with_http_adapter = staticmethod(flaky_login)
        os.environ["MEGA_EMAIL"] = "user@example.invalid"
        os.environ["MEGA_PASSWORD"] = "not-a-real-password"
        retry_archive = server.MegaArchive()
        if not retry_archive.connect(attempts=2, retry_delay=0):
            raise AssertionError("O MEGA deve recuperar a conexão após uma resposta vazia transitória")
        assert_equal(flaky_calls["count"], 2, "O login MEGA deve repetir a tentativa transitória")
        if not retry_archive.available or not retry_archive.ensure_connected():
            raise AssertionError("Reconexão bem-sucedida deve manter o arquivo MEGA disponível")
    finally:
        server.Mega = original_mega
        server.MegaArchive._login_with_http_adapter = staticmethod(original_login_adapter)
        if original_email is None: os.environ.pop("MEGA_EMAIL", None)
        else: os.environ["MEGA_EMAIL"] = original_email
        if original_password is None: os.environ.pop("MEGA_PASSWORD", None)
        else: os.environ["MEGA_PASSWORD"] = original_password

    class HttpMega:
        def __init__(self):
            self.schema, self.domain, self.sequence_num = "https", "mega.test", 7
            self.timeout, self.sid = 1, None
        def login(self, *_args):
            self._api_request({"a": "us"})
            return self
    class HttpResponse:
        def __init__(self, status_code, text, headers=None):
            self.status_code, self.text = status_code, text
            self.headers = headers or {}
    request_log = []
    original_post = server.requests.post
    try:
        server.Mega = HttpMega
        server.requests.post = lambda *args, **kwargs: (request_log.append((args, kwargs)) or HttpResponse(200, ""))
        try:
            server.MegaArchive._login_with_http_adapter("user@example.invalid", "not-a-real-password")
            raise AssertionError("Resposta HTTP vazia deve falhar antes do parse JSON")
        except server.MegaHttpError as error:
            if "operação us; HTTP 200; resposta vazia na repetição direta; resposta com 0 bytes" not in str(error):
                raise AssertionError("O adaptador deve informar status e operação para resposta vazia")
        if len(request_log) != 2 or request_log[-1][1]["headers"].get("Accept-Encoding") != "identity":
            raise AssertionError("Resposta vazia deve tentar uma conexão direta sem compressão")

        def assert_diagnostic(response, expected: str) -> None:
            server.requests.post = lambda *args, **kwargs: HttpResponse(response.status_code, response.text)
            try:
                server.MegaArchive._login_with_http_adapter("user@example.invalid", "not-a-real-password")
                raise AssertionError("Resposta HTTP inválida deve falhar")
            except server.MegaHttpError as error:
                if expected not in str(error):
                    raise AssertionError(f"Diagnóstico HTTP incompleto: {error}")

        assert_diagnostic(HttpResponse(503, "offline"), "operação us; HTTP 503; status não-2xx; resposta com 7 bytes")
        assert_diagnostic(HttpResponse(200, "not-json"), "operação us; HTTP 200; resposta não JSON; resposta com 8 bytes")
        assert_diagnostic(HttpResponse(200, "[]"), "operação us; HTTP 200; formato JSON inesperado; resposta com 2 bytes")
        response_sequence = iter([HttpResponse(200, ""), HttpResponse(200, "[{}]")])
        server.requests.post = lambda *args, **kwargs: (request_log.append((args, kwargs)) or next(response_sequence))
        server.MegaArchive._login_with_http_adapter("user@example.invalid", "not-a-real-password")
        if not request_log or request_log[-1][1]["headers"]["Content-Type"] != "application/json":
            raise AssertionError("O adaptador MEGA deve enviar JSON com cabeçalho explícito")

        original_hashcash_solver = server.MegaArchive._solve_hashcash
        original_sha256 = server.hashlib.sha256
        hashcash_challenge = "1:192:unused:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        try:
            class PrefixDigest:
                def digest(self):
                    return b"\x00\x00\x00\x00" + (b"\xff" * 28)
            server.hashlib.sha256 = lambda _payload: PrefixDigest()
            protocol_proof = server.MegaArchive._solve_hashcash(
                "1:0:unused:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            )
            if not protocol_proof.startswith("1:AAAAAAAA") or not protocol_proof.endswith(":AAAAAA"):
                raise AssertionError("A prova Hashcash deve usar os quatro bytes iniciais e nonce Base64 URL-safe")
            urlsafe_proof = server.MegaArchive._solve_hashcash("1:0:unused:____")
            if not urlsafe_proof.startswith("1:____:"):
                raise AssertionError("O seed Hashcash deve aceitar Base64 URL-safe")
            server.hashlib.sha256 = original_sha256
            solved_challenges = []
            server.MegaArchive._solve_hashcash = staticmethod(
                lambda challenge: solved_challenges.append(challenge) or "1:proof:nonce"
            )
            response_sequence = iter([
                HttpResponse(402, "", {"X-Hashcash": hashcash_challenge}),
                HttpResponse(200, "[{}]"),
            ])
            request_log.clear()
            server.requests.post = lambda *args, **kwargs: (request_log.append((args, kwargs)) or next(response_sequence))
            server.MegaArchive._login_with_http_adapter("user@example.invalid", "not-a-real-password")
            assert_equal(solved_challenges, [hashcash_challenge], "O desafio X-Hashcash deve ser resolvido uma vez")
            assert_equal(request_log[-1][1]["headers"].get("X-Hashcash"), "1:proof:nonce", "A repetição MEGA deve enviar a prova X-Hashcash")
        finally:
            server.hashlib.sha256 = original_sha256
            server.MegaArchive._solve_hashcash = staticmethod(original_hashcash_solver)
    finally:
        server.Mega = original_mega
        server.requests.post = original_post

    remote: dict[str, bytes] = {}
    upload_destinations: list[object] = []
    class FakeMegaClient:
        def find(self, name):
            if name == server.MEGA_FOLDER:
                return {"name": name, "h": "folder-node"}
            return {"name": name} if name in remote else None
        def destroy(self, node):
            remote.pop(node["name"], None)
        def upload(self, path, folder):
            upload_destinations.append(folder)
            remote[Path(path).name] = Path(path).read_bytes()
            return {"name": Path(path).name}
        def download(self, node, destination):
            output = Path(destination) / node["name"]
            output.write_bytes(remote[node["name"]])
            return str(output)
        def get_files(self):
            return {f"node-{name}": {"a": {"n": name}, "name": name, "h": f"h-{name}"} for name in remote}
    archive = server.MegaArchive()
    archive.available, archive.client, archive.folder = True, FakeMegaClient(), {"name": "IllustriousStudio", "h": "folder-node"}
    remembered = {"prompt": "last signal", "negative_prompt": "lowres", "seed": 77, "steps": 30, "guidance": 7.0, "width": 1024, "height": 768, "strength": 0.65, "mode": "text2img", "loras": []}
    if not archive.save_last_settings(remembered):
        raise AssertionError("Últimas preferências devem ser salvas no arquivo MEGA")
    if not upload_destinations or any(destination != archive.folder for destination in upload_destinations):
        raise AssertionError("Uploads MEGA devem receber um nó individual da pasta, não uma lista de resultados de find()")
    assert_equal(archive.load_last_settings(), remembered, "Últimas preferências devem ser recuperadas do MEGA")
    server.archive = archive
    restored_bootstrap = client.get("/api/bootstrap")
    assert_equal(restored_bootstrap.status_code, 200, "Bootstrap deve responder com preferências persistidas")
    assert_equal(restored_bootstrap.get_json()["last_settings"], remembered, "Bootstrap deve restaurar as últimas preferências")
    assert_equal(restored_bootstrap.get_json()["last_settings_source"], "mega", "Bootstrap deve identificar a origem MEGA das preferências")

    params = server.GenerationParams("signal", "none", 123, 28, 6.5, 1024, 768, .65, "text2img", [])
    job = server.Job("job-contract", "2026-01-01T00:00:00+00:00", "completed", 100, params, filename="job-contract.png")
    image_path = server.OUTPUTS / "job-contract.png"
    image_path.write_bytes(b"fake-png-bytes")
    if not archive.save_job(job, image_path):
        raise AssertionError("Imagem e manifesto do job devem ser enviados ao MEGA")
    assert_equal(job.mega_synced, True, "Job só deve ser marcado como sincronizado após confirmação dos uploads")
    assert_equal(remote["job-contract.png"], b"fake-png-bytes", "Imagem do job deve aparecer no armazenamento remoto")
    manifest = json.loads(remote["job-contract.json"].decode("utf-8"))
    assert_equal(manifest["mega_synced"], True, "Manifesto final deve registrar a sincronização confirmada")
    server.manager.archive = archive
    server.manager.jobs.pop(job.id, None)
    restored_count = server.manager.restore()
    assert_equal(restored_count, 1, "O histórico remoto deve restaurar o job pelo manifesto MEGA")
    restored_job = server.manager.get(job.id)
    assert_equal(restored_job.filename, "job-contract.png", "A restauração deve preservar o filename do manifesto")
    history = client.get("/api/history")
    assert_equal(history.status_code, 200, "Histórico autenticado deve responder")
    item = next(entry for entry in history.get_json()["items"] if entry["id"] == "job-contract")
    assert_equal(item["params"]["guidance"], 6.5, "Histórico deve preservar guidance")
    assert_equal(item["params"]["width"], 1024, "Histórico deve preservar dimensões")
    assert_equal(item["image_url"], "/api/history/job-contract/image", "Histórico deve expor rota de download")
    synced = client.post("/api/history/sync", headers={"X-CSRF-Token": csrf})
    assert_equal(synced.status_code, 200, "Sincronização manual do histórico deve responder")
    assert_equal(synced.get_json()["restored"], 1, "Sincronização manual deve informar jobs restaurados")

    reconnect_calls = {"count": 0}
    def authenticated_remote_client(*_args):
        reconnect_calls["count"] += 1
        return FakeMegaClient()
    original_email = os.environ.get("MEGA_EMAIL")
    original_password = os.environ.get("MEGA_PASSWORD")
    original_login_adapter = server.MegaArchive._login_with_http_adapter
    try:
        os.environ["MEGA_EMAIL"] = "user@example.invalid"
        os.environ["MEGA_PASSWORD"] = "not-a-real-password"
        server.MegaArchive._login_with_http_adapter = staticmethod(authenticated_remote_client)
        reconnect_archive = server.MegaArchive()
        server.archive = reconnect_archive
        server.manager.archive = reconnect_archive
        server.manager.jobs.clear()
        reconnected_bootstrap = client.get("/api/bootstrap")
        assert_equal(reconnected_bootstrap.status_code, 200, "Bootstrap deve conectar e restaurar preferências do MEGA")
        assert_equal(reconnected_bootstrap.get_json()["last_settings"], remembered, "Bootstrap autenticado deve restaurar o último prompt remoto")
        reconnected_sync = client.post("/api/history/sync", headers={"X-CSRF-Token": csrf})
        assert_equal(reconnected_sync.get_json()["restored"], 1, "Sincronização deve restaurar histórico após login HTTP")
        image_path.unlink(missing_ok=True)
        restored_image = client.get("/api/history/job-contract/image")
        assert_equal(restored_image.status_code, 200, "Rota de imagem deve recuperar o PNG remoto após login HTTP")
        assert_equal(restored_image.get_data(), b"fake-png-bytes", "PNG remoto deve ser devolvido sem alterar seus bytes")
        if reconnect_calls["count"] != 1:
            raise AssertionError("Bootstrap e sincronização devem reutilizar a sessão MEGA autenticada")
    finally:
        server.MegaArchive._login_with_http_adapter = staticmethod(original_login_adapter)
        if original_email is None: os.environ.pop("MEGA_EMAIL", None)
        else: os.environ["MEGA_EMAIL"] = original_email
        if original_password is None: os.environ.pop("MEGA_PASSWORD", None)
        else: os.environ["MEGA_PASSWORD"] = original_password

    class HttpBackedRemoteMega(FakeMegaClient):
        def __init__(self):
            self.schema, self.domain, self.sequence_num = "https", "mega.test", 13
            self.timeout, self.sid = 1, None
        def login(self, *_args):
            self._api_request({"a": "us"})
            return self

    original_archive = server.archive
    original_manager_archive = server.manager.archive
    original_jobs = dict(server.manager.jobs)
    fallback_responses = iter([HttpResponse(200, ""), HttpResponse(200, "[{}]")])
    fallback_request_log = []
    original_email = os.environ.get("MEGA_EMAIL")
    original_password = os.environ.get("MEGA_PASSWORD")
    try:
        os.environ["MEGA_EMAIL"] = "user@example.invalid"
        os.environ["MEGA_PASSWORD"] = "not-a-real-password"
        server.Mega = HttpBackedRemoteMega
        server.requests.post = lambda *args, **kwargs: (
            fallback_request_log.append((args, kwargs)) or next(fallback_responses)
        )
        fallback_archive = server.MegaArchive()
        server.archive = fallback_archive
        server.manager.archive = fallback_archive
        server.manager.jobs.clear()
        image_path.unlink(missing_ok=True)
        fallback_bootstrap = client.get("/api/bootstrap")
        assert_equal(
            fallback_bootstrap.get_json()["last_settings"],
            remembered,
            f"Fallback HTTP deve restaurar o último prompt no bootstrap; arquivo={fallback_archive.available}; erro={fallback_archive.error}",
        )
        fallback_sync = client.post("/api/history/sync", headers={"X-CSRF-Token": csrf})
        assert_equal(fallback_sync.get_json()["restored"], 1, "Fallback HTTP deve restaurar o manifesto remoto")
        fallback_image = client.get("/api/history/job-contract/image")
        assert_equal(fallback_image.get_data(), b"fake-png-bytes", "Fallback HTTP deve recuperar o PNG remoto")
        if len(fallback_request_log) != 2 or fallback_request_log[-1][1]["headers"].get("Connection") != "close":
            raise AssertionError("Bootstrap deve usar o fallback HTTP direto após uma resposta vazia")
    finally:
        server.Mega = original_mega
        server.requests.post = original_post
        server.archive = original_archive
        server.manager.archive = original_manager_archive
        server.manager.jobs.clear()
        server.manager.jobs.update(original_jobs)
        if original_email is None: os.environ.pop("MEGA_EMAIL", None)
        else: os.environ["MEGA_EMAIL"] = original_email
        if original_password is None: os.environ.pop("MEGA_PASSWORD", None)
        else: os.environ["MEGA_PASSWORD"] = original_password
    print("CONTRATOS_COLAB_OK")


if __name__ == "__main__":
    main()
