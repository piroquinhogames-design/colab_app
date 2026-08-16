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
    if "SERVER_START_TIMEOUT" not in launcher_source or "time.monotonic()" not in launcher_source:
        raise AssertionError("O inicializador deve tolerar inicialização lenta com timeout configurável")
    if '"--upgrade", "--no-deps"' not in launcher_source:
        raise AssertionError("O instalador deve preservar dependências globais do Colab com --no-deps")
    if 'transformers_version != "4.48.3"' not in launcher_source:
        raise AssertionError("O runtime deve confirmar a versão de Transformers compatível com PEFT")
    if "from Crypto.Cipher import AES" not in launcher_source:
        raise AssertionError("O runtime deve validar o módulo Crypto exigido pelo cliente MEGA")
    server_source = (package_root / "server.py").read_text(encoding="utf-8")
    if ".enable_vae_slicing()" in server_source or ".vae.enable_slicing()" not in server_source:
        raise AssertionError("O servidor deve usar a API VAE atual, não o atalho obsoleto do Diffusers")
    if "archive-initializer" not in server_source or "archive_ready" not in server_source:
        raise AssertionError("A conexão MEGA deve ocorrer sem bloquear a abertura do servidor")
    if '"ready": archive_ready.is_set()' not in server_source:
        raise AssertionError("O bootstrap deve expor quando a preparação do MEGA terminou")
    if "_prepare_lora_file" not in server_source or "_is_unsupported_lora_key" not in server_source:
        raise AssertionError("O servidor deve preparar LoRAs com metadados alpha incompatíveis")
    if not server.GeneratorEngine._is_unsupported_lora_key("lora_unet_label_emb_0_0.alpha"):
        raise AssertionError("A chave alpha problemática deve ser identificada")
    app_source = (package_root / "static" / "app.js").read_text(encoding="utf-8")
    if "refreshArchiveState" not in app_source or "refreshHistory({sync: true})" not in app_source:
        raise AssertionError("O frontend deve atualizar e sincronizar o arquivo quando o MEGA conectar depois do bootstrap")
    if server.GeneratorEngine._is_unsupported_lora_key("lora_unet_down_blocks_0.lora_down.weight"):
        raise AssertionError("Pesos normais da LoRA não podem ser descartados")

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
        catalog = client.get("/api/catalog?query=style+adapter&tag=portrait")
    finally:
        server.requests.get = original_get
    assert_equal(catalog.status_code, 200, "Catálogo deve responder")
    assert_equal(called["params"]["baseModels"], "Illustrious", "Catálogo deve filtrar Illustrious")
    assert_equal(called["params"]["query"], "style adapter", "Catálogo deve encaminhar pesquisa por nome")
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

    remote: dict[str, bytes] = {}
    upload_destinations: list[object] = []
    class FakeMegaClient:
        def find(self, name):
            return (name, {"h": name, "a": {"n": name}}) if name in remote else None
        def destroy(self, node):
            if isinstance(node, tuple):
                name = node[1]["a"]["n"]
            elif isinstance(node, str):
                name = node
            else:
                name = node.get("name")
            remote.pop(name, None)
        def upload(self, path, folder):
            upload_destinations.append(folder)
            remote[Path(path).name] = Path(path).read_bytes()
            return {"name": Path(path).name}
        def download(self, node, destination):
            if isinstance(node, tuple):
                name = node[1]["a"]["n"]
            else:
                name = node.get("name") or node.get("a", {}).get("n")
            output = Path(destination) / name
            output.write_bytes(remote[name])
            return str(output)
        def get_files(self):
            return {name: {"h": name, "a": {"n": name}} for name in remote}
    archive = server.MegaArchive()
    archive.available, archive.client, archive.folder = True, FakeMegaClient(), "folder-node"
    remembered = {"prompt": "last signal", "negative_prompt": "lowres", "seed": 77, "steps": 30, "guidance": 7.0, "width": 1024, "height": 768, "strength": 0.65, "mode": "text2img", "loras": []}
    if not archive.save_last_settings(remembered):
        raise AssertionError("Últimas preferências devem ser salvas no arquivo MEGA")
    if not upload_destinations or any(destination != archive.folder for destination in upload_destinations):
        raise AssertionError("Uploads MEGA devem receber um nó individual da pasta, não uma lista de resultados de find()")
    assert_equal(archive.load_last_settings(), remembered, "Últimas preferências devem ser recuperadas do MEGA")
    server.archive = archive
    server.manager.archive = archive
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
    server.manager.jobs[job.id] = job
    history = client.get("/api/history")
    assert_equal(history.status_code, 200, "Histórico autenticado deve responder")
    item = next(entry for entry in history.get_json()["items"] if entry["id"] == "job-contract")
    assert_equal(item["params"]["guidance"], 6.5, "Histórico deve preservar guidance")
    assert_equal(item["params"]["width"], 1024, "Histórico deve preservar dimensões")
    assert_equal(item["image_url"], "/api/history/job-contract/image", "Histórico deve expor rota de download")

    remote_payload = {
        "id": "remote-job-0001",
        "created_at": "2026-01-02T00:00:00+00:00",
        "status": "completed",
        "progress": 100,
        "params": {"prompt": "restored", "negative_prompt": "", "seed": 9, "steps": 20, "guidance": 6.0, "width": 512, "height": 512, "strength": .65, "mode": "text2img", "loras": []},
        "completed_at": "2026-01-02T00:00:01+00:00",
        "mega_synced": True,
    }
    remote["remote-job-0001.json"] = json.dumps(remote_payload).encode("utf-8")
    remote["remote-job-0001.png"] = b"remote-png"
    nested_nodes = archive.list_remote_metadata()
    assert_equal(len([entry for entry in nested_nodes if entry.get("id") == "remote-job-0001"]), 1, "Manifesto aninhado deve ser descoberto")
    server.manager.jobs.pop("remote-job-0001", None)
    server.manager.restore()
    restored = server.manager.get("remote-job-0001")
    if not restored or restored.filename != "remote-job-0001.png":
        raise AssertionError("Job restaurado sem filename deve usar o PNG derivado do id")
    local_remote_image = server.OUTPUTS / "remote-job-0001.png"
    local_remote_image.unlink(missing_ok=True)
    restored_image = client.get("/api/history/remote-job-0001/image")
    assert_equal(restored_image.status_code, 200, "Rota deve baixar PNG remoto quando não há cache local")
    assert_equal(restored_image.data, b"remote-png", "PNG remoto deve ser servido sem alteração")
    pending_params = server.GenerationParams("pending", "none", 456, 24, 6.5, 512, 512, .65, "text2img", [])
    pending_job = server.Job("pending-job", "2026-01-03T00:00:00+00:00", "completed", 100, pending_params, filename="pending-job.png", mega_synced=False)
    pending_image = server.OUTPUTS / "pending-job.png"
    pending_image.write_bytes(b"pending-png")
    server.manager.jobs[pending_job.id] = pending_job

    frontend_source = (package_root / "static" / "app.js").read_text(encoding="utf-8")
    if "refreshHistory({sync: true})" not in frontend_source or "item.filename || item.id" not in frontend_source:
        raise AssertionError("Interface deve sincronizar e renderizar cards restaurados sem filename original")
    if "https://civitai.red/models/" not in frontend_source or "#catalog-query" not in frontend_source:
        raise AssertionError("Interface deve abrir o Civitai no domínio .red e aceitar pesquisa por nome")
    sync = client.post("/api/history/sync", headers={"X-CSRF-Token": csrf})
    assert_equal(sync.status_code, 200, "Sincronização manual deve responder")
    sync_payload = sync.get_json()
    assert_equal(sync_payload["archive"]["available"], True, "Sincronização deve expor estado do arquivo")
    if sync_payload["synced"] < 1 or not pending_job.mega_synced:
        raise AssertionError("Sincronização manual deve reenviar imagens concluídas que ficaram pendentes")
    assert_equal(remote["pending-job.png"], b"pending-png", "Imagem pendente deve ser reenviada ao MEGA")
    assert_equal(sync_payload["last_settings_synced"], True, "Sincronização manual deve reenviar o último prompt")
    print("CONTRATOS_COLAB_OK")


if __name__ == "__main__":
    main()
