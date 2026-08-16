"""Testes de contrato executados pelo Vitest contra rotas reais do Flask.

O módulo MEGA é substituído apenas neste teste para que nenhum serviço externo,
credencial ou download de modelo seja acionado no ambiente de CI.
"""

from __future__ import annotations

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
    launcher_source = (package_root / "launch_colab.py").read_text(encoding="utf-8")
    if "def validate_runtime()" not in launcher_source or "PEFT incompatível" not in launcher_source:
        raise AssertionError("O inicializador deve validar PEFT antes de iniciar o servidor")
    server_source = (package_root / "server.py").read_text(encoding="utf-8")
    if ".enable_vae_slicing()" in server_source or ".vae.enable_slicing()" not in server_source:
        raise AssertionError("O servidor deve usar a API VAE atual, não o atalho obsoleto do Diffusers")

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
                "modelVersions": [{"id": 73, "name": "v1", "baseModel": "Illustrious", "images": [], "stats": {"downloadCount": 8}}],
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
    assert_equal(catalog_json["items"][0]["version_id"], 73, "Versão de LoRA deve ser exposta")
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
    class FakeMegaClient:
        def find(self, name):
            return {"name": name} if name in remote else None
        def destroy(self, node):
            remote.pop(node["name"], None)
        def upload(self, path, _folder):
            remote[Path(path).name] = Path(path).read_bytes()
        def download(self, node, destination):
            output = Path(destination) / node["name"]
            output.write_bytes(remote[node["name"]])
            return str(output)
    archive = server.MegaArchive()
    archive.available, archive.client, archive.folder = True, FakeMegaClient(), "archive"
    remembered = {"prompt": "last signal", "negative_prompt": "lowres", "seed": 77, "steps": 30, "guidance": 7.0, "width": 1024, "height": 768, "strength": 0.65, "mode": "text2img", "loras": []}
    if not archive.save_last_settings(remembered):
        raise AssertionError("Últimas preferências devem ser salvas no arquivo MEGA")
    assert_equal(archive.load_last_settings(), remembered, "Últimas preferências devem ser recuperadas do MEGA")
    server.archive = archive
    restored_bootstrap = client.get("/api/bootstrap")
    assert_equal(restored_bootstrap.status_code, 200, "Bootstrap deve responder com preferências persistidas")
    assert_equal(restored_bootstrap.get_json()["last_settings"], remembered, "Bootstrap deve restaurar as últimas preferências")
    assert_equal(restored_bootstrap.get_json()["last_settings_source"], "mega", "Bootstrap deve identificar a origem MEGA das preferências")

    params = server.GenerationParams("signal", "none", 123, 28, 6.5, 1024, 768, .65, "text2img", [])
    job = server.Job("job-contract", "2026-01-01T00:00:00+00:00", "completed", 100, params, filename="job-contract.png")
    server.manager.jobs[job.id] = job
    history = client.get("/api/history")
    assert_equal(history.status_code, 200, "Histórico autenticado deve responder")
    item = next(entry for entry in history.get_json()["items"] if entry["id"] == "job-contract")
    assert_equal(item["params"]["guidance"], 6.5, "Histórico deve preservar guidance")
    assert_equal(item["params"]["width"], 1024, "Histórico deve preservar dimensões")
    assert_equal(item["image_url"], "/api/history/job-contract/image", "Histórico deve expor rota de download")
    print("CONTRATOS_COLAB_OK")


if __name__ == "__main__":
    main()
