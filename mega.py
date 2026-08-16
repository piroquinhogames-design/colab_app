"""Local compatibility shim for the unmaintained mega.py package.

The application keeps using the public ``from mega import Mega`` API, while
this shim loads the installed mega.py package under a private module name and
adds support for MEGA's HTTP 402 X-Hashcash challenge.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from urllib.parse import urlparse

from mega_hashcash import challenge_from_headers, solve_hashcash


def _load_vendor_package():
    try:
        package_root = Path(distribution("mega.py").locate_file("mega"))
    except PackageNotFoundError as exc:
        raise ImportError("A dependência mega.py não está instalada.") from exc

    init_file = package_root / "__init__.py"
    if not init_file.exists():
        raise ImportError(f"Pacote mega.py inválido: {init_file} não existe.")

    spec = spec_from_file_location(
        "_mega_vendor",
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Não foi possível carregar o pacote mega.py instalado.")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_vendor = _load_vendor_package()
_VendorMega = _vendor.Mega


class Mega(_VendorMega):
    """Compatibility wrapper for mega.py's ``find`` return shape.

    Some mega.py versions return a list from ``find()`` while ``download()``
    expects a single node. The studio already normalizes upload lookups, but
    restore paths passed the raw list to ``download()``, which prevented
    persisted images and last-settings from being recovered after restart.
    """

    def find(self, *args, **kwargs):
        result = super().find(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return result[0] if result else None
        return result


def _install_hashcash_retry() -> None:
    core = sys.modules.get("_mega_vendor.mega")
    if core is None or not hasattr(core, "requests"):
        return

    original_post = core.requests.post
    if getattr(original_post, "_illustrious_hashcash", False):
        return

    def post(*args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        kwargs["headers"] = headers

        for _ in range(4):
            response = original_post(*args, **kwargs)
            parsed = urlparse(str(args[0] if args else kwargs.get("url", "")))
            if response.status_code != 402 or not parsed.hostname or not parsed.hostname.endswith("mega.co.nz"):
                return response

            challenge = challenge_from_headers(response.headers)
            if not challenge:
                return response
            headers["X-Hashcash"] = solve_hashcash(challenge)

        return response

    post._illustrious_hashcash = True
    core.requests.post = post


_install_hashcash_retry()

__all__ = ["Mega"]
