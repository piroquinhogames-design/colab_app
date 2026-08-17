from pathlib import Path
import os
import sys
import types

os.environ["STUDIO_ROOT"] = "/tmp/pony-v7-test-studio"
os.environ["HF_HOME"] = "/tmp/pony-v7-test-hf"

fake_mega = types.ModuleType("mega")
fake_mega.Mega = object
sys.modules["mega"] = fake_mega

fake_torch = types.ModuleType("safetensors.torch")
fake_torch.load_file = lambda path, device="cpu": {
    "model.double_layers.0.attn.w2q.weight": "weight",
    "model.positional_encoding": "pos",
}
fake_package = types.ModuleType("safetensors")
fake_package.torch = fake_torch
sys.modules["safetensors"] = fake_package
sys.modules["safetensors.torch"] = fake_torch

from server import GeneratorEngine

normalized = GeneratorEngine._load_auraflow_checkpoint(Path("/tmp/pony_v7_test.safetensors"))
assert "double_layers.0.attn.w2q.weight" in normalized
assert "positional_encoding" in normalized
assert not any(key.startswith("model.") for key in normalized)
print("PONY_NAMESPACE_OK")

# Keep the helper test source-only; the real checkpoint is never loaded here.
Path("/tmp/pony_v7_test.safetensors").unlink(missing_ok=True)
