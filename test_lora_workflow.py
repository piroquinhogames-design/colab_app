"""Contract tests for applying LoRA adapters to the ComfyUI workflow."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from comfy_backend import ComfyBackend


class LoraWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="modellab-workflow-")
        self.backend = ComfyBackend(Path(self.temp.name), Path(self.temp.name) / "comfy")
        self.spec = {"defaults": {}, "engine": "comfyui"}
        self.job = SimpleNamespace(
            id="job-test",
            params=SimpleNamespace(
                prompt="a test subject", negative_prompt="", seed=1, steps=24,
                guidance=5.0, width=512, height=512, sampler="euler_a",
                mode="text2img", strength=1.0, upscale=1.0,
            ),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_local_lora_is_chained_into_sampler_model(self):
        workflow = self.backend.build_workflow(
            self.job, self.spec, "checkpoint.safetensors", [("trained.safetensors", 0.8)]
        )
        loader = workflow["101"]
        self.assertEqual(loader["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(loader["inputs"]["lora_name"], "trained.safetensors")
        self.assertEqual(loader["inputs"]["strength_model"], 0.8)
        self.assertEqual(workflow["7"]["inputs"]["model"], ["101", 0])

    def test_multiple_loras_are_applied_in_order(self):
        workflow = self.backend.build_workflow(
            self.job, self.spec, "checkpoint.safetensors",
            [("first.safetensors", 0.6), ("second.safetensors", 1.1)],
        )
        self.assertEqual(workflow["101"]["inputs"]["model"], ["1", 0])
        self.assertEqual(workflow["102"]["inputs"]["model"], ["101", 0])
        self.assertEqual(workflow["7"]["inputs"]["model"], ["102", 0])


if __name__ == "__main__":
    unittest.main()
