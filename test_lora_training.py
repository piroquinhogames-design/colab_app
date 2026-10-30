"""Unit tests for dataset ingestion in the Anima LoRA training service."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import lora_training


class DatasetImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="modellab-lora-tests-")
        self.root = Path(self.temp.name)
        self.datasets = self.root / "datasets"
        self.state = self.root / "state.json"
        self.patches = [
            patch.object(lora_training, "DATASETS", self.datasets),
            patch.object(lora_training, "STATE", self.state),
        ]
        for item in self.patches:
            item.start()
        self.trainer = lora_training.AnimaLoRATrainer()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    @staticmethod
    def png_bytes():
        stream = io.BytesIO()
        Image.new("RGB", (12, 12), (30, 90, 140)).save(stream, format="PNG")
        return stream.getvalue()

    def make_zip(self, entries):
        path = self.root / "dataset.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in entries:
                archive.writestr(name, value)
        return path

    def test_shared_caption_is_created_for_images_without_txt(self):
        archive = self.make_zip([("sample.png", self.png_bytes())])
        result = self.trainer.import_dataset(archive, "sample-set", "a blue bird")
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["captions"], 1)
        self.assertEqual((self.datasets / "sample-set" / "sample.txt").read_text(), "a blue bird")

    def test_missing_caption_is_rejected_without_shared_caption(self):
        archive = self.make_zip([("sample.png", self.png_bytes())])
        with self.assertRaisesRegex(ValueError, "sem legenda"):
            self.trainer.import_dataset(archive, "missing-caption")
        self.assertFalse((self.datasets / "missing-caption").exists())

    def test_path_traversal_is_rejected(self):
        archive = self.make_zip([("../escape.png", self.png_bytes())])
        with self.assertRaisesRegex(ValueError, "caminho"):
            self.trainer.import_dataset(archive, "unsafe")
        self.assertFalse((self.root / "escape.png").exists())

    def test_project_names_are_normalized_without_path_components(self):
        self.assertEqual(lora_training._name("../unsafe name"), "unsafe-name")
        with self.assertRaisesRegex(ValueError, "2 a 48 caracteres"):
            lora_training._name("")


if __name__ == "__main__":
    unittest.main()
