import subprocess
import unittest
from unittest.mock import patch
import launch_colab


class DependencySetupTests(unittest.TestCase):
    def test_gpu_versions_are_constrained_and_mega_is_isolated(self):
        observed = []
        def run(command, **kwargs):
            observed.append(command)
            if '-c' in command and 'pip' in command:
                from pathlib import Path
                constraints = Path(command[command.index('-c') + 1]).read_text()
                self.assertIn('torch==2.11.0+cu130', constraints)
                self.assertIn('pydantic-core==2.50.0', constraints)
            return subprocess.CompletedProcess(command, 0, '', '')
        with patch.object(launch_colab.importlib.metadata, 'version', return_value='2.11.0+cu130'), patch.object(launch_colab.importlib.metadata, 'distributions', return_value=[]), patch.object(launch_colab.subprocess, 'run', side_effect=run), patch.object(launch_colab, 'validate_pydantic_runtime'), patch.dict(launch_colab.os.environ, {'STUDIO_ROOT': '/tmp/modellab-launcher-tests'}):
            launch_colab.install_requirements()
        self.assertNotIn('--no-deps', observed[0])
        self.assertIn('--no-deps', observed[1])
        self.assertIn('mega.py==1.0.8', observed[1])

    def test_invalid_pydantic_stops_before_server_start(self):
        result = subprocess.CompletedProcess([], 1, '', 'incompatible pydantic-core')
        with patch.object(launch_colab.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'incompatible pydantic-core'):
                launch_colab.validate_pydantic_runtime()

    def test_missing_torch_is_not_installed_implicitly(self):
        with patch.object(launch_colab.importlib.metadata, 'version', side_effect=launch_colab.importlib.metadata.PackageNotFoundError), patch.object(launch_colab.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'PyTorch ausente'):
                launch_colab.install_requirements()
            run.assert_not_called()
