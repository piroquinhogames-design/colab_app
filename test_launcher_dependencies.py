"""Regression checks for the no-deps Pydantic installation failure."""
import subprocess
import unittest
from unittest.mock import patch

import launch_colab


class DependencySetupTests(unittest.TestCase):
    def test_pydantic_group_is_resolved_after_no_deps_installs(self):
        with patch.object(launch_colab.subprocess, 'run') as run, patch.object(launch_colab, 'validate_pydantic_runtime') as validate:
            launch_colab.install_requirements()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertTrue(all('--no-deps' in command for command in commands[:2]))
        self.assertNotIn('--no-deps', commands[2])
        self.assertIn('pydantic~=2.0', commands[2])
        self.assertIn('pydantic-settings~=2.0', commands[2])
        self.assertNotIn('pydantic-core', commands[2])
        validate.assert_called_once_with()

    def test_invalid_runtime_stops_before_server_start(self):
        result = subprocess.CompletedProcess([], 1, '', 'incompatible pydantic-core')
        with patch.object(launch_colab.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'incompatible pydantic-core'):
                launch_colab.validate_pydantic_runtime()


if __name__ == '__main__':
    unittest.main()
