"""Consentimento, ciclo de vida dos túneis e contrato de workflow sem GPU."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

os.environ['STUDIO_START_WORKERS'] = '0'
os.environ.setdefault('STUDIO_ROOT', tempfile.mkdtemp(prefix='modellab-silly-tests-'))
os.environ.setdefault('STUDIO_PASSWORD', 'test-password')
os.environ['STUDIO_COOKIE_SECURE'] = '0'

import launch_colab as launcher
import server as studio
from comfy_backend import ComfyBackend
from studio_storage import atomic_json


class ExposureConsentTests(unittest.TestCase):
    def test_only_explicit_yes_enables_exposure(self):
        for answer in ('sim', ' SIM ', 's', 'yes', 'y', '', 'não', 'nao', 'n', 'no'):
            with self.subTest(answer=answer), patch('builtins.input', return_value=answer), redirect_stdout(io.StringIO()) as output:
                accepted = launcher.ask_comfy_exposure()
                self.assertEqual(accepted, answer.strip().casefold() in {'sim', 's', 'yes', 'y'})
                self.assertIn('SEM autenticação', output.getvalue())
                self.assertIn('NÃO protege esta API', output.getvalue())

    def test_invalid_answer_is_not_consent(self):
        with patch('builtins.input', side_effect=['talvez', 'não']) as ask, redirect_stdout(io.StringIO()):
            self.assertFalse(launcher.ask_comfy_exposure())
            self.assertEqual(ask.call_count, 2)

    def test_missing_input_keeps_api_closed(self):
        for error in (EOFError(), OSError('stdin unavailable')):
            with self.subTest(error=type(error)), patch('builtins.input', side_effect=error), redirect_stdout(io.StringIO()):
                self.assertFalse(launcher.ask_comfy_exposure())

    def run_launcher(self, answer, states, interrupt=False):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stale_url = root / 'current_comfy_tunnel_url.txt'
            stale_url.write_text('https://previous-session.trycloudflare.com')
            stack.enter_context(patch.dict(os.environ, {'STUDIO_PREPARE_COMFY': '1', 'COMFY_PORT': '9191'}))
            stack.enter_context(patch.object(launcher, 'APP_DIR', root))
            stack.enter_context(patch('builtins.input', return_value=answer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for name in ('ask_secret', 'install_requirements', 'ensure_comfyui', 'validate_runtime', 'wait_for_server'):
                stack.enter_context(patch.object(launcher, name))
            stack.enter_context(patch.object(launcher, 'ensure_cloudflared', return_value='/fake/cloudflared'))
            stack.enter_context(patch.object(launcher.threading, 'Thread'))
            stack.enter_context(patch.object(launcher.os, 'killpg'))
            stack.enter_context(patch.object(launcher.time, 'sleep', side_effect=KeyboardInterrupt if interrupt else None))
            process = MagicMock(pid=12345)
            process.poll.side_effect = [None] * len(states) + [0]
            popen = stack.enter_context(patch.object(launcher.subprocess, 'Popen', return_value=process))
            tunnels = stack.enter_context(patch.object(launcher, 'QuickTunnel'))
            panel, comfy = MagicMock(), MagicMock()
            tunnels.side_effect = [panel, comfy]
            responses = []
            for state in states:
                response = MagicMock()
                response.json.return_value = {'public_comfy': {'status': state}}
                responses.append(response)
            get = stack.enter_context(patch.object(launcher.requests, 'get', side_effect=responses))
            launcher.main()
            self.assertFalse(stale_url.exists())
            process.wait.assert_called_once()
            panel.close.assert_called_once()
            return popen.call_args.kwargs['env'], tunnels, panel, comfy, get

    def test_no_overrides_previous_session_and_creates_only_panel_tunnel(self):
        env, tunnels, panel, comfy, get = self.run_launcher('não', ['disabled'])
        self.assertEqual(env['STUDIO_PREPARE_COMFY'], '0')
        self.assertEqual(tunnels.call_count, 1)
        panel.service.assert_called_once()
        comfy.service.assert_not_called()
        get.assert_not_called()

    def test_yes_waits_for_backend_ready_and_uses_configured_port(self):
        env, tunnels, panel, comfy, get = self.run_launcher('sim', ['preparing', 'ready'])
        self.assertEqual(env['STUDIO_PREPARE_COMFY'], '1')
        self.assertEqual(tunnels.call_args_list[1].args[1], 9191)
        self.assertEqual(panel.service.call_count, 2)
        comfy.service.assert_called_once()
        comfy.close.assert_called_once()
        self.assertEqual(get.call_count, 2)

    def test_backend_failure_preserves_panel_without_opening_api(self):
        _, _, panel, comfy, get = self.run_launcher('sim', ['failed', 'failed'])
        self.assertEqual(panel.service.call_count, 2)
        comfy.service.assert_not_called()
        comfy.close.assert_called_once()
        get.assert_called_once()

    def test_interrupt_closes_both_tunnels(self):
        _, _, panel, comfy, _ = self.run_launcher('sim', ['ready'], interrupt=True)
        panel.close.assert_called_once()
        comfy.close.assert_called_once()


class TunnelProcessTests(unittest.TestCase):
    def test_real_process_url_restart_and_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            binary = root / 'fake-cloudflared'
            binary.write_text(
                f'#!{sys.executable}\nimport os, signal\n'
                'print(f"https://test-{os.getpid()}.trycloudflare.com", flush=True)\n'
                'signal.pause()\n'
            )
            binary.chmod(0o755)
            urls = []
            ready = threading.Event()
            def on_url(url):
                urls.append(url)
                ready.set()
            path = root / 'url.txt'
            tunnel = launcher.QuickTunnel(str(binary), 8188, 'test-tunnel', path, on_url)
            try:
                tunnel.service()
                self.assertTrue(ready.wait(5), 'Túnel de teste não publicou URL')
                old_process = tunnel.process
                first_url = path.read_text().strip()
                old_process.kill()
                old_process.wait(timeout=5)
                tunnel.service()
                self.assertFalse(path.exists())
                self.assertEqual(tunnel.url, '')
                ready.clear()
                with patch.object(launcher.time, 'monotonic', return_value=tunnel.retry_at):
                    tunnel.service()
                self.assertTrue(ready.wait(5), 'Túnel não reiniciou')
                self.assertNotEqual(path.read_text().strip(), first_url)
                tunnel._capture_url(old_process, 'https://stale.trycloudflare.com')
                self.assertEqual(path.read_text().strip(), urls[-1])
                active_process = tunnel.process
            finally:
                tunnel.close()
            self.assertIsNotNone(active_process.poll())
            self.assertFalse(path.exists())
            tunnel._capture_url(active_process, 'https://after-close.trycloudflare.com')
            self.assertFalse(path.exists())
            self.assertEqual(len(urls), 2)

    def test_one_tunnel_backoff_does_not_block_another(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), patch.object(launcher.threading, 'Thread'):
            root = Path(temporary)
            failed = launcher.QuickTunnel('/missing', 7860, 'panel', root / 'panel.txt', lambda _url: None)
            healthy = launcher.QuickTunnel('/working', 8188, 'comfy', root / 'comfy.txt', lambda _url: None)
            process = MagicMock()
            process.poll.return_value = None
            with patch.object(launcher.subprocess, 'Popen', side_effect=[OSError('missing'), process]) as popen:
                failed.service()
                failed.service()
                healthy.service()
                self.assertEqual(popen.call_count, 2)
                self.assertIs(healthy.process, process)
            failed.close()
            healthy.close()


class SillyWorkflowTests(unittest.TestCase):
    def test_sillytavern_substitutions_keep_types_prompts_and_anima_graph(self):
        backend = ComfyBackend(Path.cwd(), Path('/tmp/modellab-silly-workflow'))
        backend.memory_node_available = False
        workflow = backend.build_sillytavern_workflow({'defaults': {'positive_prefix': 'quality'}}, 'custom_anima.safetensors')
        self.assertEqual(workflow['1']['inputs']['unet_name'], 'custom_anima.safetensors')
        self.assertEqual(workflow['2']['inputs']['type'], 'qwen_image')
        self.assertEqual(workflow['6']['class_type'], 'EmptySD3LatentImage')
        values = {'prompt': 'a "blue" cat\nby a window', 'negative_prompt': 'low quality', 'seed': 123,
                  'steps': 24, 'scale': 5.0, 'width': 768, 'height': 768,
                  'sampler': 'euler_ancestral', 'scheduler': 'normal'}
        text = json.dumps(workflow)
        for name, value in values.items():
            text = text.replace(json.dumps(f'%{name}%'), json.dumps(value))
        self.assertNotIn('%', text)
        submitted = json.loads(text)
        self.assertEqual(submitted['4']['inputs']['text'], values['prompt'])
        self.assertIsInstance(submitted['7']['inputs']['seed'], int)
        self.assertIsInstance(submitted['6']['inputs']['width'], int)
        self.assertEqual(submitted['8']['inputs']['temporal_overlap'], 8)
        for node in submitted.values():
            for value in node['inputs'].values():
                if isinstance(value, list):
                    self.assertIn(value[0], submitted)

    def test_checked_in_template_is_api_json_with_save_image(self):
        workflow = json.loads((Path(__file__).parent / 'workflows/WAI_ANIMA_SillyTavern.json').read_text())
        self.assertNotIn('nodes', workflow)
        self.assertEqual(workflow['10']['class_type'], 'SaveImage')
        self.assertEqual(workflow['4']['inputs']['text'], '%prompt%')
        self.assertEqual(workflow['3']['inputs']['vae_name'], 'qwen_image_vae.safetensors')


class BackendPreparationTests(unittest.TestCase):
    def test_ready_means_model_registered_and_downloadable_workflow_written(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'STUDIO_PREPARE_COMFY': '1'}), \
             patch.object(studio, 'PUBLIC_COMFY_WORKFLOW', Path(temporary) / 'workflow.json'), \
             patch.object(studio, 'public_comfy_status', 'pending'), \
             patch.object(studio.engine, '_load_pipeline') as load, \
             patch.object(studio.engine.comfy, 'register_checkpoint', return_value='configured.safetensors'), \
             redirect_stdout(io.StringIO()):
            studio.initialize_public_comfy()
            load.assert_called_once()
            self.assertEqual(studio.public_comfy_status, 'ready')
            client = studio.app.test_client()
            self.assertEqual(client.get('/api/sillytavern/workflow').status_code, 401)
            with client.session_transaction() as session:
                session['authenticated'] = True
            response = client.get('/api/sillytavern/workflow')
            self.assertEqual(response.status_code, 200)
            self.assertIn('ModelLab_SillyTavern.json', response.headers['Content-Disposition'])
            workflow = json.loads(response.data)
            self.assertEqual(workflow['1']['inputs']['unet_name'], 'configured.safetensors')
            self.assertEqual(client.get('/api/health').json['public_comfy']['status'], 'ready')
            response.close()
            with patch.dict(os.environ, {'STUDIO_PREPARE_COMFY': '0'}):
                self.assertEqual(client.get('/api/sillytavern/workflow').status_code, 404)

    def test_failed_preparation_removes_stale_export_without_breaking_health(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'CIVITAI_TOKEN': 'secret-token'}), \
             patch.object(studio, 'PUBLIC_COMFY_WORKFLOW', Path(temporary) / 'workflow.json'), \
             patch.object(studio, 'public_comfy_status', 'pending'), \
             patch.object(studio.engine, '_load_pipeline', side_effect=RuntimeError('download failed: secret-token')), \
             redirect_stdout(io.StringIO()) as output:
            atomic_json(studio.PUBLIC_COMFY_WORKFLOW, {'stale': True})
            studio.initialize_public_comfy()
            self.assertEqual(studio.public_comfy_status, 'failed')
            self.assertFalse(studio.PUBLIC_COMFY_WORKFLOW.exists())
            self.assertNotIn('secret-token', output.getvalue())
            response = studio.app.test_client().get('/api/health')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['status'], 'ok')

