"""Regression tests for the review findings; no GPU or external credentials."""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['STUDIO_START_WORKERS'] = '0'
os.environ.setdefault('STUDIO_ROOT', tempfile.mkdtemp(prefix='modellab-contracts-'))
os.environ.setdefault('STUDIO_PASSWORD', 'test-password')
os.environ['STUDIO_COOKIE_SECURE'] = '0'
import server as s
from studio_storage import artifact_path
from civitai_resources import model_file
from comfy_backend import ComfyBackend
from studio_security import LoginLimiter, bounded_civitai_image


def version(version_id=123, model_id=456, kind='Checkpoint', base='Anima'):
    return {'id': version_id, 'modelId': model_id, 'baseModel': base,
            'model': {'type': kind, 'name': 'Verified model'},
            'files': [{'id': 789, 'name': 'model.safetensors', 'type': 'Model', 'primary': True,
                       'sizeKB': 12, 'hashes': {'SHA256': 'a' * 64}}]}


def params():
    return s.validate_params({'prompt': 'test image', 'seed': 123}, None)


def job(job_id='test-result', **extra):
    return s.Job(job_id, '2026-10-09T12:00:00+00:00', 'completed', 100, params(), filename=f'{job_id}.png', **extra)


class StudioContracts(unittest.TestCase):
    def setUp(self):
        s.login_limiter = LoginLimiter()
        s.manager.jobs.clear(); s.manager.deleted.clear(); s.manager.idempotency.clear()
        while not s.manager.pending.empty():
            s.manager.pending.get_nowait(); s.manager.pending.task_done()
        self.client = s.app.test_client()
        result = self.client.post('/api/login', json={'password': os.environ['STUDIO_PASSWORD']})
        self.headers = {'X-CSRF-Token': result.json['csrf']}

    def test_cache_refuses_queued_or_loaded_models(self):
        item = s.manager.enqueue(params())
        response = self.client.delete('/api/models/cache', json={'model': item.params.model_id}, headers=self.headers)
        self.assertEqual(response.status_code, 409)
        with patch.object(s.engine, 'loaded_model_id', item.params.model_id):
            s.manager.jobs.clear()
            response = self.client.delete('/api/models/cache', json={'model': item.params.model_id}, headers=self.headers)
            self.assertEqual(response.status_code, 409)

    def test_proxy_rechecks_redirect_origin(self):
        response = SimpleNamespace(status_code=302, headers={'Location': 'http://127.0.0.1/private'})
        context = unittest.mock.MagicMock(); context.__enter__.return_value = response
        with patch('studio_security.requests.get', return_value=context) as get:
            with self.assertRaises(ValueError): bounded_civitai_image('https://image.civitai.com/image.jpeg', 1024)
            self.assertEqual(get.call_count, 1)

    def tearDown(self):
        s.manager.jobs.clear()

    def test_confined_artifacts_reject_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'outputs'; root.mkdir()
            for invalid in ('../escape', '/tmp/escape', 'a/b', 'x.png', ''):
                with self.assertRaises(ValueError): artifact_path(root, invalid, '.png')
            outside = Path(temporary) / 'outside'; outside.write_text('untouched')
            (root / 'valid-id.png').symlink_to(outside)
            with self.assertRaises(ValueError): artifact_path(root, 'valid-id', '.png')
            self.assertEqual(outside.read_text(), 'untouched')

    def test_manifest_cannot_control_filename(self):
        data = job().public(); data['filename'] = '/tmp/outside-review.png'
        with self.assertRaisesRegex(ValueError, 'Caminho'): s.manager._job_from_data(data)
        data['filename'] = 'test-result.png'; data['params']['model'] = 'original-missing-model'
        restored = s.manager._job_from_data(data)
        self.assertEqual(restored.params.model_id, 'original-missing-model')

    def test_store_verifies_metadata_and_preserves_cache(self):
        with patch.object(s, 'version_metadata', return_value=version()):
            response = self.client.post('/api/model-profile', json={'version_id': 123, 'civitai_model_id': 456,
                'family': 'pony', 'defaults': {'steps': 999}, 'name': 'Untrusted'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        spec = s.MODEL_SPECS[response.json['id']]
        self.assertEqual(spec['name'], 'Verified model')
        self.assertEqual(spec['defaults']['steps'], 24)
        self.assertEqual(Path(spec['path']).parent, s.MODELS / 'diffusion_models')
        with patch.dict(os.environ, {'MODELS_CONFIG': ''}):
            self.assertIn(spec['id'], s._load_model_specs())
        with patch.object(s, 'version_metadata', return_value=version(model_id=777)):
            response = self.client.post('/api/model-profile', json={'version_id': 123, 'civitai_model_id': 456}, headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_resource_type_family_format_and_hash_are_validated(self):
        for payload in (version(base='SDXL'), version(kind='LORA')):
            with self.assertRaises(ValueError): model_file(payload, kind='Checkpoint')
        data = version(); data['files'][0]['hashes'] = {}
        with self.assertRaises(ValueError): model_file(data, kind='Checkpoint')

    def test_bad_payloads_return_400(self):
        for payload in ([], {'prompt': 'x', 'loras': {}}, {'prompt': 'x', 'loras': [3]},
                        {'prompt': 'x', 'seed': 2**100}, {'prompt': 'x', 'guidance': 'nan'},
                        {'prompt': 'x', 'loras': [{'version_id': i+1} for i in range(s.MAX_LORAS + 1)]}):
            response = self.client.post('/api/jobs', json=payload, headers=self.headers)
            self.assertEqual(response.status_code, 400, payload)

    def test_idempotent_random_seed_and_queue_limit(self):
        headers = {**self.headers, 'Idempotency-Key': 'same-request'}
        with patch.object(s.archive, 'save_last_settings', return_value=False):
            first = self.client.post('/api/jobs', json={'prompt': 'x', 'seed': -1}, headers=headers)
            second = self.client.post('/api/jobs', json={'prompt': 'x', 'seed': -1}, headers=headers)
            self.assertEqual(first.status_code, 202)
            self.assertEqual(second.json['id'], first.json['id'])
            conflict = self.client.post('/api/jobs', json={'prompt': 'changed'}, headers=headers)
            self.assertEqual(conflict.status_code, 409)
            with patch.dict(os.environ, {'STUDIO_MAX_QUEUE': '1'}):
                full = self.client.post('/api/jobs', json={'prompt': 'next'}, headers=self.headers)
                self.assertEqual(full.status_code, 409)
        self.assertEqual(len(s.manager.jobs), 1)

    def test_queue_cancel_and_batch_validation(self):
        queued = s.manager.enqueue(params())
        cancelled = self.client.post(f'/api/jobs/{queued.id}/cancel', headers=self.headers)
        self.assertEqual(cancelled.json['status'], 'cancelled')
        response = self.client.post('/api/jobs/batch', json={'count': 3, 'settings': {'prompt': 'batch', 'seed': 123}}, headers=self.headers)
        self.assertEqual(response.status_code, 202)
        self.assertEqual([item['params']['seed'] for item in response.json['items']], [123, 124, 125])

    def test_preset_favorite_pagination_export(self):
        preset = self.client.post('/api/presets', json={'name': 'Portrait', 'settings': {'prompt': 'blue'}}, headers=self.headers)
        self.assertEqual(preset.status_code, 201)
        self.assertTrue(any(item['id'] == preset.json['id'] for item in self.client.get('/api/presets').json['items']))
        self.assertEqual(self.client.delete(f"/api/presets/{preset.json['id']}", headers=self.headers).status_code, 200)
        item = job(); s.manager.jobs[item.id] = item
        from PIL import Image, PngImagePlugin
        meta = PngImagePlugin.PngInfo(); meta.add_text('prompt', '{"node": {}}')
        image = Image.new('RGB', (16, 16), 'blue'); image.save(artifact_path(s.OUTPUTS, item.id, '.png'), pnginfo=meta)
        favorite = self.client.post(f'/api/history/{item.id}/favorite', headers=self.headers)
        self.assertTrue(favorite.json['favorite'])
        response = self.client.get(f'/api/history/{item.id}/export')
        self.assertEqual(response.status_code, 200)
        import zipfile
        with zipfile.ZipFile(io.BytesIO(response.data)) as bundle:
            self.assertEqual(set(bundle.namelist()), {'test-result.png', 'manifest.json', 'workflow.json'})
        thumbnail = self.client.get(f'/api/history/{item.id}/thumbnail')
        self.assertEqual(thumbnail.status_code, 200); thumbnail.close()
        for i in range(6): s.manager.jobs[f'item-{i}'] = job(f'item-{i}')
        response = self.client.get('/api/history?limit=2')
        self.assertEqual(len(response.json['items']), 2); self.assertEqual(response.json['next_offset'], 2)

    def test_security_controls(self):
        for _ in range(10): self.client.post('/api/login', json={'password': 'wrong'})
        self.assertEqual(self.client.post('/api/login', json={'password': 'wrong'}).status_code, 429)
        response = self.client.get('/')
        self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy']); response.close()
        self.assertEqual(self.client.get('/api/history', headers={'Host': 'evil.example'}).status_code, 400)
        self.assertEqual(self.client.post('/api/presets', json={'name': 'x'}).status_code, 403)

    def test_incomplete_upload_does_not_mark_manifest_synced(self):
        archive = s.MegaArchive(); archive.available = True
        item = job('incomplete-upload')
        image_path = artifact_path(s.OUTPUTS, item.id, '.png'); image_path.write_bytes(b'test')
        with patch.object(archive, '_upload', side_effect=[None, RuntimeError('JSON upload failed')]):
            self.assertFalse(archive.save_job(item, image_path))
        persisted = json.loads(artifact_path(s.OUTPUTS, item.id, '.json').read_text())
        self.assertFalse(persisted['mega_synced'])

    def test_delete_prevents_future_upload(self):
        item = job('delete-test'); s.manager.jobs[item.id] = item
        response = self.client.delete(f'/api/history/{item.id}', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        with patch.object(s.archive, 'save_job') as save:
            s.manager._persist_job(item, None)
            save.assert_not_called()
        self.assertIn(item.id, s.manager.deleted)

    def test_lora_alpha_is_preserved(self):
        path = Path('/tmp/untouched-lora.safetensors')
        self.assertEqual(s.GeneratorEngine._prepare_lora_file(path), path)
        self.assertFalse(s.GeneratorEngine._is_unsupported_lora_key('lora_unet_0.alpha'))


class ArchiveFolderTests(unittest.TestCase):
    def test_folder_scoping_and_upload_before_delete(self):
        events = []
        class Client:
            def get_files(self):
                return {'old': {'h': 'old', 'p': 'studio', 'a': {'n': 'same.json'}},
                        'other': {'h': 'other', 'p': 'elsewhere', 'a': {'n': 'same.json'}}}
            def upload(self, path, folder): events.append(('upload', folder)); return 'new'
            def destroy(self, handle): events.append(('destroy', handle))
        archive = s.MegaArchive(); archive.available = True; archive.folder = 'studio'; archive.client = Client()
        self.assertEqual(archive._find_file('same.json')[0], 'old')
        archive._upload(Path('/tmp/same.json'))
        self.assertEqual(events, [('upload', 'studio'), ('destroy', 'old')])
        self.assertTrue(archive.delete_job('same'))
        self.assertNotIn(('destroy', 'other'), events)

    def test_failed_replacement_preserves_previous(self):
        archive = s.MegaArchive(); archive.available = True; archive.folder = 'studio'
        archive.client = SimpleNamespace(get_files=lambda: {'old': {'h': 'old', 'p': 'studio', 'a': {'n': 'same.json'}}}, upload=lambda *_: None)
        with patch.object(archive.client, 'upload', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError): archive._upload(Path('/tmp/same.json'))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        self.backend = ComfyBackend(Path(__file__).parent, self.root, 8188)
    def tearDown(self): self.temporary.cleanup()

    def test_img2img_workflow_has_no_lora_node_collision(self):
        item = job(); item.params.mode = 'img2img'; item.params.strength = .55; item.comfy_source = 'source.png'
        workflow = self.backend.build_workflow(item, {}, 'model.safetensors', [('one.safetensors', .8), ('two.safetensors', .6)])
        self.assertEqual(workflow['6']['class_type'], 'VAEEncode')
        self.assertEqual(workflow['7']['inputs']['denoise'], .55)
        self.assertEqual(workflow['7']['inputs']['model'], ['102', 0])
        self.assertEqual(workflow['12']['class_type'], 'LoadImage')
        self.assertEqual(workflow['8']['class_type'], 'VAEDecodeTiled')

    def test_upscale_preserves_latent_size_and_changes_output_size(self):
        item = job(); item.params.upscale = 2
        workflow = self.backend.build_workflow(item, {}, 'model.safetensors', [])
        self.assertEqual(workflow['6']['inputs']['width'], item.params.width)
        self.assertEqual(workflow['11']['inputs']['width'], 2 * item.params.width)
        self.assertEqual(workflow['9']['inputs']['image'], ['11', 0])

    def test_shared_components_are_pinned_and_verified(self):
        with patch.object(self.backend, 'ensure_file') as download:
            self.backend.ensure_anima_dependencies()
        for call in download.call_args_list:
            self.assertNotIn('/resolve/main/', call.args[0])
            self.assertEqual(len(call.kwargs['sha256']), 64)
            self.assertGreater(call.kwargs['expected_bytes'], 100 * 1024 * 1024)

    def test_checkpoint_registration_does_not_duplicate_bytes(self):
        source = self.root / 'custom.safetensors'; source.write_bytes(b'model')
        name = self.backend.register_checkpoint(source)
        self.assertTrue((self.backend.model_dir / name).is_symlink())
        self.assertEqual((self.backend.model_dir / name).read_bytes(), b'model')

    def test_interrupt_only_our_running_prompt(self):
        with patch.object(self.backend.session, 'get', return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'queue_running': [[0, 'somebody-else']]})), patch.object(self.backend.session, 'post') as post:
            self.backend._cancel_prompt('ours'); post.assert_not_called()
        with patch.object(self.backend.session, 'get', return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'queue_running': [[0, 'ours']]})), patch.object(self.backend.session, 'post') as post:
            self.backend._cancel_prompt('ours'); self.assertTrue(post.call_args.args[0].endswith('/interrupt'))

    def test_gpu_readings_come_from_comfy(self):
        with patch.object(self.backend.session, 'get', return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'devices': [{'vram_total': 16 * 1024**3, 'vram_free': 10 * 1024**3}]})):
            self.assertEqual(self.backend.gpu_memory()['used_gb'], 6)
