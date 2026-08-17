from pathlib import Path
from types import SimpleNamespace

from comfy_backend import ComfyBackend

root = Path('/tmp/modellab-headless-test')
backend = ComfyBackend(root, root / 'runtime', 8188)
job = SimpleNamespace(
    id='workflow-test',
    params=SimpleNamespace(
        prompt='anime character, blue hair',
        negative_prompt='low quality',
        seed=123,
        steps=24,
        guidance=5.0,
        width=1024,
        height=1024,
        sampler='euler_a',
    ),
)
spec = {'defaults': {}, 'id': 'nova-exanime-am'}
workflow = backend.build_workflow(job, spec, 'novaExanimeAM_v10.safetensors', [('style.safetensors', 0.7), ('detail.safetensors', 0.4)])
assert workflow['1']['class_type'] == 'UNETLoader'
assert workflow['1']['inputs']['unet_name'] == 'novaExanimeAM_v10.safetensors'
assert workflow['2']['inputs']['clip_name'] == 'qwen_3_06b_base.safetensors'
assert workflow['2']['inputs']['type'] == 'qwen_image'
assert workflow['3']['inputs']['vae_name'] == 'qwen_image_vae.safetensors'
assert workflow['7']['inputs']['model'] == ['12', 0]
assert workflow['11']['inputs']['model'] == ['1', 0]
assert workflow['12']['inputs']['model'] == ['11', 0]
assert workflow['9']['class_type'] == 'ModelLabMemoryCleanup'
assert workflow['10']['inputs']['images'] == ['9', 0]
command = backend._command()
for flag in ('--disable-auto-launch', '--force-fp16', '--fp16-intermediates', '--gpu-only', '--highvram', '--cache-none'):
    assert flag in command, flag
assert '--lowvram' not in command
print('HEADLESS_WORKFLOW_OK')
