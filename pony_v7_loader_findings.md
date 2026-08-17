# Investigação do erro de carregamento do Pony V7 Base

## Causa confirmada

A página do Civitai identifica o Pony V7 Base como um checkpoint baseado na arquitetura AuraFlow, não como um checkpoint SDXL. O backend atual tenta carregá-lo com `StableDiffusionXLPipeline.from_single_file`, o que explica o erro exibido no painel: o arquivo não contém os pesos esperados para `CLIPTextModel`/pipeline SDXL.

## Estrutura oficial

A página oficial do modelo no Hugging Face, https://huggingface.co/purplesmartai/pony-v7-base, classifica o modelo como `Diffusers`, `Safetensors` e `AuraFlowPipeline`. O exemplo de uso público é:

```python
import torch
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "purplesmartai/pony-v7-base",
    dtype=torch.bfloat16,
    device_map="cuda",
)
```

O modelo possui componentes em formato Diffusers, incluindo `text_encoder`, `transformer`, `vae` e scheduler, portanto o arquivo Civitai SafeTensor não deve ser tratado diretamente como um checkpoint SDXL clássico.

## Implicações para o projeto

O carregador deve usar `DiffusionPipeline.from_pretrained("purplesmartai/pony-v7-base")` ou `AuraFlowPipeline.from_pretrained(...)`, com dtype compatível com a GPU. Para preservar o download selecionado pelo usuário, é necessário separar o perfil Pony V7 do arquivo Civitai e do repositório Diffusers oficial, ou converter/baixar o formato Diffusers oficial. A URL oficial do arquivo Civitai é mantida apenas como referência do perfil.

## Fontes

1. https://civitai.red/models/1901521/pony-v7-base?modelVersionId=2152373
2. https://huggingface.co/purplesmartai/pony-v7-base
3. https://huggingface.co/docs/diffusers/main/en/api/loaders/single_file

## Confirmação via API do Hugging Face

A API pública `https://huggingface.co/api/models/purplesmartai/pony-v7-base` informa `config.diffusers._class_name = AuraFlowPipeline` e lista `model_index.json`, `scheduler/scheduler_config.json`, `text_encoder/config.json`, `text_encoder/model.fp16.safetensors`, tokenizer, `transformer/config.json` com três shards SafeTensor e `vae/config.json` com pesos FP16. O arquivo Diffusers oficial é, portanto, um repositório de múltiplos componentes, não um único arquivo de pesos SDXL.

## API do AuraFlowPipeline

A documentação oficial em https://huggingface.co/docs/diffusers/api/pipelines/aura_flow mostra `AuraFlowPipeline.from_pretrained(...)`, com `torch_dtype=torch.float16` e execução direta `pipeline(prompt).images[0]`. A documentação também confirma `AuraFlowTransformer2DModel` e `T5EncoderModel` como componentes do pipeline AuraFlow. O Pony V7 não deve ser instanciado por `StableDiffusionXLPipeline` nem por `StableDiffusionXLImg2ImgPipeline`.

## Compatibilidade do callback — Diffusers 0.39.0

A implementação de `AuraFlowPipeline.__call__` em Diffusers 0.39.0 aceita `callback_on_step_end` e o invoca como `callback(self, step_index, timestep, callback_kwargs)`, retornando um dicionário que pode atualizar `latents` e `prompt_embeds`. O callback atual do backend preserva e devolve `callback_kwargs`, portanto é compatível com esse contrato de quatro argumentos.

Fonte: pacote `diffusers==0.39.0`, arquivo `diffusers/pipelines/aura_flow/pipeline_aura_flow.py`, método `AuraFlowPipeline.__call__`.

## Otimização de download — Hugging Face Hub

A documentação oficial informa que `snapshot_download()` baixa os arquivos de um repositório de forma concorrente e reaproveita o cache local. Ela também permite limitar o conteúdo com `allow_patterns` e `ignore_patterns`, evitando artefatos não necessários ao pipeline. O cache pode ser persistido por `HF_HOME` ou `HF_HUB_CACHE`.

Na versão atual do Hub, `hf-xet` é o caminho recomendado para transferências de alto desempenho; `HF_XET_HIGH_PERFORMANCE=1` substitui a configuração legada `HF_HUB_ENABLE_HF_TRANSFER`, que está depreciada. O projeto passou a usar cache no diretório do estúdio, download seletivo dos arquivos JSON/SafeTensors/tokenizer e até oito workers concorrentes.

Fontes:
1. https://huggingface.co/docs/huggingface_hub/en/guides/download
2. https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
3. https://huggingface.co/docs/huggingface_hub/en/concepts/migration

## Diagnóstico do snapshot lento — 2026-08-17

A API pública de `purplesmartai/pony-v7-base` informa que o repositório Diffusers contém 7B parâmetros e vários arquivos grandes: `transformer/diffusion_pytorch_model-00001-of-00003.safetensors` (~9.99 GB), `00002` (~9.89 GB), `00003` (~7.55 GB), `text_encoder/model.fp16.safetensors` (~2.95 GB), VAE fp16 (~167 MB), além dos arquivos de configuração e tokenizer. O snapshot também contém variantes GGUF, o safetensor single-file de ~13.7 GB, imagens e workflows. O downloader atual filtra GGUF, single-file, imagens e formatos não usados, mas ainda precisa baixar os três shards do transformer e o text encoder para o pipeline Diffusers.

A documentação oficial do Diffusers 0.39 lista os pipelines suportados por `from_single_file()` e não inclui `AuraFlowPipeline` na lista de pipelines single-file suportados. Portanto, não é seguro trocar diretamente para `AuraFlowPipeline.from_single_file()`; isso pode reproduzir o erro de componentes ausentes. O model card oficial recomenda `DiffusionPipeline.from_pretrained("purplesmartai/pony-v7-base", dtype=torch.bfloat16, device_map="cuda")` e identifica a arquitetura como AuraFlow.

Fontes externas:
1. https://huggingface.co/api/models/purplesmartai/pony-v7-base?blobs=true
2. https://huggingface.co/docs/diffusers/en/api/loaders/single_file
3. https://huggingface.co/purplesmartai/pony-v7-base
4. https://civitai.com/articles/6309/towards-pony-diffusion-v7-going-with-the-flow

O workflow oficial do Pony V7 descreve o GGUF como caminho otimizado para ComfyUI com nós `ComfyUI-GGUF` da City96; recomenda Q8_0. A documentação oficial do Diffusers informa que GGUF pode carregar classes de modelo via `from_single_file`, mas **não é suportado diretamente por pipelines**. Portanto, trocar apenas o repositório para `gguf/base-v7-Q4_0.gguf` não é compatível com o backend AuraFlow atual; para usar Q4/Q8 seria necessário migrar o motor de geração para ComfyUI ou implementar um pipeline AuraFlow quantizado específico.

Erro observado no primeiro render: `invalid literal for int() with base 10: 'double_layers'`. A causa foi um namespace adicional no single-file do Civitai (`model.double_layers.*`), enquanto `convert_auraflow_transformer_checkpoint_to_diffusers()` do Diffusers espera `double_layers.*` no nível raiz. O backend agora carrega o safetensor com `safetensors.torch.load_file()` e remove o prefixo `model.diffusion_model.`, `diffusion_model.` ou `model.` quando detecta o marcador `double_layers.` antes de chamar `AuraFlowTransformer2DModel.from_single_file()`. O teste isolado `test_pony_namespace.py` retorna `PONY_NAMESPACE_OK` sem GPU nem download do modelo.
