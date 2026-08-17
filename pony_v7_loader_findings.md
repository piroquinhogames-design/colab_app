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
