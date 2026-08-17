# Nova EXAnime AM — verificação técnica

Data da verificação: 2026-08-17.

Fonte principal consultada: https://civitai.com/api/v1/models/2856434
Página fornecida pelo usuário: https://civitai.red/models/2856434/nova-exanime-am

A API pública do Civitai identifica o modelo como **Nova EXAnime AM**, modelo `2856434`, versão `v1.0`, publicada em 13 de agosto de 2026, com descrição `Initial Release` e base declarada `Anima B1 + A11 base`.

O arquivo principal é `novaExanimeAM_v10.safetensors`, formato SafeTensor, precisão declarada **bf16**, aproximadamente **4.084.437 KB** (cerca de 3,9 GiB), com download no version ID `3226184` e file ID `3108312`. URL direta de download: `https://civitai.com/api/download/models/3226184?fileId=3108312`.

A página web fornecida exigiu login e não expôs os detalhes no HTML extraído; os metadados acima vieram da API pública. A integração não deve presumir que o arquivo seja um checkpoint SDXL/Diffusers: a base declarada é **Anima**, e o formato/loader correto precisa ser compatível com o backend do ComfyUI usado pelo projeto.

Requisito do usuário: tornar Nova EXAnime AM o padrão, executar somente o backend/API do ComfyUI, não abrir o frontend, não retirar carga da GPU, preservar o modelo carregado e otimizar ao máximo sem descarregar o modelo.

Fonte adicional: https://docs.comfy.org/tutorials/image/anima/anima

A documentação oficial do ComfyUI descreve Anima como um modelo de 2B parâmetros, nativo do ComfyUI, carregado com `UNETLoader`, `CLIPLoader` e `VAELoader`. O encoder de texto compartilhado é Qwen-3 0.6B e o VAE é `qwen_image_vae.safetensors`. O layout oficial usa `models/diffusion_models/`, `models/text_encoders/` e `models/vae/`. O workflow é text-to-image e usa um Subgraph; isso confirma que o caminho correto para Nova EXAnime AM é o backend de execução do ComfyUI, não `StableDiffusionXLPipeline` do Diffusers.

A documentação também afirma suporte nativo via workflow e permite baixar o JSON do workflow, o que é adequado para enviar prompts pela API `/prompt` sem abrir o frontend.

Fonte adicional: https://huggingface.co/circlestone-labs/Anima

O model card oficial descreve Anima como um modelo de 2 bilhões de parâmetros e lista as variantes Base, Aesthetic e Turbo. A variante Base é a versão pré-treinada para maior flexibilidade e é a versão indicada para treinar LoRAs; Aesthetic é uma versão fine-tuned; Turbo é uma versão destilada para geração rápida, recomendada em CFG 1 e 8–12 steps. O repositório é marcado como Diffusion Single File e ComfyUI, com licença não comercial do CircleStone Labs.

Isso reforça que o Nova EXAnime AM, baseado em Anima B1 + A11, não deve ser tratado como SDXL. O backend precisa preservar o caminho ComfyUI/Anima e não aplicar automaticamente `StableDiffusionXLPipeline`, `torch.float16` ou VAE de SDXL.

Fonte adicional: https://github.com/Comfy-Org/ComfyUI/issues/12230

A issue #12230 do ComfyUI registra que Anima aparentava exigir BF16 e pergunta sobre suporte a computação FP16. A discussão foi fechada como concluída em 8 de fevereiro de 2026 após o autor apontar um patch publicado no Civitai (modelo 2356447, versão 2652286). Isso indica que o suporte FP16 usado pelo usuário provavelmente depende de um patch/loader específico, e não de um simples `torch_dtype=torch.float16` no Diffusers. A implementação deve tornar esse patch/configuração explícito e reproduzível, sem forçar FP16 em componentes incompatíveis.

Checagem adicional: a API do modelo Civitai `2356447` retorna `RDBT | Anima`, um checkpoint/finetune baseado em Anima com versões múltiplas; a rota direta `/api/v1/model-versions/2652286` retornou `Model not found`. Portanto, a issue do ComfyUI aponta um recurso Civitai que não está mais resolvendo por esse endpoint, e ele não deve ser baixado ou tratado como requisito do Nova EXAnime AM sem uma fonte atualizada. A implementação deve depender do comportamento do ComfyUI/loader disponível, não desse ID obsoleto.

ComfyUI inspecionado localmente a partir de https://github.com/comfyanonymous/ComfyUI (clone shallow em 2026-08-17; o comando exibiu o commit atual). O código contém suporte interno a `lowvram`/patches parciais, mas a meta do usuário é não retirar carga da GPU, portanto essa opção não deve ser habilitada no perfil padrão. A implementação deverá consultar/usar flags reais do launcher e manter o carregamento residente quando a VRAM permitir; não usar `/free` nem descarregar o modelo entre jobs.

## Diagnóstico do erro `trampoline` (17/08/2026)

O traceback do Colab mostra que o ComfyUI falha ao importar `comfy.samplers`, que importa `k_diffusion.sampling`, antes de abrir a API. A falha final é `ModuleNotFoundError: No module named 'trampoline'` dentro do ambiente Python.

A documentação pública do ComfyUI lista `torchsde` como dependência direta do backend. A página oficial do PyPI para `torchsde` identifica o pacote como o solver SDE usado com PyTorch; a listagem de dependências do PyPI também indica `trampoline` como dependência. O PyPI publica `trampoline` 0.1.2 como pacote Python puro. Portanto, a correção deve instalar explicitamente `trampoline` (ou instalar `torchsde` com dependências) usando o mesmo interpretador do Colab, sem reinstalar torch, torchvision ou torchaudio.

Fontes consultadas: https://pypi.org/project/torchsde/ ; https://pypi.org/project/trampoline/ ; https://github.com/comfyanonymous/ComfyUI/blob/master/requirements.txt
