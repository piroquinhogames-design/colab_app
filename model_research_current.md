# Pesquisa atual de modelos

## Nova Anime XL — Civitai

Fonte: https://civitai.com/models/376130/nova-anime-xl

A página consultada em 16/08/2026 mostra a versão **IL v19.0**, arquivo `NovaAnimeILV190.safetensors`, fp16, aproximadamente 6,46 GB. A página classifica o modelo como checkpoint SDXL baseado em Illustrious e informa que a versão IL é distinta da linha Pony. A página exibe recomendação positiva e um conjunto de versões anteriores.

Configurações recomendadas para a variante Illustrious: sampler **Euler a**, **20–30 steps**, CFG **4–6**. Para img2img, a página recomenda denoising strength entre 0,65 e 0,8 na seção geral. O prompt sugerido usa `masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest`, e o negative prompt inclui termos como `bad anatomy`, `bad hands`, `missing fingers`, `extra digits`, `long body`, `deformed`, `mutated`, `disfigured`, `conjoined` e `bad ai-generated`.

A própria página também declara restrições de uso comercial para imagens geradas não editadas. Isso deve ser tratado como uma condição de licença, não como garantia de qualidade anatômica.

## NoobAI-XL V-Pred 1.0 — Civitai

Fonte: https://civitai.com/models/833294/noobai-xl-nai-xl

A página consultada mostra `noobai-xl-vpred-v1.0.safetensors`, BF16, aproximadamente 6,62 GB, base declarada como **NoobAI**, com VRAM recomendada de 7,8 GB. A descrição afirma que a versão foi ajustada para maior precisão anatômica e composição racional, mas ela não é um checkpoint SDXL tradicional simples: é uma variante **V-Pred**.

A própria documentação recomenda Dynamic CFG/CFG Rescale, informa que a série V-Pred não suporta samplers da família Karras e recomenda Euler ou DDIM para maior estabilidade. Portanto, embora seja uma candidata forte para testar anatomia e rostos, ela exigiria uma adaptação técnica no ModelLab Studio: o backend atual não expõe V-Pred/Dynamic CFG e o manifesto atual trata o motor como SDXL/Illustrious convencional.

A página também informa licença NoobAI com restrições adicionais, incluindo proibição de comercialização e exigência de abertura de derivados. Isso precisa ser avaliado antes de qualquer uso público ou comercial.

## Illustrious XL v2.0-STABLE — Hugging Face

Fonte: https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0

A página oficial descreve o checkpoint como uma versão estabilizada do Illustrious XL v2.0, com comportamento geralmente mais estável na geração. O modelo usa licença `creativeml-openrail-m`. A página lista no model tree **104 adapters**, **99 finetunes** e **209 merges** associados ao checkpoint, o que indica um ecossistema amplo de adaptadores, embora essa contagem não prove que toda LoRA funcione igualmente bem em qualquer derivado.

Para o ModelLab, a consequência prática é preferir uma variante SDXL/Illustrious convencional, com estrutura de pesos compatível e prediction type convencional, em vez de começar por V-Pred. Isso reduz o risco de incompatibilidade com LoRAs já treinadas para Illustrious e preserva a implementação atual do carregador Diffusers.

## Evidência de LoRAs específicas para NoobAI

Fonte: https://huggingface.co/Doctor-Shotgun/NoobAI-XL-Character-Lora/blob/main/README.md

A coleção declara que suas LoRAs de personagem foram treinadas em 1024 px especificamente nos checkpoints `Laxhar/noobai-XL-1.0`, `Laxhar/noobai-XL-Vpred-0.75` e `Laxhar/noobai-XL-Vpred-1.0`, e que são destinadas a esses modelos e derivados/merges. A própria página ressalva que o resultado varia conforme o checkpoint base. Isso confirma que existe um ecossistema de LoRAs NoobAI, mas também reforça que a compatibilidade deve ser tratada por variante/prediction type, não apenas pelo nome “NoobAI”.

## Guia oficial de geração do Illustrious XL

Fonte: https://www.illustrious-xl.ai/updates/21

O guia oficial recomenda tratar steps, seed, CFG scale, sampler e demais configurações como um preset coerente e permite salvar esses conjuntos para repetir resultados. O guia também descreve múltiplas linhas de modelo Illustrious, incluindo EPS e VPred, o que reforça que a família não é homogênea: uma LoRA deve ser avaliada contra a variante/prediction type para a qual foi treinada.

## NoobAI-XL Epsilon-pred 1.1 — Civitai

Fonte: https://civitai.com/models/833294/noobai-xl-nai-xl?modelVersionId=1116447

A página da versão Epsilon 1.1 declara base `noobai-XL_v1.0`, com tags nativas e captioning natural language, e diferencia explicitamente Epsilon-pred de V-pred. O material de uso exibido na página recomenda `EulerDiscreteScheduler`, com `prediction_type` convencional para a variante, resolução de exemplo 832×1216, 28 steps e CFG 5. A página também indica um guia específico de LoRAs recomendadas e documentação de treinamento de LoRA para NoobAI.

Essa variante parece tecnicamente mais adequada ao ModelLab do que NoobAI V-Pred quando a prioridade é manter compatibilidade com LoRAs: continua sendo uma família NoobAI/Illustrious, mas não exige o scheduler V-Pred. Ainda assim, a compatibilidade não é universal; LoRAs treinadas especificamente em Epsilon/NoobAI são a escolha mais segura.

## Nova EXAnime AM — API pública do Civitai

Fonte: https://civitai.com/api/v1/models/2856434

A API pública confirma o nome **Nova EXAnime AM**, base model **Anima**, versão `v1.0`, e arquivo SafeTensor disponível em `https://civitai.com/api/download/models/3226184`. O arquivo está marcado como `bf16` nos metadados. O preset do ModelLab deve usar o `model_id` interno `nova-exanime-am`, família `anima`, URL de download `.../3226184` e defaults próprios da família Anima.

## Compatibilidade de engine Anima

Fontes: https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers, https://huggingface.co/circlestone-labs/Anima, https://civitai.com/articles/33274/anima-generation-guide-part-1-general-model-a-bit-of-history

Anima é uma arquitetura própria de aproximadamente 2B parâmetros, derivada de Cosmos-Predict2 e usando Qwen3 0.6B como encoder. O card oficial indica uso de `DiffusionPipeline.from_pretrained` no formato Diffusers, enquanto o checkpoint Civitai é nativo de workflows ComfyUI; portanto não deve passar pelo atual `StableDiffusionXLPipeline.from_single_file`. O card recomenda, para Anima Aesthetic/Base, 30–50 steps, CFG 4–5 e samplers como `euler_a`; LoRAs devem ser treinadas para a família Anima/Base. O backend deve marcar `engine=anima` e bloquear o carregador SDXL quando não houver um engine Anima configurado.
