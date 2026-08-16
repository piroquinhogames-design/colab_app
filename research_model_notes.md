# Notas de pesquisa — variantes Illustrious

## Conclusão operacional

Não existe fundamento para prometer um modelo “perfeito” para anatomia. A implementação deve manter o checkpoint configurável, tratar modelos como variantes da família SDXL/Illustrious e ajustar defaults por perfil, sem presumir compatibilidade universal de LoRAs.

## Fontes e achados

1. O artigo original do Illustrious descreve o modelo como baseado na arquitetura SDXL, com foco em ilustração/anime, treinamento em resolução alta e melhoria de anatomia, mas também reconhece limitações herdadas de arquiteturas SDXL/CLIP. Fonte: https://arxiv.org/html/2409.19946v1

2. A página do ecossistema Illustrious no Civitai descreve prompting baseado em tags Danbooru, recomenda tags curtas separadas por vírgulas e negative prompts com termos como `bad anatomy`, `bad hands`, `extra digits`; também informa que a família inclui descendentes como NoobAI. Fonte: https://civitai.com/ecosystems/illustrious

3. Nova Anime XL — IL v19.0 é identificado como checkpoint baseado em Illustrious. A página publica CFG recomendado de 4–6 para Illustrious, denoising de 0.65–0.8, e um negative prompt com `bad anatomy`, `bad hands`, `missing fingers`, `extra digits`, `conjoined` e outros termos de artefato. A descrição informa que a versão combina NoobAI EPS v1.1 + Illustrious v2.0-stable + ChenkinNoob v0.5. Fonte: https://civitai.com/models/376130/nova-anime-xl

4. Holy Mix [illustriousXL] é outro checkpoint baseado em Illustrious. O autor recomenda CFG 6, ou 5 para menor contraste, e negative prompt com `bad anatomy`, `bad hands` e `watermark`; afirma que as mãos são muito boas, mas não perfeitas em todos os casos. Fonte: https://civitai.com/models/959490/holy-mix-illustriousxl-high-contrast-anime-checkpoint

## Decisão

Não adicionar automaticamente um segundo checkpoint pesado ao inicializador neste momento. Em vez disso, implementar um manifesto de modelos configuráveis via `MODELS_CONFIG`/`MODEL_URL`/`MODEL_PATH`, um perfil de parâmetros por modelo e um modo de seleção do checkpoint que permita experimentar Nova Anime XL ou outras variantes depois, sem reestruturar o app. O preset inicial pode permanecer WAI-illustrious-SDXL, com uma entrada opcional de Nova Anime XL documentada e desativada por padrão.
