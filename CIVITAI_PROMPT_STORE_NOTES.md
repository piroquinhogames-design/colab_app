# Base da Loja de Prompts

Fonte oficial consultada em 16/08/2026: [Civitai Developer — Images](https://developer.civitai.com/site/reference/images)

O endpoint público é `GET /api/v1/images`. Para a galeria de imagens com metadados, a consulta usa `limit`, `cursor`, `sort` (`Most Reactions`, `Newest`, `Random`), `nsfw`, `type=image` e `withMeta=true`. A resposta inclui `items`, `metadata.nextCursor`, URL da imagem, dimensões, usuário, nível NSFW, estatísticas e `meta`.

Os campos relevantes para remix são `meta.prompt`, `meta.negativePrompt`, `meta.seed`, `meta.steps`, `meta.cfgScale` e `meta.civitaiResources`. Cada recurso pode informar `type=lora`, `modelVersionId` e `weight`, permitindo reconstruir o rack de LoRAs quando os metadados foram publicados pelo autor. `meta` pode estar ausente ou ser livre, portanto a implementação trata prompt como opcional e mantém o card desabilitado quando não há dados suficientes.

A implementação filtra localmente termos combináveis em prompt, negative prompt, autor e tags; o filtro +18 continua condicionado ao token do servidor, como no catálogo de LoRAs existente.
