# Notas de pesquisa do Civitai

Fonte oficial: [Enums | Civitai Developer](https://developer.civitai.com/site/reference/enums)

A documentação informa que `GET /api/v1/enums` retorna os valores atuais de `ModelType`, `BaseModel` e outros enums usados pela API. O filtro `types=` de `GET /models` deve usar `LORA` para buscar LoRAs, enquanto `baseModels=` deve usar valores oficiais de `BaseModel`; a lista pode mudar e não deve ser presumida como fixa.

Fonte oficial: [Models | Civitai Developer](https://developer.civitai.com/site/reference/models)

A pesquisa de modelos usa `GET /api/v1/models`, com paginação por `metadata.nextCursor`/`cursor`. A implementação deve manter o cursor recebido pela API e repetir uma consulta textual sem o filtro rígido de base quando a combinação de texto e `baseModels` retornar vazia. Para consultas numéricas, o parâmetro `ids` permite buscar o modelo diretamente.

## Filtros de data e paginação — verificação em 22/08/2026

Fonte oficial: [Models | Civitai Developer](https://developer.civitai.com/site/reference/models). O endpoint `GET /api/v1/models` aceita `limit` de 1 a 100, `cursor`, `query`, `ids`, `tag`, `types`, `baseModels`, `sort` (incluindo `Newest`) e `period` com `AllTime`, `Year`, `Month`, `Week` e `Day`. A documentação não expõe `dateFrom`/`dateTo`; a data de versão vem em `modelVersions[].publishedAt`.

Fonte oficial: [Images | Civitai Developer](https://developer.civitai.com/site/reference/images). O endpoint `GET /api/v1/images` aceita `limit` de 0 a 200, `cursor`, `period` com `AllTime`, `Year`, `Month`, `Week` e `Day`, `sort` com `Newest`, `Oldest`, `Random` e métricas, `type=image`, `withMeta=true`, `baseModels` e `createdAt` em cada item. Como não há filtro remoto documentado de intervalo de datas, a implementação deve buscar páginas por cursor e filtrar `createdAt` localmente quando o usuário informar datas inicial/final.
