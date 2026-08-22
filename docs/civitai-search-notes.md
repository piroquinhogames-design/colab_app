# Notas de pesquisa do Civitai

Fonte oficial: [Enums | Civitai Developer](https://developer.civitai.com/site/reference/enums)

A documentação informa que `GET /api/v1/enums` retorna os valores atuais de `ModelType`, `BaseModel` e outros enums usados pela API. O filtro `types=` de `GET /models` deve usar `LORA` para buscar LoRAs, enquanto `baseModels=` deve usar valores oficiais de `BaseModel`; a lista pode mudar e não deve ser presumida como fixa.

Fonte oficial: [Models | Civitai Developer](https://developer.civitai.com/site/reference/models)

A pesquisa de modelos usa `GET /api/v1/models`, com paginação por `metadata.nextCursor`/`cursor`. A implementação deve manter o cursor recebido pela API e repetir uma consulta textual sem o filtro rígido de base quando a combinação de texto e `baseModels` retornar vazia. Para consultas numéricas, o parâmetro `ids` permite buscar o modelo diretamente.
