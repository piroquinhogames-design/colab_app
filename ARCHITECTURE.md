# Arquitetura do ModelLab Studio

O pacote é executado integralmente em uma sessão Google Colab com GPU T4. O processo Flask serve a interface, recebe solicitações autenticadas, gerencia uma única fila de geração e publica a porta local por meio de um Quick Tunnel. Nenhuma chave de Civitai, credencial MEGA ou senha do estúdio é enviada ao navegador.

## Interface e seleção de família

O formulário principal mostra somente o modelo ativo. Checkpoint, sampler e loja de modelos ficam no diálogo **Configurações**. O bootstrap envia os perfis públicos com família, base, engine, disponibilidade, cache e defaults. A interface aplica o preset do perfil e atualiza automaticamente os rótulos e as lojas de LoRAs e prompts.

O perfil inicial é `prefect-pony-xl-v6`, com base `Pony`, modelo Civitai `439889`, versão `2114187` e engine `sdxl`. O runtime baixa o arquivo SafeTensor single-file do Civitai com retomada HTTP e cria as pipelines `StableDiffusionXLPipeline` e `StableDiffusionXLImg2ImgPipeline` diretamente a partir do checkpoint fp16. O fluxo oferece TXT→IMG e IMG→IMG; o cache Hugging Face permanece disponível para perfis que precisem de componentes auxiliares, mas não é exigido pelo perfil padrão.

O perfil opcional `pony-v7-base` usa a família `pony-v7` e engine `auraflow`. O `GeneratorEngine` carrega `purplesmartai/pony-v7-base` com `DiffusionPipeline.from_pretrained`, usando `HF_HUB_CACHE`, `torch.float16` e offload de CPU quando nenhum `device_map` explícito foi configurado. A integração inicial expõe apenas TXT→IMG em 768–1024 px com `flow_euler`; IMG→IMG e LoRAs são recusados pelo contrato até que sejam validados para AuraFlow. A separação por engine mantém o pipeline SDXL do V6 intacto.

## Catálogo Civitai

A rota `GET /api/model-catalog` consulta checkpoints Civitai com query, tag, família, ordenação, paginação e filtro adulto. A base Civitai é traduzida por `civitai_base_for_family`: Illustrious/NoobAI, Pony, SDXL, Flux e SD 3. Cada versão é convertida em um perfil público com família, engine, defaults e arquivo principal.

A rota `POST /api/model-profile` recebe somente o identificador numérico do modelo/versão e cria um perfil interno validado pelo servidor. A seleção é mantida em `MODEL_SPECS` e serializada em `model_profiles.json` dentro de `STUDIO_ROOT`, permitindo que o catálogo selecionado sobreviva ao reinício do processo. A URL de download é construída no servidor a partir do version ID, não aceita uma URL arbitrária enviada pelo navegador.

## Lojas dependentes da família

A rota `GET /api/catalog` inicia em `family=pony` e converte a família para o parâmetro `baseModels` da API Civitai. Quando o modelo muda, o frontend envia a nova família, atualiza a loja de LoRAs e limita os resultados às versões cujo `baseModel` seja compatível.

A rota `GET /api/prompt-store` também recebe a família ativa. O servidor consulta imagens com metadados, extrai recursos Civitai, prompts, dimensões e LoRAs, filtra recursos de outra família quando há informação suficiente e embaralha os itens no modo `Random`. A busca textual combina prompt, negative prompt, autor, tags e família; os filtros de chips são combinados como interseção.

## Engines e perfis

O `GeneratorEngine` troca pipelines somente quando o `model_id` muda. Para `sdxl`, descarrega o pipeline anterior, libera a memória CUDA, retoma o checkpoint Civitai se houver `.part` e cria as pipelines text-to-image e image-to-image; jobs seguintes não repetem o download. Para `auraflow`, o Diffusers gerencia o snapshot por repositório/cache e o engine não cria uma pipeline IMG→IMG. Flux e SD 3 podem ser catalogados, mas ficam explicitamente bloqueados até receberem engines próprios.

| Família | Engine | Loja de LoRAs | Estado padrão |
|---|---|---|---|
| `sdxl-illustrious` | `sdxl` | Illustrious | Geração disponível com pipeline SDXL |
| `pony` | `sdxl` | Pony | Geração disponível com SDXL em TXT→IMG e IMG→IMG |
| `pony-v7` | `auraflow` | Desativada inicialmente | TXT→IMG disponível; IMG→IMG e LoRAs bloqueadas |
| `sdxl` | `sdxl` | SDXL 1.0 | Geração disponível com pipeline SDXL |
| `flux` | engine separado | Flux | Catalogável; geração bloqueada |
| `sd3` | engine separado | SD 3 | Catalogável; geração bloqueada |

## Persistência

Cada resultado gera dois arquivos: `outputs/<job_id>.png` e `outputs/<job_id>.json`. O JSON inclui identificador, data UTC, prompts, seed efetivamente usada, modelo, sampler, parâmetros, LoRAs, estado final e referências de armazenamento. Após uma geração bem-sucedida, os dois arquivos e o manifesto `last_settings.json` são enviados ao diretório configurado da conta MEGA. Ao inicializar uma nova sessão, o servidor recupera os JSONs para recompor a galeria; a imagem é baixada sob demanda apenas se não existir no cache local.

A rota `DELETE /api/history/<job_id>` exige autenticação e CSRF. Jobs em execução não podem ser removidos. Para um job sincronizado, o servidor primeiro tenta excluir `<job_id>.png` e `<job_id>.json` no MEGA; somente após essa confirmação remove o cache local e o registro em memória. Se o arquivo remoto estiver indisponível, a operação é recusada para evitar exclusão incompleta.

## Recursos e proteção

| Recurso | Rota ou armazenamento | Proteção |
|---|---|---|
| Interface e API | Processo Flask na porta 7860 | Login por senha e sessão HTTP-only |
| Geração | Fila de um trabalhador no processo Colab | Somente usuário autenticado; uma tarefa ativa por vez |
| Perfis de checkpoint | `MODEL_SPECS`, `MODELS_CONFIG` e `model_profiles.json` | IDs Civitai validados no servidor |
| Checkpoints | `STUDIO_ROOT/models` e `HF_HUB_CACHE` | Download executado somente pelo servidor |
| LoRAs | `STUDIO_ROOT/loras` | Catálogo Civitai e download executados somente pelo servidor |
| Resultados e metadata | Conta MEGA em `MEGA_FOLDER` | Credenciais exclusivas do processo Colab |
| Exclusão | `DELETE /api/history/<job_id>` | Sessão autenticada, CSRF e confirmação remota |

## Contrato de job

```json
{
  "id": "uuid",
  "status": "queued | running | completed | failed",
  "progress": 0,
  "params": {
    "prompt": "...",
    "negative_prompt": "...",
    "model": "prefect-pony-xl-v6",
    "sampler": "euler_a",
    "seed": 123456,
    "mode": "text2img | img2img",
    "steps": 30,
    "guidance": 5.5,
    "width": 1024,
    "height": 1024,
    "strength": 0.65,
    "loras": [{"version_id": 123, "weight": 0.8}]
  },
  "result": {"filename": "uuid.png", "mega_synced": true},
  "error": null
}
```

O app aceita um máximo de 3 LoRAs por job, resoluções múltiplas de 64 entre 512 e 1024 pixels para SDXL, 10–60 steps e escala de guidance entre 1 e 15. O Prefect Pony XL V6 aceita TXT→IMG, IMG→IMG e LoRAs compatíveis. O Pony V7 aceita inicialmente apenas TXT→IMG entre 768 e 1024 pixels e exige `flow_euler`; IMG→IMG e LoRAs são bloqueadas por perfil. Os perfis individuais podem fornecer defaults diferentes, mas os limites de segurança continuam sendo validados no servidor.
