# Arquitetura do ModelLab Studio

O pacote é executado integralmente em uma sessão Google Colab com GPU T4. O processo Flask serve a interface, recebe solicitações autenticadas, gerencia uma única fila de geração e publica a porta local por meio de um Cloudflare Quick Tunnel. Nenhuma chave de Civitai, credencial MEGA ou senha do estúdio é enviada ao navegador.

## Interface e seleção de família

O formulário principal mostra somente o modelo ativo. Checkpoint, sampler e loja de modelos ficam no diálogo **Configurações**. O bootstrap envia os perfis públicos com família, base, engine, disponibilidade, cache e defaults. A interface aplica o preset do perfil e atualiza automaticamente os rótulos e as lojas de LoRAs e prompts.

O perfil inicial é `nova-exanime-am`, com base `Anima B1 + A11`, modelo Civitai `2856434`, versão `3226184` e engine `comfyui`. O runtime baixa o checkpoint SafeTensor do Civitai com retomada HTTP, prepara as dependências Anima do ComfyUI e envia workflows headless para a API local. O fluxo atual oferece TXT→IMG; IMG→IMG fica bloqueado para o perfil Anima até existir um workflow compatível.

## Catálogo Civitai

A rota `GET /api/model-catalog` consulta checkpoints Civitai com query, tag, família, ordenação, paginação e filtro adulto. A base Civitai é traduzida por `civitai_base_for_family`: Illustrious/NoobAI, Pony, SDXL, Flux e SD 3. Cada versão é convertida em um perfil público com família, engine, defaults e arquivo principal.

A rota `POST /api/model-profile` recebe somente o identificador numérico do modelo/versão e cria um perfil interno validado pelo servidor. A seleção é mantida em `MODEL_SPECS` e serializada em `model_profiles.json` dentro de `STUDIO_ROOT`, permitindo que o catálogo selecionado sobreviva ao reinício do processo. A URL de download é construída no servidor a partir do version ID, não aceita uma URL arbitrária enviada pelo navegador.

## Lojas dependentes da família

A rota `GET /api/catalog` inicia na família ativa, consulta somente modelos do tipo `LORA`, aceita texto livre e IDs/URLs de modelos, usa cursor para paginação e aplica fallback quando a combinação de texto e base retorna vazia. Para Anima, o backend não depende de um valor rígido de `baseModels`, pois os enums do Civitai são mutáveis; a compatibilidade é filtrada localmente por `baseModel` e nome da versão.

A rota `GET /api/prompt-store` também recebe a família ativa. O servidor consulta imagens com metadados, extrai recursos Civitai, prompts, dimensões e LoRAs, filtra recursos de outra família quando há informação suficiente e embaralha os itens no modo `Random`. A busca textual combina prompt, negative prompt, autor, tags e família; os filtros de chips são combinados como interseção.

## Engines e perfis

O `GeneratorEngine` mantém o backend ComfyUI headless e evita repetir o download quando o checkpoint já está preparado. O job publica progresso separado para download do modelo, carregamento da pipeline e geração; a barra principal reinicia em 0% ao iniciar cada fase e permanece em 100% somente após a imagem ser salva. Flux e SD 3 podem ser catalogados, mas ficam explicitamente bloqueados até receberem engines próprios.

| Família | Engine | Loja de LoRAs | Estado padrão |
|---|---|---|---|
| `sdxl-illustrious` | `sdxl` | Illustrious | Geração disponível com pipeline SDXL |
| `pony` | `sdxl` | Pony | Geração disponível com SDXL em TXT→IMG e IMG→IMG |
| `sdxl` | `sdxl` | SDXL 1.0 | Geração disponível com pipeline SDXL |
| `flux` | engine separado | Flux | Catalogável; geração bloqueada |
| `sd3` | engine separado | SD 3 | Catalogável; geração bloqueada |

## Persistência

Cada resultado gera dois arquivos: `outputs/<job_id>.png` e `outputs/<job_id>.json`. O JSON inclui identificador, data UTC, prompts, seed efetivamente usada, modelo, sampler, parâmetros, LoRAs, estado final e referências de armazenamento. Após uma geração bem-sucedida, o PNG local libera o job imediatamente como concluído; o envio dos dois arquivos e do manifesto `last_settings.json` ao diretório configurado da conta MEGA continua em segundo plano. Ao inicializar uma nova sessão, o servidor recupera os JSONs para recompor a galeria; a imagem é baixada sob demanda apenas se não existir no cache local.

A rota `DELETE /api/history/<job_id>` exige autenticação e CSRF. Jobs em execução não podem ser removidos. Para um job sincronizado, o servidor primeiro tenta excluir `<job_id>.png` e `<job_id>.json` no MEGA; somente após essa confirmação remove o cache local e o registro em memória. Se o arquivo remoto estiver indisponível, a operação é recusada para evitar exclusão incompleta.

## Recursos e proteção

| Recurso | Rota ou armazenamento | Proteção |
|---|---|---|
| Interface e API | Processo Flask na porta 7860 | Login por senha e sessão HTTP-only |
| Geração | Fila de um trabalhador no processo Colab | Somente usuário autenticado; uma tarefa ativa por vez |
| Perfis de checkpoint | `MODEL_SPECS`, `MODELS_CONFIG` e `model_profiles.json` | IDs Civitai validados no servidor |
| Checkpoints | `STUDIO_ROOT/models` | Download executado somente pelo servidor |
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

O app aceita até `MODELLAB_MAX_LORAS` LoRAs por job, com padrão de 8, resoluções múltiplas de 64 entre 512 e 1024 pixels, 10–60 steps e escala de guidance entre 1 e 15. O perfil Nova EXAnime AM aceita TXT→IMG no workflow Anima atual. Os perfis individuais podem fornecer defaults diferentes, mas os limites de segurança continuam sendo validados no servidor.
