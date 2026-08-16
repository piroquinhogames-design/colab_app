# Arquitetura do ModelLab Studio

O pacote é executado integralmente em uma sessão Google Colab com GPU T4. O processo Flask serve a interface, recebe solicitações autenticadas, gerencia uma única fila de geração e publica a porta local por meio de um Quick Tunnel. Nenhuma chave de Civitai, credencial MEGA ou senha do estúdio é enviada ao navegador.

## Persistência

Cada resultado gera dois arquivos: `outputs/<job_id>.png` e `outputs/<job_id>.json`. O JSON inclui identificador, data UTC, prompts, seed efetivamente usada, modelo, sampler, parâmetros, LoRAs, estado final e referências de armazenamento. Após uma geração bem-sucedida, os dois arquivos e o manifesto `last_settings.json` são enviados ao diretório configurado da conta MEGA. Ao inicializar uma nova sessão, o servidor recupera os JSONs para recompor a galeria; a imagem é baixada sob demanda apenas se não existir no cache local.

A rota `DELETE /api/history/<job_id>` exige autenticação e CSRF. Jobs em execução não podem ser removidos. Para um job sincronizado, o servidor primeiro tenta excluir `<job_id>.png` e `<job_id>.json` no MEGA; somente após essa confirmação remove o cache local e o registro em memória. Se o arquivo remoto estiver indisponível, a operação é recusada para evitar exclusão incompleta.

## Perfis de modelo

O manifesto `MODEL_SPECS` é criado a partir do perfil padrão e, opcionalmente, de `MODELS_CONFIG`. Cada perfil declara `id`, `name`, `family`, `base`, `url`, `path`, `defaults` e `notes`. O bootstrap expõe os perfis ao frontend. A seleção altera os defaults de steps, guidance, strength e sampler; o backend valida a família e o sampler novamente antes de enfileirar.

O `GeneratorEngine` mantém uma pipeline SDXL em cache. Ao receber outro `model_id`, descarrega referências da pipeline anterior, libera o cache CUDA, garante que o novo checkpoint esteja disponível e cria as pipelines text-to-image e image-to-image. A implementação atualmente habilita famílias `sdxl`, `sdxl-illustrious` e `illustrious`, com samplers `euler_a` e `dpmpp_2m`. Um perfil de outra família é recusado explicitamente, evitando que uma LoRA ou scheduler incompatível falhe de forma silenciosa no worker.

| Recurso | Rota ou armazenamento | Proteção |
|---|---|---|
| Interface e API | Processo Flask na porta 7860 | Login por senha e sessão HTTP-only |
| Geração | Fila de um trabalhador no processo Colab | Somente usuário autenticado; uma tarefa ativa por vez |
| Perfis de checkpoint | `MODEL_SPECS` em memória e `MODELS_CONFIG` no ambiente | Download autenticado feito somente pelo servidor |
| Checkpoint padrão | `/content/illustrious-studio/models` | Carregamento preguiçoso e troca sob demanda |
| LoRAs | `/content/illustrious-studio/loras` | Catálogo Civitai e download executados somente pelo servidor |
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
    "model": "wai-illustrious",
    "sampler": "euler_a",
    "seed": 123456,
    "mode": "text2img | img2img",
    "steps": 28,
    "guidance": 6.5,
    "width": 1024,
    "height": 1024,
    "strength": 0.65,
    "loras": [{"version_id": 123, "weight": 0.8}]
  },
  "result": {"filename": "uuid.png", "mega_synced": true},
  "error": null
}
```

O app aceita um máximo de 3 LoRAs por job, resoluções múltiplas de 64 entre 512 e 1024 pixels, 10–60 steps e escala de guidance entre 1 e 15. Esses limites foram escolhidos para manter os jobs previsíveis em uma T4. Os perfis individuais podem fornecer defaults diferentes, mas os limites de segurança continuam sendo validados no servidor.
