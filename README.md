# ModelLab Studio

Este projeto executa o painel, a API e a geração de imagens no mesmo runtime Google Colab. O `launch_colab.py` instala dependências, inicia o servidor em `localhost:7860` e cria um endereço temporário público. A URL deixa de funcionar quando a célula ou a sessão Colab termina. As imagens e seus metadados são enviados ao diretório definido por `MEGA_FOLDER`, portanto não dependem do disco efêmero do Colab.

## Início rápido

No Colab, com GPU T4 selecionada em **Ambiente de execução → Alterar tipo de ambiente de execução**, execute:

```python
!rm -rf /content/colab_app
!git clone https://github.com/piroquinhogames-design/colab_app.git /content/colab_app
!python /content/colab_app/launch_colab.py
```

O inicializador solicitará, sem imprimir os valores, apenas a senha do painel, o e-mail e a senha do MEGA e, opcionalmente, o token Civitai. O perfil Prefect Pony XL V6 e o pipeline SDXL single-file são configurados automaticamente com valores padronizados; não é necessário conhecer essa parte técnica. O token é utilizado apenas pelo processo do servidor para consultar o catálogo Civitai; ele não aparece na interface nem é enviado ao navegador. A senha de acesso protege o painel durante a sessão do túnel.

A instalação usa a matriz mínima compatível de **Transformers 4.51.0+**, **Tokenizers 0.21.x**, **Diffusers 0.39.0**, **Accelerate 1.3.0+**, **Hugging Face Hub 0.34.0–0.x**, **hf-xet 1.1.0+**, **PEFT 0.17.0+**, **Safetensors 0.8.0+** e **PyCryptodome 3.21.0**, sem atualizar PyTorch, CUDA ou dependências indiretas globais do Colab.

> Se uma sessão anterior instalou uma versão incompatível de PyTorch, use **Ambiente de execução → Desconectar e excluir ambiente de execução** antes de executar o setup novamente.

## Interface e geração

O painel usa a identidade **ModelLab Studio**, com tema escuro, camadas translúcidas, reflexos e animações suaves inspiradas na linguagem visual *liquid glass*. A escolha do checkpoint e do sampler foi retirada do formulário principal e está na aba **CONFIGURAÇÕES**. O painel principal mostra apenas o modelo ativo e a família em uso.

O histórico apresenta o checkpoint e o sampler usados em cada resultado, oferece remix e possui um botão de exclusão que remove o registro local e, quando sincronizado, o PNG e o manifesto correspondente do MEGA.

## Modelo padrão: Prefect Pony XL V6

O perfil inicial é **Prefect Pony XL V6**, obtido do modelo Civitai `439889`, versão `2114187`, com base declarada como **Pony**. Esse checkpoint é tratado como **SDXL single-file fp16**, carregado diretamente pelo `StableDiffusionXLPipeline` do Diffusers, com pipeline separado para IMG→IMG. O fluxo oferece TXT→IMG, IMG→IMG, LoRAs Pony/SDXL compatíveis e economia automática de memória para uma GPU T4.

O endpoint do checkpoint é `https://civitai.com/api/download/models/2114187?fileId=2008663`. O backend baixa o arquivo com retomada HTTP (`Range`) e preserva um `.part` quando a célula é interrompida. O arquivo completo fica em `STUDIO_ROOT/models/prefect_pony_v6.fp16.safetensors`; uma vez presente, o carregador não repete a transferência. O cache Hugging Face continua disponível para componentes auxiliares de outros perfis, mas o Prefect Pony XL V6 não exige um repositório Diffusers separado.

## Configurações, famílias e presets

Os modelos ficam ocultos no formulário principal e são administrados pela aba **CONFIGURAÇÕES**. O sistema expõe perfis com família, engine, base, disponibilidade, cache local, defaults e notas. Ao trocar o modelo, steps, guidance, strength e sampler são atualizados automaticamente; o identificador escolhido acompanha o job.

| Família | Engine | Defaults iniciais | Loja de LoRAs |
|---|---|---|---|
| `sdxl-illustrious` | `sdxl` | 28 steps, CFG 6.5, Euler a | Illustrious |
| `pony` | `sdxl` | 30 steps, CFG 5.5, Euler a | Pony |
| `sdxl` | `sdxl` | 28 steps, CFG 6.5, Euler a | SDXL 1.0 |
| `flux` | engine separado | 28 steps, CFG 3.5 | Flux |
| `sd3` | engine separado | 28 steps, CFG 5.0 | SD 3 |

Os perfis `flux` e `sd3` podem aparecer na loja e ser catalogados, mas o backend não tenta gerá-los sem engines próprios. Isso evita incompatibilidades de arquitetura. O pipeline Pony/SDXL permite TXT→IMG e IMG→IMG; as LoRAs precisam estar em formato compatível com Pony/SDXL.

## Loja de modelos Civitai

A aba **CONFIGURAÇÕES → ABRIR LOJA DE MODELOS** consulta checkpoints no Civitai com busca por nome, autor, tag, família, ordenação, paginação e opção de conteúdo adulto condicionado ao `CIVITAI_TOKEN`. O usuário pode filtrar Illustrious/NoobAI, Pony, SDXL, Flux e SD3.

Ao escolher **USAR ESTE PERFIL**, o servidor registra o modelo selecionado na sessão e a interface aplica imediatamente a família, engine, base, sampler, steps, guidance, strength e filtros de loja correspondentes. A URL do modelo e a versão do Civitai ficam associadas ao perfil para que o checkpoint seja baixado sob demanda quando o engine suportar a família.

## Loja de LoRAs e Loja de Prompts

A loja de LoRAs começa em **Pony**, pois essa é a família do Prefect Pony XL V6. Ao mudar o checkpoint para Pony, Illustrious ou SDXL, a busca usa automaticamente o `baseModels` correspondente e a interface atualiza o rótulo da família. Até três LoRAs podem ser usadas simultaneamente, com pesos entre 0 e 1,5.

A Loja de Prompts também recebe a família ativa. Ela usa metadados de recursos do Civitai quando disponíveis para descartar imagens associadas a outra família, preserva busca por texto, tags combináveis, ordenação e remix, e embaralha localmente os resultados quando o modo **ALEATÓRIO** é usado. Isso produz variação real sem perder os filtros de família.

Para conteúdo marcado como adulto, é necessário habilitar **INCLUIR +18** e configurar `CIVITAI_TOKEN`; a disponibilidade continua sujeita às permissões da conta Civitai e às regras do próprio serviço. Use apenas personagens claramente adultos.

## Perfis customizados

Outros checkpoints podem ser adicionados por `MODELS_CONFIG`. O perfil herda automaticamente os defaults da família e pode substituí-los quando necessário:

```bash
export MODELS_CONFIG='[
  {
    "id": "nova-anime-xl",
    "name": "Nova Anime XL",
    "family": "sdxl-illustrious",
    "base": "Illustrious",
    "url": "https://civitai.com/api/download/models/SEU_VERSION_ID",
    "path": "/content/modellab-studio/models/nova-anime-xl.safetensors",
    "defaults": {"steps": 28, "guidance": 6.0, "strength": 0.65, "sampler": "euler_a"},
    "notes": "Preset opcional para avaliação manual."
  }
]'
```

Os samplers aceitos pelo contrato atual são `euler_a`, `euler`, `dpmpp_2m` e `dpmpp_2m_sde_gpu`. O servidor valida a família e o engine antes de enfileirar; quando o engine não está disponível, retorna uma mensagem explícita.

## Arquivo persistente e exclusão

Ao concluir um job, o servidor grava uma PNG e um JSON de metadados com prompt, seed, checkpoint, sampler, parâmetros, LoRAs, data e estado. Ambos são enviados para `MEGA_FOLDER`, que usa `ModelLabStudio` por padrão. O manifesto `last_settings.json` preserva o último prompt, modelo, sampler e parâmetros reutilizáveis.

A exclusão exige CSRF, bloqueia jobs em execução e confirma a remoção remota antes de retirar um item sincronizado da interface. Se o MEGA estiver indisponível, o app preserva o registro sincronizado e informa que é necessário reconectar; isso evita deixar uma cópia remota esquecida.

## Variáveis de ambiente

Use variáveis de ambiente antes de executar o inicializador. Nunca coloque segredos dentro de `server.py`, `app.js` ou no repositório.

| Variável | Finalidade | Padrão |
|---|---|---|
| `MEGA_FOLDER` | Pasta remota para imagens e metadados | `ModelLabStudio` |
| `MODEL_ID` | Identificador do perfil padrão | `prefect-pony-xl-v6` |
| `MODEL_URL` | Endpoint de download do perfil padrão | `https://civitai.com/api/download/models/2114187?fileId=2008663` |
| `MODEL_REPO` | Repositório Diffusers auxiliar, não usado pelo perfil padrão | vazio |
| `MODEL_PATH` | Caminho do checkpoint Civitai | `/content/modellab-studio/models/prefect_pony_v6.fp16.safetensors` |
| `MODEL_FAMILY` | Família declarada do perfil padrão | `pony` |
| `MODELS_CONFIG` | JSON com perfis adicionais | vazio; somente o perfil padrão |
| `STUDIO_ROOT` | Diretório temporário da sessão e cache padrão do Hub | `/content/modellab-studio` |
| `HF_HOME` | Raiz opcional para cache persistente do Hugging Face | `STUDIO_ROOT/huggingface-cache` |
| `HF_HUB_CACHE` | Cache de snapshots do Hub | `HF_HOME/hub` |
| `HF_XET_HIGH_PERFORMANCE` | Transferência Xet de alto desempenho | `1` |
| `PORT` | Porta local do Flask | `7860` |

## Limites operacionais

O túnel é temporário e a URL é renovada em cada sessão. Enquanto a sessão T4 estiver desligada, o painel e a geração não estarão acessíveis; os itens já sincronizados continuam na conta MEGA. O MEGA é usado para o arquivo persistente, enquanto o Colab mantém apenas o cache temporário de checkpoints, LoRAs e imagens recém-geradas.
