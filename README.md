# ModelLab Studio

Este projeto executa o painel, a API e a geração de imagens no mesmo runtime Google Colab. O `launch_colab.py` instala dependências, inicia o servidor em `localhost:7860` e cria um endereço temporário público. A URL deixa de funcionar quando a célula ou a sessão Colab termina. As imagens e seus metadados são enviados ao diretório definido por `MEGA_FOLDER`, portanto não dependem do disco efêmero do Colab.

## Início rápido

No Colab, com GPU T4 selecionada em **Ambiente de execução → Alterar tipo de ambiente de execução**, execute:

```python
!rm -rf /content/colab_app
!git clone https://github.com/piroquinhogames-design/colab_app.git /content/colab_app
!python /content/colab_app/launch_colab.py
```

O inicializador solicitará, sem imprimir os valores, apenas a senha do painel, o e-mail e a senha do MEGA e, opcionalmente, o token Civitai. O perfil **Nova EXAnime AM** e o backend **ComfyUI headless** são configurados automaticamente; nenhum navegador do ComfyUI é aberto. O token é utilizado apenas pelo processo do servidor para consultar/baixar recursos do Civitai; ele não aparece na interface nem é enviado ao navegador. A senha de acesso protege o painel durante a sessão do túnel.

A instalação usa a matriz mínima compatível de **Transformers 4.51.0+**, **Tokenizers 0.21.x**, **Hugging Face Hub 0.34.0–0.x**, **hf-xet 1.1.0+**, **Safetensors 0.8.0+** e **PyCryptodome 3.21.0**, além das dependências nativas do ComfyUI, sem atualizar PyTorch, CUDA ou dependências indiretas globais do Colab.

> Se uma sessão anterior instalou uma versão incompatível de PyTorch, use **Ambiente de execução → Desconectar e excluir ambiente de execução** antes de executar o setup novamente.

## Interface e geração

O painel usa a identidade **ModelLab Studio**, com tema escuro, camadas translúcidas, reflexos e animações suaves inspiradas na linguagem visual *liquid glass*. A escolha do checkpoint e do sampler foi retirada do formulário principal e está na aba **CONFIGURAÇÕES**. O painel principal mostra apenas o modelo ativo e a família em uso.

O histórico apresenta o checkpoint e o sampler usados em cada resultado, oferece remix e possui um botão de exclusão que remove o registro local e, quando sincronizado, o PNG e o manifesto correspondente do MEGA.

## Modelo padrão: Nova EXAnime AM

O perfil inicial é **Nova EXAnime AM**, modelo Civitai `2856434`, versão `3226184`, arquivo `novaExanimeAM_v10.safetensors`, com base declarada como **Anima B1 + A11** e precisão publicada como BF16. Ele não é SDXL e não é carregado pelo Diffusers. O fluxo atual oferece **TXT→IMG** e LoRAs compatíveis com Anima; IMG→IMG não é exposto porque não faz parte do workflow Anima validado.

O endpoint do checkpoint é `https://civitai.com/api/download/models/3226184?fileId=3108312`. O backend baixa o arquivo com retomada HTTP (`Range`) e preserva um `.part` quando a célula é interrompida. O arquivo completo fica em `STUDIO_ROOT/models/diffusion_models/novaExanimeAM_v10.safetensors`. Também são baixados, uma única vez, `qwen_3_06b_base.safetensors` em `models/text_encoders/` e `qwen_image_vae.safetensors` em `models/vae/`.

O servidor inicia somente o processo backend do ComfyUI e envia workflows API para `/prompt`; a UI web do ComfyUI não é aberta. O loader nativo Anima é usado com `--force-fp16` e `--fp16-intermediates`, configuração que reproduz o caminho compatível com T4 observado no ComfyUI, sem tentar usar `torch.float16` diretamente no Diffusers.

## Configurações, famílias e presets

Os modelos ficam ocultos no formulário principal e são administrados pela aba **CONFIGURAÇÕES**. O sistema expõe perfis com família, engine, base, disponibilidade, cache local, defaults e notas. Ao trocar o modelo, steps, guidance, strength e sampler são atualizados automaticamente; o identificador escolhido acompanha o job.

| Família | Engine | Defaults iniciais | Loja de LoRAs |
|---|---|---|---|
| `anima` | `comfyui` headless | 24 steps, CFG 5.0, Euler a | Anima |

O perfil padrão e os perfis escolhidos pela loja são restritos à família Anima nesta versão, porque o engine residente foi implementado especificamente para o workflow nativo do Nova EXAnime AM. Isso evita selecionar acidentalmente checkpoints SDXL, Flux ou SD3 que não podem ser carregados por este backend.

## Loja de modelos Civitai

A aba **CONFIGURAÇÕES → ABRIR LOJA DE MODELOS** consulta checkpoints Anima no Civitai com busca por nome, autor, tag, ordenação, paginação e opção de conteúdo adulto condicionado ao `CIVITAI_TOKEN`. Modelos de outras arquiteturas são recusados antes de serem registrados como perfil gerável.

Ao escolher **USAR ESTE PERFIL**, o servidor registra o modelo selecionado na sessão e a interface aplica imediatamente a família, engine, base, sampler, steps, guidance, strength e filtros de loja correspondentes. A URL do modelo e a versão do Civitai ficam associadas ao perfil para que o checkpoint seja baixado sob demanda quando o engine suportar a família.

## Loja de LoRAs e Loja de Prompts

A loja de LoRAs começa em **Anima**, pois essa é a família do Nova EXAnime AM. Até três LoRAs podem ser usadas simultaneamente, com pesos entre 0 e 1,5. O backend copia cada arquivo para `models/loras/` e encadeia os nodes `LoraLoaderModelOnly` sem descarregar o modelo-base da GPU.

A Loja de Prompts também recebe a família ativa. Ela usa metadados de recursos do Civitai quando disponíveis para descartar imagens associadas a outra família, preserva busca por texto, tags combináveis, ordenação e remix, e embaralha localmente os resultados quando o modo **ALEATÓRIO** é usado. Isso produz variação real sem perder os filtros de família.

Para conteúdo marcado como adulto, é necessário habilitar **INCLUIR +18** e configurar `CIVITAI_TOKEN`; a disponibilidade continua sujeita às permissões da conta Civitai e às regras do próprio serviço. Use apenas personagens claramente adultos.

## Perfis customizados

Outros checkpoints podem ser adicionados por `MODELS_CONFIG`. O perfil herda automaticamente os defaults da família e pode substituí-los quando necessário:

```bash
export MODELS_CONFIG='[
  {
    "id": "nova-anime-xl",
    "name": "Nova Anime XL",
    "family": "anima",
    "base": "Anima",
    "url": "https://civitai.com/api/download/models/SEU_VERSION_ID",
    "path": "/content/modellab-studio/models/nova-anime-xl.safetensors",
    "defaults": {"steps": 28, "guidance": 6.0, "strength": 0.65, "sampler": "euler_a"},
    "notes": "Preset opcional para avaliação manual."
  }
]'
```

Os samplers aceitos pelo contrato atual são `euler_a`, `euler`, `dpmpp_2m` e `dpmpp_2m_sde_gpu`. O servidor valida a família e o engine antes de enfileirar; o engine atual aceita somente a família `anima` e retorna uma mensagem explícita para qualquer arquitetura incompatível.

## Execução sem frontend do ComfyUI

O processo do ComfyUI é iniciado em segundo plano com `--disable-auto-launch` e não abre a interface. O servidor do ModelLab envia workflows API para `POST http://127.0.0.1:8188/prompt` e acompanha a conclusão pelo histórico/WebSocket. O perfil residente usa `--gpu-only`, `--force-fp16`, `--fp16-intermediates` e `--cache-none`; não combina `--gpu-only` com `--highvram`, pois são opções mutuamente exclusivas no parser do ComfyUI. Também não usa `--lowvram`, `--cpu-vae`, `--novram` ou `/free`, pois essas opções descarregariam componentes ou trocariam VRAM por RAM.

O node `ModelLabMemoryCleanup` é copiado automaticamente da pasta do código para `COMFY_ROOT/custom_nodes/modellab_memory.py` antes de o backend iniciar, pois `COMFY_ROOT` é o `--base-directory` do ComfyUI. Ele permanece no workflow para compatibilidade, mas por padrão apenas encaminha a imagem: a limpeza explícita do allocator CUDA fica desativada porque pode sincronizar a T4 por minutos justamente antes do `SaveImage`. Para habilitá-la em uma sessão específica, use `MODELLAB_CLEANUP_CUDA=1`. O node não remove referências ao checkpoint, ao Qwen ou ao VAE carregados. O servidor expõe `GET /api/comfy-health`, incluindo `memory_node_available`, para verificar backend, fila e carregamento do node sem acessar a UI. Se uma instância antiga do ComfyUI estiver residente sem o node, o servidor usa um workflow de fallback sem cleanup em vez de falhar; reinicie o processo para ativar a otimização. O stdout/stderr do processo agora fica salvo em `COMFY_ROOT/logs/comfyui.log`, e o painel inclui o final desse log no erro de inicialização e em `/api/comfy-health`.

## Arquivo persistente e exclusão

Ao concluir a renderização, o servidor grava uma PNG e um JSON de metadados com prompt, seed, checkpoint, sampler, parâmetros, LoRAs, data e estado. A interface pode exibir o resultado imediatamente; o envio da PNG e do manifesto para `MEGA_FOLDER`, que usa `ModelLabStudio` por padrão, continua em segundo plano e pode ser acompanhado pelo campo `mega_synced`. Jobs que ainda não foram sincronizados permanecem disponíveis para o retry da rota de sincronização. O manifesto `last_settings.json` preserva o último prompt, modelo, sampler e parâmetros reutilizáveis.

A exclusão exige CSRF, bloqueia jobs em execução e confirma a remoção remota antes de retirar um item sincronizado da interface. Se o MEGA estiver indisponível, o app preserva o registro sincronizado e informa que é necessário reconectar; isso evita deixar uma cópia remota esquecida.

## Variáveis de ambiente

Use variáveis de ambiente antes de executar o inicializador. Nunca coloque segredos dentro de `server.py`, `app.js` ou no repositório.

| Variável | Finalidade | Padrão |
|---|---|---|
| `MEGA_FOLDER` | Pasta remota para imagens e metadados | `ModelLabStudio` |
| `MODEL_ID` | Identificador do perfil padrão | `nova-exanime-am` |
| `MODEL_URL` | Endpoint de download do perfil padrão | `https://civitai.com/api/download/models/3226184?fileId=3108312` |
| `MODEL_REPO` | Repositório Diffusers auxiliar, não usado pelo perfil padrão | vazio |
| `MODEL_PATH` | Caminho do checkpoint Civitai | `/content/modellab-studio/models/diffusion_models/novaExanimeAM_v10.safetensors` |
| `MODEL_FAMILY` | Família declarada do perfil padrão | `anima` |
| `COMFYUI_DIR` | Clone do backend ComfyUI | `/content/ComfyUI` |
| `COMFYUI_COMMIT` | Revisão validada do backend | `c1739380c6fa…` |
| `COMFY_ROOT` | Base-directory compartilhado com `STUDIO_ROOT` | igual a `STUDIO_ROOT` |
| `COMFYUI_EXTRA_ARGS` | Flags adicionais opcionais; não use `--lowvram` se quiser manter a carga na GPU | vazio |
| `MODELLAB_CLEANUP_CUDA` | Habilita limpeza explícita do allocator após cada imagem | `0` (recomendado) |
| `MODELS_CONFIG` | JSON com perfis adicionais | vazio; somente o perfil padrão |
| `STUDIO_ROOT` | Diretório temporário da sessão e cache padrão do Hub | `/content/modellab-studio` |
| `HF_HOME` | Raiz opcional para cache persistente do Hugging Face | `STUDIO_ROOT/huggingface-cache` |
| `HF_HUB_CACHE` | Cache de snapshots do Hub | `HF_HOME/hub` |
| `HF_XET_HIGH_PERFORMANCE` | Transferência Xet de alto desempenho | `1` |
| `PORT` | Porta local do Flask | `7860` |

## Limites operacionais

O túnel é temporário e a URL é renovada em cada sessão. Enquanto a sessão T4 estiver desligada, o painel e a geração não estarão acessíveis; os itens já sincronizados continuam na conta MEGA. O MEGA é usado para o arquivo persistente, enquanto o Colab mantém apenas o cache temporário de checkpoints, LoRAs e imagens recém-geradas.
