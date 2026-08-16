# ModelLab Studio

Este projeto executa o painel, a API e a geração de imagens no mesmo runtime Google Colab. O `launch_colab.py` instala dependências, inicia o servidor em `localhost:7860` e cria um endereço temporário público. A URL deixa de funcionar quando a célula ou a sessão Colab termina. As imagens e seus metadados são enviados ao diretório definido por `MEGA_FOLDER`, portanto não dependem do disco efêmero do Colab.

## Início rápido

No Colab, com GPU T4 selecionada em **Ambiente de execução → Alterar tipo de ambiente de execução**, execute:

```python
!rm -rf /content/colab_app
!git clone https://github.com/piroquinhogames-design/colab_app.git /content/colab_app
!python /content/colab_app/launch_colab.py
```

O inicializador solicitará, sem imprimir os valores, a senha do painel, e-mail e senha do MEGA e, opcionalmente, seu token Civitai. O token é utilizado apenas pelo processo do servidor para consultar e baixar modelos; ele não aparece na interface nem é enviado ao navegador. A senha de acesso protege o painel durante a sessão do túnel.

A instalação usa a matriz testada de **Transformers 4.48.3**, **Tokenizers 0.21.0**, **Diffusers 0.32.2**, **Accelerate 1.3.0**, **Hugging Face Hub 0.28.1**, **PEFT 0.17.0** e **PyCryptodome 3.21.0**, sem atualizar PyTorch, CUDA ou dependências indiretas globais do Colab.

> Se uma sessão anterior instalou uma versão incompatível de PyTorch, use **Ambiente de execução → Desconectar e excluir ambiente de execução** antes de executar o setup novamente.

## Interface e geração

O painel usa a identidade **ModelLab Studio**, com tema escuro, camadas translúcidas, reflexos e animações suaves inspiradas na linguagem visual *liquid glass*. O histórico apresenta o checkpoint e o sampler usados em cada resultado, oferece remix e possui um botão de exclusão que remove o registro local e, quando sincronizado, o PNG e o manifesto correspondente do MEGA.

O primeiro render pode levar mais tempo porque o checkpoint é baixado e carregado. Depois disso, o modelo fica em cache no disco da sessão. O catálogo mostra LoRAs cujo modelo-base devolvido pelo Civitai é **Illustrious**. Até três LoRAs podem ser usadas simultaneamente, com pesos entre 0 e 1,5. Para conteúdo marcado como adulto, é necessário habilitar **INCLUIR +18** e configurar `CIVITAI_TOKEN`; a disponibilidade continua sujeita às permissões da conta Civitai e às regras do próprio serviço.

## Perfis de modelo

O backend não presume que todo checkpoint possui os mesmos defaults. Cada perfil declara família, caminho, URL e parâmetros iniciais. O painel recebe esses perfis no bootstrap, atualiza steps, guidance, strength e sampler ao trocar de modelo e envia o identificador selecionado com o job. O servidor valida a família antes de enfileirar e descarta ou troca a pipeline somente quando o checkpoint selecionado é diferente do que está em VRAM.

Por padrão existe um único perfil **WAI-illustrious-SDXL**. Isso mantém o projeto focado em variantes Illustrious sem prometer um checkpoint universal para anatomia. Outros checkpoints podem ser experimentados sem reestruturar o projeto por meio de `MODELS_CONFIG`, desde que sejam compatíveis com o motor SDXL/Illustrious configurado.

Exemplo de configuração com uma segunda variante local ou baixável:

```bash
export MODELS_CONFIG='[
  {
    "id": "nova-anime-xl",
    "name": "Nova Anime XL",
    "family": "sdxl-illustrious",
    "base": "Illustrious",
    "url": "https://civitai.com/api/download/models/SEU_VERSION_ID",
    "path": "/content/illustrious-studio/models/nova-anime-xl.safetensors",
    "defaults": {"steps": 28, "guidance": 6.0, "strength": 0.65, "sampler": "euler_a"},
    "notes": "Preset opcional para avaliação manual."
  }
]'
```

Os samplers aceitos atualmente são `euler_a` e `dpmpp_2m`. Se um perfil usar outra família, ele será recusado antes da geração com uma mensagem explícita, em vez de produzir um erro opaco no worker.

## Arquivo persistente e exclusão

Ao concluir um job, o servidor grava uma PNG e um JSON de metadados com prompt, seed, checkpoint, sampler, parâmetros, LoRAs, data e estado. Ambos são enviados para `MEGA_FOLDER`, que agora usa `ModelLabStudio` por padrão. O manifesto `last_settings.json` preserva o último prompt, modelo, sampler e parâmetros reutilizáveis. Na abertura de uma nova sessão, os JSONs remotos reconstroem a galeria e uma imagem é restaurada sob demanda quando não existe no cache local.

A exclusão exige CSRF, bloqueia jobs em execução e confirma a remoção remota antes de retirar um item sincronizado da interface. Se o MEGA estiver indisponível, o app preserva o registro sincronizado e informa que é necessário reconectar; isso evita deixar uma cópia remota esquecida.

## Configuração opcional

Use variáveis de ambiente antes de executar o inicializador. Nunca coloque segredos dentro de `server.py`, `app.js` ou no repositório.

| Variável | Finalidade | Padrão |
|---|---|---|
| `MEGA_FOLDER` | Pasta remota para imagens e metadados | `ModelLabStudio` |
| `MODEL_ID` | Identificador do perfil padrão | `wai-illustrious` |
| `MODEL_URL` | Endpoint de download do perfil padrão | versão WAI-illustrious-SDXL configurada no servidor |
| `MODEL_PATH` | Cache local do perfil padrão | `/content/illustrious-studio/models/waiIllustriousSDXL_v170.safetensors` |
| `MODEL_FAMILY` | Família declarada do perfil padrão | `sdxl-illustrious` |
| `MODELS_CONFIG` | JSON com perfis adicionais | vazio; somente o perfil padrão |
| `STUDIO_ROOT` | Diretório temporário da sessão | `/content/illustrious-studio` |
| `PORT` | Porta local do Flask | `7860` |

## Limites operacionais

O túnel é temporário e a URL é renovada em cada sessão. Enquanto a sessão T4 estiver desligada, o painel e a geração não estarão acessíveis; os itens já sincronizados continuam na conta MEGA. O MEGA é usado para o arquivo persistente, enquanto o Colab mantém apenas o cache temporário de checkpoints, LoRAs e imagens recém-geradas.
