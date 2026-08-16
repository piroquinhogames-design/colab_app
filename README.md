# Illustrious LoRA Studio — Colab T4

Este pacote executa o painel, a API e a geração de imagens no mesmo runtime Google Colab. O `launch_colab.py` instala dependências, inicia o servidor em `localhost:7860` e cria um endereço temporário público. A URL deixa de funcionar quando a célula ou a sessão Colab termina. As imagens e seus metadados são enviados ao diretório definido por `MEGA_FOLDER`, portanto não dependem do disco efêmero do Colab.

## Início rápido

No Colab, com GPU T4 selecionada em **Ambiente de execução → Alterar tipo de ambiente de execução**, execute a célula abaixo. Ela baixa exatamente a versão pública do GitHub e evita reutilizar uma pasta ou ZIP antigo.

```python
!rm -rf /content/colab_app
!git clone https://github.com/piroquinhogames-design/colab_app.git /content/colab_app
!python /content/colab_app/launch_colab.py
```

O inicializador solicitará, sem imprimir os valores, a senha do painel, e-mail e senha do MEGA e, opcionalmente, seu token Civitai. O token é utilizado apenas pelo processo do servidor para consultar e baixar modelos; ele não aparece na interface nem é enviado ao navegador. Copie o endereço `trycloudflare.com` exibido no fim do setup e abra-o em qualquer dispositivo. A senha de acesso protege o painel durante a sessão do túnel. O servidor também inclui um adaptador de compatibilidade para a dependência MEGA em runtimes Python 3.12 do Colab. A instalação fixa **PEFT 0.17.0**, usado para carregar LoRAs, executa `pip check` e confirma a GPU e as versões importadas antes de iniciar o servidor; portanto, uma incompatibilidade de pacotes interrompe o setup com uma mensagem clara, em vez de falhar durante a geração.

## Operação

O primeiro render pode levar mais tempo porque o checkpoint WAI-illustrious-SDXL é baixado e carregado. Depois disso, o modelo fica em cache no disco da sessão. A loja mostra apenas LoRAs cujo modelo-base devolvido pelo Civitai é **Illustrious**. Os LoRAs selecionados são baixados pelo servidor para o cache local e até três podem ser usados simultaneamente, com pesos entre 0 e 1,5. Para solicitar resultados marcados como adultos, marque **INCLUIR +18** na loja; o servidor então envia a preferência autenticada `nsfw=true` ao Civitai. A disponibilidade continua sujeita às permissões da sua conta Civitai e às regras do próprio serviço.

Ao concluir um job, o servidor grava uma PNG e um JSON de metadados contendo prompts, seed, parâmetros, LoRAs, data e estado. Ambos são enviados ao diretório MEGA `IllustriousStudio` por padrão. No envio do job, o app também substitui o manifesto `last_settings.json` no MEGA com o último prompt, negative prompt, seed, parâmetros e LoRAs. Quando uma nova sessão começa, o app reconstrói a galeria ao baixar os JSONs do arquivo remoto, recupera uma PNG do MEGA quando ela não está no cache local e preenche novamente esses últimos controles. Imagens-base de img2img não são persistidas nesse manifesto; envie a imagem-base de novo antes de repetir esse modo.

## Configuração opcional

Use variáveis de ambiente antes de executar o inicializador somente se quiser alterar os valores abaixo. Nunca coloque segredos dentro de `server.py`, `app.js` ou em um repositório.

| Variável | Finalidade | Padrão |
|---|---|---|
| `MEGA_FOLDER` | Pasta remota para imagens e metadados | `IllustriousStudio` |
| `MODEL_URL` | Endpoint de download do checkpoint | versão atual WAI-illustrious-SDXL configurada no servidor |
| `MODEL_PATH` | Cache local do checkpoint v17.0 | `/content/illustrious-studio/models/waiIllustriousSDXL_v170.safetensors` |
| `STUDIO_ROOT` | Diretório de trabalho temporário da sessão | `/content/illustrious-studio` |
| `PORT` | Porta local do Flask | `7860` |

## Limites operacionais

O túnel é temporário e a URL é renovada em cada sessão. Enquanto a sessão T4 estiver desligada, o painel e a geração não estarão acessíveis; os itens já sincronizados continuam na sua conta MEGA. O histórico na página é reconstruído ao iniciar o servidor, mas o aplicativo precisa estar rodando para exibi-lo. O MEGA é usado para o arquivo persistente, enquanto o Colab mantém apenas cache temporário de checkpoint, LoRAs e imagens recém-geradas.
