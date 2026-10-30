# ModelLab Studio

Painel HTML/CSS/JavaScript e API Flask para gerar imagens Anima com ComfyUI headless no Google Colab. WAI-ANIMA v1.0 é o perfil padrão; Nova EXAnime AM continua disponível. O túnel Cloudflare é temporário e o MEGA preserva os resultados que tenham sincronização confirmada.

## Executar no Colab

Selecione uma GPU T4 e execute:

```python
!git clone https://github.com/piroquinhogames-design/colab_app.git /content/colab_app
!python /content/colab_app/launch_colab.py
```

Se já houver um clone, atualize com `git -C /content/colab_app pull --ff-only`. O launcher solicita senha do painel, credenciais MEGA e token Civitai opcional. Recursos do Civitai podem exigir token ou acesso específico da conta. Não cole segredos em arquivos versionados.

O launcher instala o grupo de dependências com resolução normal, preservando o stack GPU instalado no Colab. Apenas `mega.py` é instalado isoladamente porque sua dependência antiga de tenacity conflita com Python moderno. `runtime_constraints.txt` fixa Pydantic 2.14.0 com pydantic-core 2.50.0, além do grupo Transformers/Tokenizers/Hub. Imports são verificados em um processo novo antes de iniciar o servidor. Isso corrige a incompatibilidade entre Pydantic e pydantic-core relatada.

## Usar no SillyTavern

Em cada execução, o setup pergunta: `Expor a API ComfyUI publicamente nesta sessão? [sim/NÃO]`. A resposta padrão é não; Enter ou entrada indisponível mantêm a API externa desativada. A escolha não é salva para a próxima sessão.

Antes da pergunta, o setup avisa que o link do ComfyUI permite gerar e cancelar imagens, consumir GPU e acessar arquivos servidos pelo backend sem autenticação. A senha do painel não protege esse link. Responda `sim` somente se aceitar esse acesso público.

Ao aceitar, o servidor prepara o modelo configurado, encoder e VAE em segundo plano. O painel fica acessível enquanto os arquivos são baixados. O launcher abre o segundo túnel apenas quando o ComfyUI e seu workflow estão prontos; uma falha na preparação mantém somente o painel acessível. Não é necessário gerar uma imagem no painel para inicializar a integração.

O console mostra dois endereços separados: `URL ATUAL DO PAINEL` e `URL DO COMFYUI PARA SILLYTAVERN`. No SillyTavern, use:

| Campo | Valor |
| --- | --- |
| Source | ComfyUI |
| Server Type | Standard Server |
| ComfyUI URL | O link HTTPS de `URL DO COMFYUI PARA SILLYTAVERN`, sem acrescentar uma porta |
| ComfyUI Workflow | ModelLab_SillyTavern.json |

1. Entre no painel com sua senha e abra o endereço `WORKFLOW SILLYTAVERN` mostrado no console para baixar o JSON. Ele também fica em `STUDIO_ROOT/integrations/ModelLab_SillyTavern.json`.
2. Toque em Connect no SillyTavern, crie um workflow pelo botão `+` e cole todo o JSON no editor. Os placeholders já estão preparados, inclusive os valores numéricos entre aspas.
3. Comece com 768×768, 24 passos, CFG 5, sampling method `euler_ancestral` e scheduler `normal`. Teste no chat com `/sd a blue cat sitting by a window`.

O JSON usa o mesmo TXT→IMG Anima do projeto, com encoder Qwen, VAE, latent de 16 canais e decode em tiles. Modelo, encoder e VAE ficam fixados para os arquivos preparados no setup; os seletores Model/VAE do SillyTavern não trocam esses arquivos neste workflow. Prompt, negativo, seed, passos, CFG, resolução, sampler e scheduler vêm do SillyTavern. Prefixos de qualidade/estilo e LoRAs escolhidos no painel não são aplicados a esses pedidos; configure os prompts no SillyTavern e, se necessário, edite o workflow. Uma cópia para o WAI-ANIMA padrão fica em [workflows/WAI_ANIMA_SillyTavern.json](workflows/WAI_ANIMA_SillyTavern.json).

O SillyTavern conversa diretamente com ComfyUI: esses pedidos não passam pela fila limitada, histórico, ampliação ou sincronização MEGA do painel. A fila do ComfyUI é compartilhada com as gerações do painel. Cada túnel é recuperado separadamente quando cai, com um novo link; atualize a URL no SillyTavern se o endereço mudar. Parar a célula encerra ambos os túneis e o backend.

## Treinamento de LoRAs Anima

O botão **TREINAR LoRA** abre o fluxo de treinamento no próprio painel. O treinador usa o script nativo `anima_train_network.py` do projeto `sd-scripts`, com o WAI-ANIMA v1.0, encoder Qwen3 e Qwen-Image VAE já utilizados pelo ComfyUI.

1. Prepare o treinador e os arquivos do modelo. As dependências de treino ficam em `STUDIO_ROOT/lora-training/venv`, isoladas do PyTorch do Colab e do ComfyUI.
2. Envie um ZIP de até 200 MB com imagens PNG/JPG/WEBP e legendas `.txt` com o mesmo nome-base. Se as imagens não tiverem legendas individuais, o painel pode aplicar uma legenda comum.
3. Defina nome, resolução (512/640/768), rank, épocas, repeats e learning rate. Confirme que o dataset é exclusivamente SFW.
4. Acompanhe estado, passos, loss e logs. O treinamento pode ser cancelado pelo painel.
5. Ao terminar, o arquivo `.safetensors` é copiado para `COMFY_ROOT/models/loras` e aparece na lista de LoRAs treinadas. Use **CONECTAR AO RACK** para aplicá-lo nas próximas gerações.

O perfil de memória T4 usa FP16 (T4 não oferece BF16 nativo), batch size 1, AdamW8bit, alpha 1, gradient checkpointing, cache de latents e saídas do encoder, VAE 2D e block swapping H2D-only com ring buffer 1 para priorizar VRAM. Antes do treino, o painel libera a VRAM e pausa o processo ComfyUI para evitar disputa pela T4; ao terminar ou cancelar, o backend é reiniciado automaticamente sem carregar o checkpoint. Se o túnel ComfyUI externo tiver sido habilitado, ele ficará indisponível somente durante o treinamento. A instalação do treinador não altera as dependências de inferência.

O limite de 200 MB vale para o ZIP; o conteúdo descompactado é limitado a 768 MB e 3000 arquivos. Imagens acima de 24 megapixels são recusadas. O checkbox SFW é uma confirmação do operador, não um classificador automático de conteúdo. Treinamento real exige uma sessão Colab com GPU CUDA compatível; os testes locais não simulam uma T4 nem garantem duração ou qualidade do resultado.

## Recursos

- TXT→IMG e workflow IMG→IMG com intensidade configurável; decode VAE em tiles e ampliação Lanczos de 1,5×/2× após gerar.
- Até três LoRAs Anima, sem remover indiscriminadamente os valores alpha.
- Progresso real do ComfyUI, estados de download/carregamento/decode e memória do backend.
- Fila limitada, cancelamento, pedidos individuais idempotentes e lotes de até quatro seeds.
- Presets, favoritos, comparação de duas imagens e reutilização de parâmetros separada de edição de imagem.
- Histórico paginado com miniaturas, status de sincronização, reenvio ao MEGA e exportação ZIP.
- PNG com workflow e parâmetros incorporados, manifestos atômicos e proteção contra caminhos externos.
- Perfil Civitai validado no servidor; arquivos com SHA-256, downloads retomáveis e verificação SafeTensor.
- Login com limite de tentativas, CSRF, cookies protegidos no túnel, CSP e proxy de previews com limites e validação de redirecionamentos.

A implementação de IMG→IMG e tiles tem testes de workflow. A geração real desses fluxos, a qualidade e o consumo de VRAM ainda precisam ser medidos na T4. Nenhum ganho de qualidade, velocidade ou capacidade de quantização é prometido sem essa medição.

## Configuração

`STUDIO_ROOT` define o armazenamento local (padrão `/content/modellab-studio`). Checkpoints ficam em `models/diffusion_models`; Qwen/VAE e LoRAs têm diretórios próprios. `COMFY_ROOT`, `COMFY_PORT` (padrão 8188) e `PORT` (painel, padrão 7860) permitem alterar os serviços. `COMFY_MEMORY_MODE` aceita `--gpu-only`, `--normalvram` ou `--lowvram`; comece com o padrão e use modo de menor VRAM se a geração real falhar por memória. `STUDIO_MAX_QUEUE` limita a fila (padrão 8). `MEGA_FOLDER` escolhe a pasta remota. Veja [arquitetura](ARCHITECTURE.md).

A configuração `STUDIO_COOKIE_SECURE=0` é exclusiva para desenvolvimento HTTP local. Em produção no túnel, mantenha cookies seguros. `STUDIO_START_WORKERS=0` desativa workers nos testes. `STUDIO_TRUSTED_HOSTS` permite hosts adicionais quando necessário.

## Testes e benchmark

```bash
python contract_check.py
npm ci
npm run check
npm test
npx playwright install chromium
npm run test:browser
```

O CI executa contratos Python, testes JS e o cenário de navegador. A suíte não baixa checkpoints nem exige GPU ou credenciais reais. O navegador verifica login, presets, controles, foco e ausência de overflow em tela móvel.

Para medir geração real em um Studio já iniciado:

```bash
python benchmark_colab.py --count 20 --model wai-anima --output benchmark-results.json
```

O script pede a senha sem exibi-la e registra 20 gerações sequenciais, alternando 512/768/1024, com tempo total, resultado e diagnóstico. A primeira execução pode incluir download e carga; as seguintes não garantem cache quente entre tamanhos. O benchmark não prova ausência de vazamento sozinho: compare diagnósticos e teste também LoRAs, IMG→IMG, cancelamento e reinício.

## Limites e próximos experimentos

A implementação mantém um worker de geração. GGUF/quantização, upscale por modelo e expansão para outras famílias precisam de loaders, pesos e validação específicos; não são apresentados como compatíveis automaticamente. Encoder/VAE compartilhados usam revisão imutável e hashes de publicação. Em Configurações, a limpeza de checkpoint local só aceita arquivos internos que não estejam carregados nem reservados na fila. O perfil é preservado e o arquivo pode ser baixado novamente. Resultados locais sem confirmação MEGA precisam ser exportados antes de encerrar o Colab.

Não há licença de código definida neste repositório. A licença dos pesos de cada modelo também precisa ser respeitada pelo usuário.
