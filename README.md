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

`STUDIO_ROOT` define o armazenamento local (padrão `/content/modellab-studio`). Checkpoints ficam em `models/diffusion_models`; Qwen/VAE e LoRAs têm diretórios próprios. `COMFY_ROOT`, `COMFY_PORT` e `STUDIO_PORT` permitem alterar os serviços. `COMFY_MEMORY_MODE` aceita `--gpu-only`, `--normalvram` ou `--lowvram`; comece com o padrão e use modo de menor VRAM se a geração real falhar por memória. `STUDIO_MAX_QUEUE` limita a fila (padrão 8). `MEGA_FOLDER` escolhe a pasta remota. Veja [arquitetura](ARCHITECTURE.md).

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
