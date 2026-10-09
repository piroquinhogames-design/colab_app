# Arquitetura do ModelLab Studio

O launcher prepara dependências, protege as versões de torch/torchvision/torchaudio presentes, valida imports em processo novo e inicia Flask e Cloudflare Tunnel. O servidor escuta somente em loopback. O encerramento do launcher encerra o grupo de processos do servidor, incluindo ComfyUI.

`server.py` contém as rotas autenticadas, os perfis, a fila de geração e o arquivo MEGA. As responsabilidades de armazenamento (`studio_storage.py`), downloads (`studio_downloads.py`), validação Civitai (`civitai_resources.py`) e segurança (`studio_security.py`) são módulos separados. `comfy_backend.py` gerencia exclusivamente o processo e a API ComfyUI. A interface permanece HTML/CSS/JS, sem framework de aplicação.

Um worker serial gera imagens. Outro worker serial envia arquivos ao MEGA. Cancelamento e idempotência impedem execução duplicada nos pedidos individuais; a fila tem limite. Jobs persistem antes de entrar na fila e após conclusão. Reinícios marcam execuções antigas como interrompidas. Locks por job e tombstones persistentes impedem que uploads ressuscitem resultados excluídos.

O workflow usa UNETLoader, CLIPLoader Qwen, VAELoader, KSampler e VAEDecodeTiled. TXT→IMG usa EmptySD3LatentImage, com 16 canais compatíveis com Anima; IMG→IMG carrega e redimensiona a imagem antes de VAEEncode. LoRAs usam identificadores de nodes separados. O progresso vem do WebSocket do ComfyUI; sem eventos, é indeterminado. VRAM vem de system_stats do backend.

Checkpoints e LoRAs do Civitai são selecionados por metadados do servidor, tipo, família, arquivo e SHA-256. Downloads usam .part, Range, verificação estrutural e recibo do arquivo validado. Componentes compartilhados Qwen/VAE têm revisão Hugging Face imutável e SHA-256 fixados, além da verificação estrutural.

Cada PNG contém workflow e parâmetros. O ZIP de exportação inclui PNG, manifesto e workflow. MEGA armazena PNG e manifesto na pasta selecionada; uploads substituem arquivos anteriores somente após confirmação. Preferências também podem ser sincronizadas. A sessão Colab continua efêmera: resultados ainda não sincronizados devem ser exportados antes de encerrar.

A aplicação é de uma sessão e um usuário. Não é uma plataforma multiusuário e não usa banco de dados. O conjunto de registros fica em memória, com paginação da API. Para históricos muito grandes, SQLite e consulta paginada devem substituir a leitura de todos os manifestos.
