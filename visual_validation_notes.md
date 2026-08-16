# Validação visual local — ModelLab Studio

A instância local foi aberta em 16/08/2026 com `STUDIO_PASSWORD=test`, sem credenciais MEGA. A tela de acesso exibiu o título `ModelLab Studio // Access Gate` e a identidade escura com cartão translúcido, reflexo e botão de entrada.

Após o login, o painel exibiu `MODEL LAB STUDIO`, tema escuro com painéis translúcidos, bordas luminosas, animações suaves, seletor `WAI-illustrious-SDXL`, sampler `Euler a`, badge `SDXL-ILLUSTRIOUS // DOWNLOAD SOB DEMANDA`, telemetry `WAI-ILLUSTRIOUS-SDXL` e histórico vazio. O estado do arquivo apareceu como `MEGA // INDISPONÍVEL`, esperado porque a instância de preview não recebeu `MEGA_EMAIL` nem `MEGA_PASSWORD`.

A interface carregou sem erros de navegação e mostrou os controles de modelo/sampler no formulário. O teste visual não executou geração para evitar download de um checkpoint grande sem uma GPU dedicada.
