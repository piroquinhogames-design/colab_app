# Notas de validação visual — ModelLab Studio

A interface mantém a identidade escura liquid glass do ModelLab Studio, com painéis translúcidos, reflexos, bordas luminosas e animações suaves. O formulário principal exibe apenas o modelo ativo; checkpoint e sampler ficam na aba Configurações.

O padrão declarado no HTML e no backend é **Pony V7 Base**, família **Pony**, com defaults adaptativos de Pony. A loja de LoRAs inicia em Pony e os rótulos das lojas de LoRAs e prompts acompanham a família do modelo selecionado.

O perfil Pony V7 Base usa o pipeline SDXL clássico; a validação visual de geração deve ser executada em uma sessão T4 com o checkpoint disponível no cache ou com `CIVITAI_TOKEN` configurado para download.


Validações executadas:

- `python3 -m py_compile server.py contract_check.py launch_colab.py`.
- `node --check static/app.js` e `node --check static/settings.js`.
- Smoke test da restauração de modelo, sampler, parâmetros, modo e LoRAs.
- Contratos de autenticação, bootstrap, catálogo Pony, persistência MEGA, sincronização e exclusão: `CONTRATOS_COLAB_OK`.
- Verificação automatizada de referências entre IDs do HTML e seletores usados pelo frontend: nenhum ID ausente.

O teste visual de geração não baixa o checkpoint nem inicia o pipeline Pony/SDXL quando o checkpoint não está disponível no cache e não há token para download; essa limitação é registrada para evitar uma falsa validação de renderização.

Validação visual adicional em 16/08/2026: a instância local autenticada abriu com `PONY V7 BASE`, badge `PONY // PERFIL ADAPTATIVO` e botão `CONFIGURAÇÕES`; o checkpoint não aparece como select no formulário principal. O painel exibiu a mensagem correta de que LoRAs devem ser treinadas para Pony ou explicitamente compatíveis. MEGA permaneceu indisponível porque a instância de teste não recebeu credenciais, como esperado.

A tentativa de executar `node --test static/settings.test.js static/settings.dom.test.js` não foi concluída porque o repositório não possui `package.json` nem a dependência `vitest`; em substituição, o smoke test nativo de restauração terminou com `SETTINGS_SMOKE_OK`.

A aba **Configurações** foi aberta visualmente e exibiu `settings-model` com `Pony V7 Base · PONY`, `settings-sampler` com Euler a e o botão `ABRIR LOJA DE MODELOS`. A loja abriu com filtros de consulta, tag, família, ordenação, conteúdo adulto e paginação; o texto inicial confirmou `Família inicial: Pony // Pony V7 Base`.

Após fechar a loja de modelos e a aba Configurações, o painel principal continuou exibindo `PONY V7 BASE`, `PONY // PERFIL ADAPTATIVO` e o botão separado `CONFIGURAR`, sem reintroduzir um seletor de checkpoint no formulário de geração.

A Loja de Adaptadores foi aberta visualmente e confirmou o cabeçalho `CIVITAI / PONY ECOSYSTEM`, a descrição de que a loja acompanha a família ativa e resultados como `Pony Turbo LoRA`, `Pony Highres/Aesthetic Boost` e outras entradas com base Pony. O rodapé também informou que a loja inicia em Pony.

O botão `LOJA DE PROMPTS` do arquivo foi acionado no painel inferior. A página permaneceu rolada na área de memória de imagens, então a confirmação final do diálogo será feita após reposicionar a viewport; a chamada não alterou o modelo ativo nem a família Pony.

A inspeção do DOM confirmou que `#prompt-store-dialog` existe e que o handler está correto: acionar `#open-prompt-store-top` via DOM resultou em `open: true`. A tentativa de clique visual anterior não manteve o diálogo aberto porque a viewport estava rolada na área inferior; o fluxo JavaScript da loja permanece funcional.

O botão `↝ SORTEAR` da Loja de Prompts foi testado com sucesso: a loja renovou a consulta e exibiu `4 prompts encontrados // PONY // ordem aleatória renovada // MODO PADRÃO // TOKEN AUSENTE`, mantendo os filtros e o contexto Pony.
