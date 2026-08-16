# Notas de validação visual — ModelLab Studio

A interface mantém a identidade escura liquid glass do ModelLab Studio, com painéis translúcidos, reflexos, bordas luminosas e animações suaves. O formulário principal exibe apenas o modelo ativo; checkpoint e sampler ficam na aba Configurações.

O padrão declarado no HTML e no backend é **Nova EXAnime AM**, família **Anima**, com defaults adaptativos de Anima. A loja de LoRAs inicia em Anima e os rótulos das lojas de LoRAs e prompts acompanham a família do modelo selecionado.

A geração do perfil Anima é bloqueada de forma intencional quando `ANIMA_ENGINE` e `ANIMA_DIFFUSERS_REPO` não estão configurados. Isso evita que o checkpoint seja enviado incorretamente ao carregador SDXL. A loja de modelos, a descoberta de LoRAs, a seleção de perfis e a restauração de configurações continuam disponíveis nesse modo.

Validações executadas:

- `python3 -m py_compile server.py contract_check.py launch_colab.py`.
- `node --check static/app.js` e `node --check static/settings.js`.
- Smoke test da restauração de modelo, sampler, parâmetros, modo e LoRAs.
- Contratos de autenticação, bootstrap, catálogo Anima, persistência MEGA, sincronização e exclusão: `CONTRATOS_COLAB_OK`.
- Verificação automatizada de referências entre IDs do HTML e seletores usados pelo frontend: nenhum ID ausente.

O teste visual de geração não baixa o checkpoint nem inicia o engine Anima quando as variáveis de engine não estão presentes; essa limitação é registrada para evitar uma falsa validação de renderização.

Validação visual adicional em 16/08/2026: a instância local autenticada abriu com `NOVA EXANIME AM`, badge `ANIMA // ENGINE PENDENTE` e botão `CONFIGURAÇÕES`; o checkpoint não aparece como select no formulário principal. O painel exibiu a mensagem correta de que LoRAs devem ser treinadas para Anima/Base ou explicitamente compatíveis. MEGA permaneceu indisponível porque a instância de teste não recebeu credenciais, como esperado.

A tentativa de executar `node --test static/settings.test.js static/settings.dom.test.js` não foi concluída porque o repositório não possui `package.json` nem a dependência `vitest`; em substituição, o smoke test nativo de restauração terminou com `SETTINGS_SMOKE_OK`.

A aba **Configurações** foi aberta visualmente e exibiu `settings-model` com `Nova EXAnime AM · ANIMA`, `settings-sampler` com Euler a e o botão `ABRIR LOJA DE MODELOS`. A loja abriu com filtros de consulta, tag, família, ordenação, conteúdo adulto e paginação; o texto inicial confirmou `Família inicial: Anima // Nova EXAnime AM`.

Após fechar a loja de modelos e a aba Configurações, o painel principal continuou exibindo `NOVA EXANIME AM`, `ANIMA // ENGINE PENDENTE` e o botão separado `CONFIGURAR`, sem reintroduzir um seletor de checkpoint no formulário de geração.

A Loja de Adaptadores foi aberta visualmente e confirmou o cabeçalho `CIVITAI / ANIMA ECOSYSTEM`, a descrição de que a loja acompanha a família ativa e resultados como `Anima Turbo LoRA`, `Anima Highres/Aesthetic Boost` e outras entradas com base Anima. O rodapé também informou que a loja inicia em Anima.

O botão `LOJA DE PROMPTS` do arquivo foi acionado no painel inferior. A página permaneceu rolada na área de memória de imagens, então a confirmação final do diálogo será feita após reposicionar a viewport; a chamada não alterou o modelo ativo nem a família Anima.

A inspeção do DOM confirmou que `#prompt-store-dialog` existe e que o handler está correto: acionar `#open-prompt-store-top` via DOM resultou em `open: true`. A tentativa de clique visual anterior não manteve o diálogo aberto porque a viewport estava rolada na área inferior; o fluxo JavaScript da loja permanece funcional.

O botão `↝ SORTEAR` da Loja de Prompts foi testado com sucesso: a loja renovou a consulta e exibiu `4 prompts encontrados // ANIMA // ordem aleatória renovada // MODO PADRÃO // TOKEN AUSENTE`, mantendo os filtros e o contexto Anima.
