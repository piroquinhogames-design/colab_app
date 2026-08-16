# Illustrious LoRA Studio — sincronização final

Esta versão contém o pacote executável no Google Colab T4, com servidor Flask, painel retro-futurista, geração Illustrious SDXL, suporte a múltiplos LoRAs, catálogo Civitai filtrado para Illustrious e controle autenticado de inclusão de conteúdo adulto.

O token Civitai é lido somente no processo do servidor. O histórico de imagens e os manifestos JSON são enviados ao MEGA. O manifesto `last_settings.json` preserva o último prompt, negative prompt, modo, parâmetros e LoRAs e é restaurado quando o servidor inicia.

A versão também inclui o adaptador de compatibilidade do cliente MEGA para Python 3.12, os testes de contrato Flask, os testes do módulo de restauração e o teste com DOM real. Os arquivos de teste de cliente estão em `static/settings.test.js` e `static/settings.dom.test.js`.

A sincronização foi verificada arquivo a arquivo entre este clone e `/home/ubuntu/illustrious-lora-studio/colab_app`. O Colab deve atualizar seu conteúdo a partir deste repositório, em vez de reutilizar um ZIP antigo.
