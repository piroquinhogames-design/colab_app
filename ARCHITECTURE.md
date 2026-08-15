# Arquitetura do Illustrious LoRA Studio para Colab

O pacote é executado integralmente em uma sessão Google Colab com GPU T4. O processo Flask serve a interface, recebe solicitações autenticadas, gerencia uma única fila de geração e publica a porta local por meio de um Quick Tunnel. Nenhuma chave de Civitai, credencial MEGA ou senha do estúdio é enviada ao navegador.

## Persistência

Cada resultado gera dois arquivos: `outputs/<job_id>.png` e `outputs/<job_id>.json`. O JSON inclui identificador, data UTC, prompts, seed efetivamente usada, parâmetros, LoRAs, estado final e referências de armazenamento. Após uma geração bem-sucedida, os dois arquivos e o manifesto `history.json` são enviados ao diretório configurado da conta MEGA. Ao inicializar uma nova sessão, o servidor recupera o manifesto para recompor a galeria; a imagem é baixada sob demanda apenas se não existir no cache local.

| Recurso | Rota ou armazenamento | Proteção |
|---|---|---|
| Interface e API | Processo Flask na porta 7860 | Login por senha e sessão HTTP-only |
| Geração | Fila de um trabalhador no processo Colab | Somente usuário autenticado; uma tarefa ativa por vez |
| Checkpoint | `/content/illustrious-studio/models` | Download autenticado feito somente pelo servidor |
| LoRAs | `/content/illustrious-studio/loras` | Catálogo Civitai e download executados somente pelo servidor |
| Resultados e metadata | Conta MEGA em `MEGA_FOLDER` | Credenciais exclusivas do processo Colab |

## Contrato de job

```json
{
  "id": "uuid",
  "status": "queued | running | completed | failed",
  "progress": 0,
  "seed": 123456,
  "mode": "text2img | img2img",
  "params": {"steps": 28, "guidance": 6.5, "width": 1024, "height": 1024, "strength": 0.65},
  "loras": [{"versionId": 123, "weight": 0.8}],
  "result": {"filename": "uuid.png", "megaUrl": null},
  "error": null
}
```

O app aceita um máximo de 3 LoRAs por job, resoluções múltiplas de 64 entre 512 e 1024 pixels, 10–60 steps e escala de guidance entre 1 e 15. Esses limites foram escolhidos para manter os jobs previsíveis em uma T4 e podem ser alterados no arquivo de configuração.
