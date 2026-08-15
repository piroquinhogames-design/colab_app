import { describe, expect, it } from 'vitest';
import { JSDOM } from 'jsdom';
import { restoreLastSettings } from './settings.js';

function createPanel() {
  const dom = new JSDOM(`<!doctype html><body>
    <textarea id="prompt"></textarea><textarea id="negative-prompt"></textarea>
    <input id="seed"><input id="steps"><input id="guidance"><select id="width"></select><select id="height"></select><input id="strength">
    <output id="steps-value"></output><output id="guidance-value"></output><output id="strength-value"></output>
    <button class="mode" data-mode="text2img"></button><button class="mode" data-mode="img2img"></button>
    <div id="selected-loras"></div>
  </body>`);
  const { document } = dom.window;
  const state = { selectedLoras: [], mode: 'text2img' };
  return { document, state };
}

describe('restoreLastSettings com DOM', () => {
  it('preenche os campos, ativa img2img e renderiza o chip do LoRA', () => {
    const { document, state } = createPanel();
    const renderSelectedLoras = () => {
      document.querySelector('#selected-loras').innerHTML = state.selectedLoras
        .map((lora) => `<article class="lora-chip" data-version="${lora.version_id}">${lora.name}</article>`).join('');
    };
    const setMode = (mode) => {
      state.mode = mode;
      document.querySelectorAll('.mode').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
    };

    restoreLastSettings({
      prompt: 'neon portrait', negative_prompt: 'lowres', seed: 123, steps: 32, guidance: 7,
      width: 1024, height: 768, strength: .7, mode: 'img2img',
      loras: [{ version_id: 73, model_id: 42, name: 'Illustrious Style', weight: .9 }],
    }, { query: (selector) => document.querySelector(selector), state, renderSelectedLoras, setMode, log: () => {} });

    expect(document.querySelector('#prompt').value).toBe('neon portrait');
    expect(document.querySelector('#negative-prompt').value).toBe('lowres');
    expect(document.querySelector('#seed').value).toBe('123');
    expect(document.querySelector('.mode.active').dataset.mode).toBe('img2img');
    expect(document.querySelector('#selected-loras .lora-chip').textContent).toBe('Illustrious Style');
    expect(document.querySelector('#selected-loras .lora-chip').dataset.version).toBe('73');
  });
});
