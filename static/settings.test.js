import { describe, expect, it } from 'vitest';
import { restoreLastSettings } from './settings.js';

function field(value = '') {
  return { value };
}

describe('restoreLastSettings', () => {
  it('restaura prompt, negative prompt, modo e adaptadores no estado do cliente', () => {
    const elements = new Map([
      ['#prompt', field()], ['#negative-prompt', field()], ['#seed', field()], ['#steps', field()],
      ['#guidance', field()], ['#width', field()], ['#height', field()], ['#strength', field()],
      ['#steps-value', field()], ['#guidance-value', field()], ['#strength-value', field()],
    ]);
    const state = { selectedLoras: [], mode: 'text2img' };
    const rendered = { count: 0 };
    const logs = [];
    const restored = restoreLastSettings({
      prompt: 'illustrated hero', negative_prompt: 'lowres, bad anatomy', seed: 77,
      steps: 31, guidance: 7, width: 1024, height: 768, strength: .7, mode: 'img2img',
      loras: [
        { version_id: 73, model_id: 42, name: 'Adapter 1', weight: .85 },
        { version_id: 74, model_id: 43, name: 'Adapter 2', weight: .75 },
        { version_id: 75, model_id: 44, name: 'Adapter 3', weight: 1.0 },
        { version_id: 76, model_id: 45, name: 'Adapter 4', weight: .5 }
      ],
    }, {
      query: (selector) => elements.get(selector) || null,
      state,
      renderSelectedLoras: () => { rendered.count += 1; },
      setMode: (mode) => { state.mode = mode; },
      log: (message) => logs.push(message),
    });

    expect(restored).toBe(true);
    expect(elements.get('#prompt').value).toBe('illustrated hero');
    expect(elements.get('#negative-prompt').value).toBe('lowres, bad anatomy');
    expect(elements.get('#seed').value).toBe(77);
    expect(elements.get('#steps-value').value).toBe(31);
    expect(state.mode).toBe('img2img');
    expect(state.selectedLoras).toEqual([
      { version_id: 73, model_id: 42, name: 'Adapter 1', weight: .85 },
      { version_id: 74, model_id: 43, name: 'Adapter 2', weight: .75 },
      { version_id: 75, model_id: 44, name: 'Adapter 3', weight: 1.0 },
      { version_id: 76, model_id: 45, name: 'Adapter 4', weight: .5 }
    ]);
    expect(rendered.count).toBe(1);
    expect(logs).toHaveLength(1);
  });

  it('não altera o painel para um manifesto inválido', () => {
    const state = { selectedLoras: [] };
    expect(restoreLastSettings(null, {
      query: () => null, state, renderSelectedLoras: () => {}, setMode: () => {}, log: () => {},
    })).toBe(false);
  });
});
