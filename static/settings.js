export function restoreLastSettings(settings, dependencies) {
  const { query, state, renderSelectedLoras, setMode, setEditLevel, log } = dependencies;
  if (!settings || typeof settings !== 'object') return false;

  const fieldIds = {
    prompt: 'prompt',
    negative_prompt: 'negative-prompt',
    seed: 'seed',
    steps: 'steps',
    guidance: 'guidance',
    width: 'width',
    height: 'height',
    strength: 'strength',
    edit_level: 'edit-level',
    model: 'model',
    sampler: 'sampler',
  };
  Object.entries(fieldIds).forEach(([settingKey, elementId]) => {
    const field = query(`#${elementId}`);
    if (settings[settingKey] !== undefined && field) field.value = settings[settingKey];
  });
  if (settings.model !== undefined) {
    const settingsModel = query('#settings-model');
    if (settingsModel) settingsModel.value = settings.model;
  }
  if (settings.sampler !== undefined) {
    const settingsSampler = query('#settings-sampler');
    if (settingsSampler) settingsSampler.value = settings.sampler;
  }
  ['steps', 'guidance', 'strength'].forEach((id) => {
    const output = query(`#${id}-value`);
    if (output && settings[id] !== undefined) output.value = settings[id];
  });
  const maxLoras = Number(state.limits?.maxLoras || 8);
  state.selectedLoras = Array.isArray(settings.loras) ? settings.loras
    .filter((item) => item && Number.isFinite(Number(item.version_id)) && typeof item.name === 'string')
    .slice(0, maxLoras).map((item) => {
      const rawWeight = Number(item.weight ?? .8);
      const weight = Number.isFinite(rawWeight) ? Math.max(0, Math.min(1.5, rawWeight)) : .8;
      return { ...item, version_id: Number(item.version_id), model_id: item.model_id ? Number(item.model_id) : null, weight };
    }) : [];
  renderSelectedLoras();
  setMode(settings.mode === 'img2img' ? 'img2img' : 'text2img');
  if (typeof setEditLevel === 'function') setEditLevel(settings.edit_level || 'medium', {silent: true});
  log('Último prompt, modelo e parâmetros restaurados do arquivo MEGA.');
  return true;
}
