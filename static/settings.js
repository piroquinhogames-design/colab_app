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
  };
  Object.entries(fieldIds).forEach(([settingKey, elementId]) => {
    const field = query(`#${elementId}`);
    if (settings[settingKey] !== undefined && field) field.value = settings[settingKey];
  });
  ['steps', 'guidance', 'strength'].forEach((id) => {
    const output = query(`#${id}-value`);
    if (output && settings[id] !== undefined) output.value = settings[id];
  });
  state.selectedLoras = Array.isArray(settings.loras) ? settings.loras
    .filter((item) => item && Number.isFinite(Number(item.version_id)) && typeof item.name === 'string')
    .slice(0, 3).map((item) => ({ ...item, weight: Number(item.weight ?? .8) })) : [];
  renderSelectedLoras();
  setMode(settings.mode === 'img2img' ? 'img2img' : 'text2img');
  if (typeof setEditLevel === 'function') setEditLevel(settings.edit_level || 'medium', {silent: true});
  log('Último prompt e parâmetros restaurados do arquivo MEGA.');
  return true;
}
