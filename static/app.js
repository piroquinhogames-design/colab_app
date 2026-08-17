import { restoreLastSettings as applyLastSettings } from '/static/settings.js';

const state = {
  csrf: null,
  mode: 'text2img',
  editLevel: 'medium',
  selectedLoras: [],
  catalogCursor: null,
  modelStoreCursor: null,
  modelStoreItems: [],
  promptStoreCursor: null,
  promptStoreItems: [],
  promptFilters: new Set(),
  activeJobId: null,
  pollTimer: null,
  toastTimer: null,
  archiveTimer: null,
  archiveReady: false,
  historyItems: [],
  previewJobId: null,
  models: [],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const EDIT_LEVELS = {
  low: {label: 'BAIXO', strength: 0.25},
  medium: {label: 'MÉDIO', strength: 0.55},
  high: {label: 'ALTO', strength: 0.85},
};

function syncResolutionOptions(engine) {
  const auraFlow = String(engine || '').toLowerCase() === 'auraflow';
  const minimum = auraFlow ? 768 : 512;
  const maximum = auraFlow ? 1536 : 1024;
  const sizes = Array.from({length: ((maximum - minimum) / 64) + 1}, (_, index) => minimum + index * 64);
  ['width', 'height'].forEach((id) => {
    const select = $(`#${id}`);
    if (!select) return;
    const current = Number(select.value);
    select.innerHTML = sizes.map((size) => `<option>${size}</option>`).join('');
    select.value = String(sizes.includes(current) ? current : Math.min(Math.max(1024, minimum), maximum));
  });
}

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function timestamp() {
  return new Date().toLocaleTimeString('pt-BR', {hour12: false});
}

function log(message) {
  const line = document.createElement('p');
  line.innerHTML = `<span>${timestamp()}</span> ${escapeHtml(message)}`;
  const target = $('#telemetry-log');
  target.prepend(line);
  while (target.children.length > 8) target.lastElementChild.remove();
}

function toast(message, isError = false) {
  const target = $('#toast');
  target.textContent = message;
  target.classList.toggle('error', isError);
  target.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => target.classList.remove('show'), 4200);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.csrf && !['GET', 'HEAD'].includes((options.method || 'GET').toUpperCase())) {
    headers.set('X-CSRF-Token', state.csrf);
  }
  const response = await fetch(path, {...options, headers});
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401) {
    window.location.assign('/');
    throw new Error('Sessão encerrada.');
  }
  if (!response.ok) throw new Error(payload.error || `Erro de comunicação (${response.status}).`);
  return payload;
}

function formatDate(value) {
  if (!value) return '--';
  return new Intl.DateTimeFormat('pt-BR', {dateStyle: 'short', timeStyle: 'short'}).format(new Date(value));
}

function setNode(online) {
  const node = $('#node-status');
  node.classList.toggle('offline', !online);
  node.querySelector('span').textContent = online ? 'NODE // ONLINE' : 'NODE // OFFLINE';
}

function updateModelProfile(modelId, {silent = false, applyDefaults = true} = {}) {
  const model = state.models.find((item) => item.id === modelId) || state.models[0];
  if (!model) return;
  const defaults = model.defaults || {};
  syncResolutionOptions(model.engine);
  if ($('#model') && $('#model').value !== model.id) $('#model').value = model.id;
  if ($('#settings-model')) $('#settings-model').value = model.id;
  const family = String(model.family || 'sdxl').toUpperCase();
  if (applyDefaults) {
    [['steps', defaults.steps], ['guidance', defaults.guidance], ['strength', defaults.strength]].forEach(([id, value]) => {
      if (value === undefined || !$(`#${id}`)) return;
      $(`#${id}`).value = value;
      const output = $(`#${id}-value`);
      if (output) output.value = value;
    });
    if (defaults.sampler && $('#sampler')) $('#sampler').value = defaults.sampler;
    if (defaults.sampler && $('#settings-sampler')) $('#settings-sampler').value = defaults.sampler;
  }
  const badge = $('#model-badge');
  if (badge) badge.textContent = `${family} // ${model.ready === false ? 'ENGINE PENDENTE' : (model.cached ? 'CACHE LOCAL' : 'DOWNLOAD SOB DEMANDA')}`;
  const engine = $('#engine-readout');
  if (engine) engine.textContent = String(model.name || model.id).slice(0, 32).toUpperCase();
  if ($('#active-model-name')) $('#active-model-name').textContent = String(model.name || model.id).toUpperCase();
  if ($('#active-model-help')) $('#active-model-help').textContent = `${family} // ${model.engine || 'ENGINE'} // ${model.notes || 'defaults adaptativos ativos.'}`;
  const img2imgButton = $('.mode[data-mode="img2img"]');
  const auraFlowOnlyText = String(model.engine || '').toLowerCase() === 'auraflow';
  if (img2imgButton) {
    img2imgButton.disabled = auraFlowOnlyText;
    img2imgButton.title = auraFlowOnlyText ? 'Pony V7 Base/AuraFlow suporta apenas TXT→IMG.' : '';
  }
  if (auraFlowOnlyText && state.mode === 'img2img') setMode('text2img', {silent: true});
  if ($('#settings-model-info')) $('#settings-model-info').textContent = `${family} // base ${model.base || '--'} // ${model.notes || 'perfil adaptativo'}`;
  if ($('#settings-engine-status')) $('#settings-engine-status').textContent = `ENGINE // ${String(model.engine || '--').toUpperCase()} // ${model.ready === false ? 'PENDENTE' : 'READY'}`;
  if ($('#catalog-family-label')) $('#catalog-family-label').textContent = family;
  if ($('#prompt-family-label')) $('#prompt-family-label').textContent = family;
  if (!silent) log(`Perfil ${model.name || model.id} selecionado; defaults adaptativos e lojas ${family} aplicados.`);
}

function renderModels(models, selectedId = '') {
  state.models = Array.isArray(models) ? models.filter((item) => item && item.id) : [];
  const select = $('#model');
  if (!select) return;
  if (!state.models.length) {
    state.models = [{id: 'pony-v7-base', name: 'Pony V7 Base', family: 'pony', base: 'Pony', engine: 'auraflow', ready: true, cached: false, defaults: {steps: 30, guidance: 5.5, strength: .65, sampler: 'euler_a'}}];
  }
  const options = state.models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name || model.id)} · ${escapeHtml(String(model.family || 'sdxl').toUpperCase())}</option>`).join('');
  select.innerHTML = options;
  if ($('#settings-model')) $('#settings-model').innerHTML = options;
  const initial = selectedId && state.models.some((item) => item.id === selectedId) ? selectedId : state.models[0].id;
  select.value = initial;
  if ($('#settings-model')) $('#settings-model').value = initial;
  updateModelProfile(initial, {silent: true});
}

function setMode(mode, {silent = false} = {}) {
  const selectedModel = state.models.find((item) => item.id === ($('#model')?.value || '')) || state.models[0];
  if (mode === 'img2img' && String(selectedModel?.engine || '').toLowerCase() === 'auraflow') {
    mode = 'text2img';
    if (!silent) toast('O Pony V7 Base usa AuraFlow e suporta apenas TXT→IMG.', true);
  }
  state.mode = mode;
  $$('.mode').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
  $('#upload-zone').classList.toggle('hidden', mode !== 'img2img');
  $('#edit-control').classList.toggle('hidden', mode !== 'img2img');
  $('.strength-control').classList.toggle('hidden', mode !== 'img2img');
  if (!silent) log(mode === 'img2img' ? 'Modo IMG→IMG selecionado; injete a imagem-base.' : 'Modo TXT→IMG selecionado.');
}

function setEditLevel(level, {silent = false} = {}) {
  const selected = EDIT_LEVELS[level] ? level : 'medium';
  const preset = EDIT_LEVELS[selected];
  state.editLevel = selected;
  const hidden = $('#edit-level');
  if (hidden) hidden.value = selected;
  $$('.edit-level').forEach((button) => button.classList.toggle('active', button.dataset.editLevel === selected));
  const readout = $('#edit-level-readout');
  if (readout) readout.textContent = preset.label;
  if (!silent && $('#strength')) {
    $('#strength').value = preset.strength;
    $('#strength-value').value = preset.strength;
  }
}

function wireParameterReadouts() {
  [['steps', 'steps-value'], ['guidance', 'guidance-value'], ['strength', 'strength-value']].forEach(([input, output]) => {
    $(`#${input}`).addEventListener('input', (event) => { $(`#${output}`).value = event.target.value; });
  });
}

function appendTag(targetId, tag) {
  const field = $(`#${targetId}`);
  field.value = field.value.trim() ? `${field.value.trim()}, ${tag}` : tag;
  field.focus();
}

function renderSelectedLoras() {
  const rack = $('#selected-loras');
  if (!state.selectedLoras.length) {
    rack.className = 'lora-rack empty-rack';
    rack.innerHTML = '<p>Nenhum adaptador conectado. O sinal será renderizado apenas pelo checkpoint.</p>';
    return;
  }
  rack.className = 'lora-rack';
  rack.innerHTML = state.selectedLoras.map((lora, index) => `
    <article class="lora-chip" data-version="${lora.version_id}">
      <strong class="lora-name" title="${escapeHtml(lora.name)}">${escapeHtml(lora.name)}</strong>
      <label><input type="range" min="0" max="1.5" step="0.05" value="${lora.weight}" data-lora-weight="${index}" /><output>${Number(lora.weight).toFixed(2)}</output></label>
      <button class="remove-lora" data-remove-lora="${index}" type="button" aria-label="Remover ${escapeHtml(lora.name)}">×</button>
    </article>`).join('');
  $$('[data-lora-weight]').forEach((input) => input.addEventListener('input', (event) => {
    const index = Number(event.currentTarget.dataset.loraWeight);
    state.selectedLoras[index].weight = Number(event.currentTarget.value);
    event.currentTarget.nextElementSibling.value = Number(event.currentTarget.value).toFixed(2);
  }));
  $$('[data-remove-lora]').forEach((button) => button.addEventListener('click', () => {
    const index = Number(button.dataset.removeLora);
    const [removed] = state.selectedLoras.splice(index, 1);
    renderSelectedLoras();
    toast(`${removed.name} desconectado.`);
  }));
}

function addLora(lora) {
  if (state.selectedLoras.some((item) => item.version_id === lora.version_id)) {
    toast('Esse LoRA já está no rack.', true);
    return;
  }
  if (state.selectedLoras.length >= 3) {
    toast('O rack suporta no máximo três LoRAs por geração.', true);
    return;
  }
  state.selectedLoras.push({...lora, weight: .8});
  renderSelectedLoras();
  $('#catalog-dialog').close();
  toast(`${lora.name} conectado com peso 0.80.`);
}

function renderCatalog(items) {
  const grid = $('#catalog-grid');
  if (!items.length) {
    grid.innerHTML = `<p class="catalog-empty">Nenhum LoRA ${escapeHtml(String($('#catalog-family-label')?.textContent || 'compatível'))} corresponde aos filtros atuais.</p>`;
    return;
  }
  grid.innerHTML = items.map((item) => {
    const versions = Array.isArray(item.versions) && item.versions.length
      ? item.versions
      : [{id: item.version_id, name: item.version || 'Versão principal', downloads: item.downloads || 0}];
    const selected = versions.find((version) => Number(version.id) === Number(item.version_id)) || versions[0];
    const versionOptions = versions.map((version) => `
      <option value="${escapeHtml(version.id)}" ${Number(version.id) === Number(selected.id) ? 'selected' : ''}>
        ${escapeHtml(version.name || `Versão ${version.id}`)}
      </option>`).join('');
    const modelData = {model_id: item.id, name: item.name || 'LoRA sem nome', versions};
    const civitaiUrl = `https://civitai.red/models/${encodeURIComponent(item.id)}?modelVersionId=${encodeURIComponent(selected.id)}`;
    return `
    <article class="catalog-card">
      ${item.image ? `<img loading="lazy" src="${escapeHtml(item.image)}" alt="" referrerpolicy="no-referrer" />` : ''}
      ${item.mature ? '<span class="mature-badge">+18</span>' : ''}
      <h3>${escapeHtml(item.name || 'LoRA sem nome')}</h3>
      <p>${escapeHtml(item.creator || 'autor desconhecido')} // ${Number(item.downloads || 0).toLocaleString('pt-BR')} DL</p>
      <label class="version-picker">VERSÃO
        <select data-version-select="${escapeHtml(item.id)}" aria-label="Versão de ${escapeHtml(item.name || 'LoRA')}">${versionOptions}</select>
      </label>
      <div class="catalog-card-actions">
        <button type="button" data-add-lora='${escapeHtml(JSON.stringify(modelData))}'>CONECTAR VERSÃO</button>
        <a class="civitai-link" data-civitai-link href="${escapeHtml(civitaiUrl)}" target="_blank" rel="noopener noreferrer">ABRIR NO CIVITAI ↗</a>
      </div>
    </article>`;
  }).join('');
  bindCatalogActions();
}

function bindCatalogActions() {
  $$('[data-add-lora]').forEach((button) => button.addEventListener('click', () => {
    try {
      const model = JSON.parse(button.dataset.addLora);
      const picker = button.closest('.catalog-card').querySelector('[data-version-select]');
      const version = model.versions.find((candidate) => String(candidate.id) === String(picker.value)) || model.versions[0];
      addLora({version_id: Number(version.id), model_id: Number(model.model_id), name: model.name, version: version.name});
    } catch { toast('Não foi possível interpretar essa versão de LoRA.', true); }
  }));
  $$('[data-version-select]').forEach((picker) => picker.addEventListener('change', () => {
    const card = picker.closest('.catalog-card');
    const link = card && card.querySelector('[data-civitai-link]');
    if (link) link.href = `https://civitai.red/models/${encodeURIComponent(picker.dataset.versionSelect)}?modelVersionId=${encodeURIComponent(picker.value)}`;
  }));
}

function renderModelStore(items) {
  const grid = $('#model-store-grid');
  if (!items.length) {
    grid.innerHTML = '<p class="catalog-empty">Nenhum checkpoint compatível foi encontrado. Tente outra família ou termo.</p>';
    return;
  }
  grid.innerHTML = items.map((item) => `
    <article class="catalog-card model-store-card">
      ${item.image ? `<img loading="lazy" src="${escapeHtml(item.image)}" alt="Preview de ${escapeHtml(item.name)}" referrerpolicy="no-referrer" />` : '<div class="model-card-placeholder">MODEL // PREVIEW</div>'}
      <span class="model-family-pill">${escapeHtml(String(item.family || item.base_model || 'MODEL').toUpperCase())}</span>
      <h3>${escapeHtml(item.name || 'Checkpoint sem nome')}</h3>
      <p>${escapeHtml(item.creator || 'autor desconhecido')} // ${Number(item.downloads || 0).toLocaleString('pt-BR')} DL</p>
      <small>${escapeHtml(item.version || 'versão principal')} // ${escapeHtml(item.base_model || '--')}</small>
      <div class="catalog-card-actions"><button type="button" data-use-model='${escapeHtml(JSON.stringify(item))}'>USAR ESTE PERFIL</button><a class="civitai-link" href="https://civitai.red/models/${encodeURIComponent(item.civitai_model_id)}?modelVersionId=${encodeURIComponent(item.version_id)}" target="_blank" rel="noopener noreferrer">ABRIR NO CIVITAI ↗</a></div>
    </article>`).join('');
  $$('[data-use-model]').forEach((button) => button.addEventListener('click', () => {
    try { useModelFromStore(JSON.parse(button.dataset.useModel)); } catch { toast('Não foi possível interpretar este perfil de modelo.', true); }
  }));
}

async function useModelFromStore(item) {
  try {
    const profile = await api('/api/model-profile', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(item)});
    const selected = [...state.models.filter((model) => model.id !== profile.id), profile];
    renderModels(selected, profile.id);
    updateModelProfile(profile.id, {applyDefaults: true});
    $('#model-store-dialog').close();
    $('#settings-dialog').showModal();
    toast(`${profile.name} aplicado // família ${String(profile.family).toUpperCase()}.`);
    log(`Perfil Civitai carregado: ${profile.name}. As lojas foram trocadas para ${String(profile.family).toUpperCase()}.`);
  } catch (error) { toast(error.message, true); }
}

async function loadModelStore({append = false} = {}) {
  const button = $('#search-model-store');
  button.disabled = true;
  const params = new URLSearchParams({
    query: $('#model-store-query').value.trim(), tag: $('#model-store-tag').value.trim(), family: $('#model-store-family').value,
    sort: $('#model-store-sort').value, limit: '24', include_adult: $('#model-store-adult').checked ? 'true' : 'false',
  });
  if (append && state.modelStoreCursor) params.set('cursor', state.modelStoreCursor);
  try {
    const payload = await api(`/api/model-catalog?${params}`);
    state.modelStoreCursor = payload.next_cursor || null;
    state.modelStoreItems = append ? [...state.modelStoreItems, ...(payload.items || [])] : (payload.items || []);
    renderModelStore(state.modelStoreItems);
    $('#next-model-store').disabled = !state.modelStoreCursor;
    $('#model-store-note').textContent = `${(payload.items || []).length} checkpoints encontrados // base ${payload.base_model || 'todas'}${payload.includes_adult ? ' // +18 INCLUÍDO' : ' // MODO PADRÃO'}`;
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

async function loadCatalog({append = false} = {}) {
  const button = $('#search-catalog');
  button.disabled = true;
  const params = new URLSearchParams({
    query: $('#catalog-query').value.trim(), tag: $('#catalog-tag').value.trim(), family: state.models.find((model) => model.id === $('#model').value)?.family || 'pony',
    sort: $('#catalog-sort').value, limit: '24', include_adult: $('#catalog-adult').checked ? 'true' : 'false',
  });
  if (append && state.catalogCursor) params.set('cursor', state.catalogCursor);
  try {
    const payload = await api(`/api/catalog?${params}`);
    state.catalogCursor = payload.next_cursor || null;
    if (append) {
      const current = $('#catalog-grid').innerHTML;
      const staging = document.createElement('div');
      renderCatalog(payload.items);
      const next = $('#catalog-grid').innerHTML;
      $('#catalog-grid').innerHTML = `${current}${next}`;
      bindCatalogActions();
    } else {
      renderCatalog(payload.items);
    }
    $('#next-catalog').disabled = !state.catalogCursor;
    const authState = payload.catalog_query?.authenticated ? 'TOKEN OK' : 'TOKEN AUSENTE';
    $('#catalog-note').textContent = `${payload.items.length} sinais encontrados // base ${payload.base_model || 'compatível'}${payload.includes_adult ? ' // +18 INCLUÍDO' : ' // MODO PADRÃO'} // ${authState}`;
  } catch (error) {
    toast(error.message, true);
  } finally { button.disabled = false; }
}

function promptStoreCard(item, index) {
  const prompt = item.prompt || 'Prompt não informado pelo autor.';
  const tags = (item.tags || []).slice(0, 5).join(' // ') || 'sem tags públicas';
  const size = item.width && item.height ? `${item.width}×${item.height}` : 'DIMENSÃO --';
  const loraCount = (item.loras || []).length;
  return `<article class="prompt-card${item.prompt ? '' : ' no-meta'}">
    <div class="prompt-card-preview"><img loading="lazy" src="${escapeHtml(item.image || '')}" alt="Preview de prompt por ${escapeHtml(item.username || 'autor desconhecido')}" referrerpolicy="no-referrer" />${item.nsfw ? '<span class="prompt-card-badge">+18</span>' : ''}</div>
    <div class="prompt-card-body"><p class="prompt-card-prompt" title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</p><div class="prompt-card-meta"><span>${escapeHtml(item.username || 'AUTOR --')}</span><span>${escapeHtml(size)}</span></div><div class="prompt-card-tags" title="${escapeHtml(tags)}">${escapeHtml(tags)}</div><button type="button" data-prompt-remix="${index}" ${item.prompt ? '' : 'disabled'}>⟳ REMIXAR PROMPT${loraCount ? ` // ${loraCount} LoRA` : ''}</button></div>
  </article>`;
}

function renderPromptStore(items, {append = false} = {}) {
  const grid = $('#prompt-store-grid');
  if (append) state.promptStoreItems = [...state.promptStoreItems, ...items];
  else state.promptStoreItems = items;
  if (!state.promptStoreItems.length) {
    grid.innerHTML = '<p class="catalog-empty">Nenhuma imagem com prompt corresponde aos filtros atuais.</p>';
    return;
  }
  grid.innerHTML = state.promptStoreItems.map(promptStoreCard).join('');
  bindPromptStoreActions();
}

function bindPromptStoreActions() {
  $$('[data-prompt-remix]').forEach((button) => button.addEventListener('click', () => remixPromptStoreItem(Number(button.dataset.promptRemix))));
}

async function remixPromptStoreItem(index) {
  const item = state.promptStoreItems[index];
  if (!item?.prompt) return;
  const auraFlow = String(activeModel().engine || '').toLowerCase() === 'auraflow';
  try {
    let mode = auraFlow ? 'text2img' : 'img2img';
    if (!auraFlow) {
      const response = await fetch(`/api/prompt-store/image?url=${encodeURIComponent(item.image)}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || 'Não foi possível preparar a imagem da Loja de Prompts.');
      }
      const blob = await response.blob();
      const file = new File([blob], `prompt-store-${item.id || Date.now()}.jpg`, {type: blob.type || 'image/jpeg'});
      const transfer = new DataTransfer();
      transfer.items.add(file);
      $('#source-image').files = transfer.files;
      $('#upload-name').textContent = `PROMPT STORE // ${file.name}`;
    }
    const minimum = auraFlow ? 768 : 512;
    const maximum = auraFlow ? 1536 : 1024;
    const sizes = Array.from({length: ((maximum - minimum) / 64) + 1}, (_, offset) => minimum + offset * 64);
    const width = sizes.includes(Number(item.width)) ? Number(item.width) : 1024;
    const height = sizes.includes(Number(item.height)) ? Number(item.height) : 1024;
    restoreLastSettings({prompt: item.prompt, negative_prompt: item.negative_prompt || '', seed: -1, steps: Number(item.steps) || 28, guidance: Number(item.guidance) || 6.5, width, height, strength: .55, mode, edit_level: 'medium', loras: item.loras || []});
    setMode(mode);
    $('#generation-form').scrollIntoView({behavior: 'smooth', block: 'start'});
    $('#prompt-store-dialog').close();
    toast(auraFlow ? 'Prompt e recursos carregados para TXT→IMG no AuraFlow.' : 'Prompt, imagem de referência e recursos disponíveis carregados para remix.');
    log(`Remix da Loja de Prompts ${String(item.id || '').slice(0, 10)} preparado.`);
  } catch (error) { toast(error.message, true); }
}

function shufflePromptItems(items) {
  const shuffled = [...items];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

async function loadPromptStore({append = false, random = false} = {}) {
  const button = $('#search-prompt-store');
  button.disabled = true;
  if (random) $('#prompt-store-sort').value = 'Random';
  const params = new URLSearchParams({
    query: $('#prompt-store-query').value.trim(), sort: $('#prompt-store-sort').value,
    limit: '24', include_adult: $('#prompt-store-adult').checked ? 'true' : 'false',
    filters: [...state.promptFilters].join(','),
    family: state.models.find((model) => model.id === $('#model').value)?.family || 'pony',
  });
  if (append && state.promptStoreCursor) params.set('cursor', state.promptStoreCursor);
  try {
    const payload = await api(`/api/prompt-store?${params}`);
    const randomMode = $('#prompt-store-sort').value === 'Random';
    const incomingItems = randomMode ? shufflePromptItems(payload.items || []) : (payload.items || []);
    state.promptStoreCursor = payload.next_cursor || null;
    renderPromptStore(incomingItems, {append});
    $('#next-prompt-store').disabled = !state.promptStoreCursor;
    const authState = payload.catalog_query?.authenticated ? 'TOKEN OK' : 'TOKEN AUSENTE';
    $('#prompt-store-note').textContent = `${incomingItems.length} prompts encontrados // ${String(payload.family || 'pony').toUpperCase()} // ${randomMode ? 'ordem aleatória renovada' : 'ordem por relevância'} // ${state.promptFilters.size ? `filtros: ${[...state.promptFilters].join(' + ')} // ` : ''}${payload.includes_adult ? '+18 INCLUÍDO' : 'MODO PADRÃO'} // ${authState}`;
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

function imageCard(job) {
  const prompt = job.params?.prompt || '';
  const loras = (job.params?.loras || []).map((item) => item.name).join(', ') || 'checkpoint puro';
  const mode = job.params?.mode === 'img2img' ? 'IMG→IMG' : 'TXT→IMG';
  const editLevel = job.params?.mode === 'img2img' ? ` // ${({'low': 'BAIXO', 'medium': 'MÉDIO', 'high': 'ALTO'})[job.params?.edit_level] || 'MÉDIO'}` : '';
  const strength = job.params?.mode === 'img2img' ? ` // DENOISE ${job.params?.strength ?? '--'}` : '';
  const model = job.params?.model || 'MODEL PROFILE';
  const sampler = job.params?.sampler ? ` // ${String(job.params.sampler).toUpperCase()}` : '';
  const settings = `${mode}${editLevel} // ${job.params?.width ?? '--'}×${job.params?.height ?? '--'} // ${job.params?.steps ?? '--'} STEPS // CFG ${job.params?.guidance ?? '--'}${strength}${sampler}`;
  const jobId = encodeURIComponent(job.id);
  return `<article class="history-card" data-history-id="${jobId}">
    <img loading="lazy" data-fullscreen="${jobId}" src="/api/history/${jobId}/image" alt="Resultado com seed ${escapeHtml(job.params?.seed)}" />
    <div class="history-overlay"><p title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</p><div class="history-actions-row"><button class="history-icon-button" data-fullscreen="${jobId}" type="button" title="Tela cheia" aria-label="Abrir imagem em tela cheia">⛶</button><button class="history-icon-button remix" data-remix-history="${jobId}" type="button" title="Remixar materiais" aria-label="Remixar esta imagem">⟳</button><button class="history-icon-button delete" data-delete-history="${jobId}" type="button" title="Excluir do histórico e do MEGA" aria-label="Excluir esta imagem">⌫</button><a class="download-link" href="/api/history/${jobId}/image?download=1" title="Baixar PNG">↓</a></div></div>
    <div class="history-meta"><strong>SEED ${escapeHtml(job.params?.seed)} // ${escapeHtml(model)}</strong><span>${escapeHtml(settings)}</span><span>${escapeHtml(loras)} // ${escapeHtml(formatDate(job.completed_at || job.created_at))}</span></div>
  </article>`;
}

function renderHistory(items) {
  const grid = $('#history-grid');
  const complete = items.filter((item) => item.status === 'completed' && (item.filename || item.id));
  state.historyItems = complete;
  grid.innerHTML = complete.length ? complete.map(imageCard).join('') : '<div class="empty-history"><span>///</span><p>O arquivo ainda não contém sinais gerados.</p></div>';
  bindHistoryActions();
}

function historyJob(jobId) {
  return state.historyItems.find((item) => String(item.id) === String(jobId));
}

function openImagePreview(jobId) {
  const job = historyJob(jobId);
  if (!job) return;
  state.previewJobId = job.id;
  const prompt = job.params?.prompt || 'Prompt não disponível.';
  const loras = (job.params?.loras || []).map((item) => `${item.name} (${Number(item.weight ?? .8).toFixed(2)})`).join(', ') || 'checkpoint puro';
  const level = ({low: 'BAIXO', medium: 'MÉDIO', high: 'ALTO'})[job.params?.edit_level] || 'MÉDIO';
  $('#image-dialog-title').textContent = `SEED ${job.params?.seed ?? '--'} // ${level}`;
  $('#image-dialog-preview').src = `/api/history/${encodeURIComponent(job.id)}/image`;
  $('#image-dialog-preview').alt = `Imagem gerada a partir do prompt ${prompt}`;
  $('#image-dialog-prompt').textContent = prompt;
  $('#image-dialog-meta').innerHTML = `${escapeHtml(job.params?.model || 'MODEL PROFILE')} // ${escapeHtml(job.params?.sampler || 'euler_a')}<br />${escapeHtml(job.params?.mode === 'img2img' ? 'IMG→IMG' : 'TXT→IMG')} // ${escapeHtml(job.params?.width)}×${escapeHtml(job.params?.height)} // ${escapeHtml(job.params?.steps)} STEPS // CFG ${escapeHtml(job.params?.guidance)}<br />${escapeHtml(loras)}<br />${escapeHtml(formatDate(job.completed_at || job.created_at))}`;
  $('#image-dialog').showModal();
}

async function remixHistoryJob(jobId) {
  const job = historyJob(jobId);
  if (!job) return;
  const auraFlow = String(activeModel().engine || '').toLowerCase() === 'auraflow';
  try {
    let mode = auraFlow ? 'text2img' : 'img2img';
    if (!auraFlow) {
      const response = await fetch(`/api/history/${encodeURIComponent(job.id)}/image`);
      if (!response.ok) throw new Error('A imagem arquivada não está disponível para remix.');
      const blob = await response.blob();
      const file = new File([blob], `remix-${job.id}.png`, {type: blob.type || 'image/png'});
      const transfer = new DataTransfer();
      transfer.items.add(file);
      const source = $('#source-image');
      source.files = transfer.files;
      $('#upload-name').textContent = `REMIX // ${file.name}`;
    }
    restoreLastSettings({...job.params, mode, seed: -1, edit_level: job.params?.edit_level || 'medium'});
    setMode(mode);
    $('#generation-form').scrollIntoView({behavior: 'smooth', block: 'start'});
    if ($('#image-dialog').open) $('#image-dialog').close();
    toast(auraFlow ? 'Prompt e parâmetros carregados para TXT→IMG no AuraFlow.' : 'Materiais carregados: prompt, LoRAs, parâmetros e imagem-base. Seed definida como aleatória.');
    log(`Remix preparado a partir do job ${String(job.id).slice(0, 8)}.`);
  } catch (error) { toast(error.message, true); }
}

async function deleteHistoryJob(jobId) {
  const job = historyJob(jobId);
  if (!job) return;
  const confirmed = window.confirm('Excluir esta imagem do histórico local e do MEGA? Esta ação não pode ser desfeita.');
  if (!confirmed) return;
  try {
    const result = await api(`/api/history/${encodeURIComponent(job.id)}`, {method: 'DELETE'});
    if ($('#image-dialog').open && String(state.previewJobId) === String(job.id)) $('#image-dialog').close();
    await refreshHistory();
    toast(result.remote_deleted ? 'Imagem excluída do histórico local e do MEGA.' : 'Imagem excluída do histórico local.');
    log(`Job ${String(job.id).slice(0, 8)} removido do arquivo.`);
  } catch (error) { toast(error.message, true); }
}

function bindHistoryActions() {
  $$('[data-fullscreen]').forEach((element) => element.addEventListener('click', (event) => {
    event.preventDefault();
    openImagePreview(decodeURIComponent(element.dataset.fullscreen));
  }));
  $$('[data-remix-history]').forEach((button) => button.addEventListener('click', () => remixHistoryJob(decodeURIComponent(button.dataset.remixHistory))));
  $$('[data-delete-history]').forEach((button) => button.addEventListener('click', () => deleteHistoryJob(decodeURIComponent(button.dataset.deleteHistory))));
}

function setArchiveState(archive) {
  if (!archive) return;
  const available = Boolean(archive.available);
  const ready = archive.ready !== false;
  $('#archive-readout').textContent = available ? `MEGA // ${archive.folder || 'ModelLabStudio'}` : (ready ? 'MEGA // INDISPONÍVEL' : 'MEGA // CONECTANDO');
  $('#archive-readout').style.color = available ? 'var(--acid)' : (ready ? 'var(--danger)' : 'var(--muted)');
}

async function refreshArchiveState() {
  try {
    const payload = await api('/api/bootstrap');
    setArchiveState(payload.archive);
    const ready = payload.archive?.ready !== false;
    if (!ready) return;
    clearInterval(state.archiveTimer);
    state.archiveTimer = null;
    const wasReady = state.archiveReady;
    state.archiveReady = true;
    if (payload.archive?.available && !wasReady) {
      restoreLastSettings(payload.last_settings);
      await refreshHistory({sync: true});
      log('Conexão MEGA concluída; histórico, imagens pendentes e último prompt atualizados automaticamente.');
    }
  } catch {
    // O bootstrap inicial continua funcional; a próxima tentativa fará a atualização.
  }
}

async function refreshHistory({sync = false} = {}) {
  try {
    const payload = await api(sync ? '/api/history/sync' : '/api/history', sync ? {method: 'POST'} : {});
    renderHistory(payload.items || []);
    $('#queue-readout').textContent = `${(payload.items || []).filter((item) => ['queued', 'running'].includes(item.status)).length} JOBS`;
    if (sync && payload.archive && !payload.archive.available) toast(payload.archive.error || 'MEGA indisponível para sincronização.', true);
    else if (sync) {
      const synced = payload.synced || 0;
      const restored = payload.restored || 0;
      const prompt = payload.last_settings_synced ? ' último prompt atualizado.' : '';
      toast(`${synced} imagem(ns) reenviada(s), ${restored} registro(s) restaurado(s).${prompt}`);
    }
  } catch (error) { toast(error.message, true); }
}

function setTelemetry(job) {
  const status = job?.status || 'idle';
  const progress = job?.progress || 0;
  $('#progress-bar').style.width = `${progress}%`;
  $('#progress-number').textContent = `${progress}%`;
  $('#vram-readout').textContent = `VRAM // ${job?.vram_gb ? `${job.vram_gb} GB` : '--'}`;
  const labels = {queued:'NO BUFFER', running:'SINAL EM PROCESSAMENTO', completed:'SINAL ARQUIVADO', failed:'FALHA DE SINAL', idle:'EM ESPERA'};
  $('#telemetry-title').textContent = labels[status] || 'EM ESPERA';
  $('#telemetry-subtitle').textContent = job?.error || (status === 'running' ? `Job ${job.id.slice(0, 8)} em execução.` : 'Sem job ativo no buffer.');
  $('#job-state').textContent = labels[status] || 'PRONTO PARA INICIAR';
  $('#generate').disabled = ['queued', 'running'].includes(status);
}

async function pollJob() {
  if (!state.activeJobId) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.activeJobId)}`);
    setTelemetry(job);
    if (['completed', 'failed'].includes(job.status)) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      state.activeJobId = null;
      await refreshHistory();
      if (job.status === 'completed') {
        log(`Render ${job.id.slice(0, 8)} concluído. Arquivo MEGA: ${job.mega_synced ? 'sincronizado' : 'pendente'}.`);
        toast(job.mega_synced ? 'Imagem renderizada e enviada ao MEGA.' : 'Imagem renderizada; sincronização MEGA pendente.', !job.mega_synced);
      } else {
        log(`Falha no render: ${job.error}`);
        toast(job.error || 'A geração falhou.', true);
      }
    }
  } catch (error) { toast(error.message, true); clearInterval(state.pollTimer); }
}

async function submitJob(event) {
  event.preventDefault();
  if (state.activeJobId) return;
  if (state.mode === 'img2img' && !$('#source-image').files[0]) {
    toast('Selecione uma imagem-base antes de iniciar img2img.', true);
    return;
  }
  const payload = {
    prompt: $('#prompt').value, negative_prompt: $('#negative-prompt').value,
    mode: state.mode, seed: Number($('#seed').value), steps: Number($('#steps').value),
    guidance: Number($('#guidance').value), width: Number($('#width').value), height: Number($('#height').value),
    strength: Number($('#strength').value), edit_level: state.editLevel, loras: state.selectedLoras,
    model: $('#model').value, sampler: $('#sampler').value,
  };
  const data = new FormData();
  data.append('payload', JSON.stringify(payload));
  if (state.mode === 'img2img') data.append('image', $('#source-image').files[0]);
  $('#generate').disabled = true;
  try {
    const job = await api('/api/jobs', {method: 'POST', body: data});
    state.activeJobId = job.id;
    setTelemetry(job);
    if (!job.preferences_persisted) log('Preferências enfileiradas localmente; arquivo MEGA indisponível para salvar o último prompt.');
    log(`Job ${job.id.slice(0, 8)} colocado na fila.`);
    toast('Job enviado para o nó T4.');
    state.pollTimer = setInterval(pollJob, 1200);
    pollJob();
  } catch (error) { toast(error.message, true); $('#generate').disabled = false; }
}

async function bootstrap() {
  try {
    const payload = await api('/api/bootstrap');
    state.csrf = payload.csrf;
    setNode(true);
    setArchiveState(payload.archive);
    state.archiveReady = payload.archive?.ready === true;
    if (!payload.archive.available && payload.archive?.ready !== true) {
      log('Arquivo MEGA conectando em segundo plano; a galeria local continua disponível.');
      clearInterval(state.archiveTimer);
      state.archiveTimer = setInterval(refreshArchiveState, 1500);
    } else if (!payload.archive.available) {
      log(payload.archive.error || 'Arquivo MEGA indisponível; a galeria local continua disponível.');
    }
    renderModels(payload.models || [], payload.last_settings?.model || payload.model?.id || '');
    restoreLastSettings(payload.last_settings);
    updateModelProfile($('#model')?.value, {silent: true, applyDefaults: false});
    if (payload.last_settings_source === 'mega') log('Manifesto de preferências recuperado do MEGA.');
    renderHistory(payload.jobs || []);
    // Releitura após o bootstrap cobre o caso em que a sessão MEGA acabou de conectar.
    await refreshHistory();
    const active = (payload.jobs || []).find((item) => ['queued', 'running'].includes(item.status));
    if (active) { state.activeJobId = active.id; setTelemetry(active); state.pollTimer = setInterval(pollJob, 1200); }
    else setTelemetry(null);
    log(payload.model.cached ? 'Checkpoint encontrado no cache do nó.' : 'Checkpoint será obtido na primeira renderização.');
  } catch (error) {
    setNode(false);
    toast(error.message, true);
  }
}

function restoreLastSettings(settings) {
  return applyLastSettings(settings, { query: $, state, renderSelectedLoras, setMode, setEditLevel, log });
}

function activeModel() {
  return state.models.find((model) => model.id === $('#model')?.value) || state.models[0] || {family: 'pony'};
}

function openModelSettings() {
  updateModelProfile($('#model')?.value, {silent: true, applyDefaults: false});
  $('#settings-dialog').showModal();
}

function openModelStore() {
  const family = String(activeModel().family || 'pony');
  const familySelect = $('#model-store-family');
  if (familySelect && [...familySelect.options].some((option) => option.value === family)) familySelect.value = family;
  state.modelStoreCursor = null;
  state.modelStoreItems = [];
  $('#model-store-dialog').showModal();
  loadModelStore();
}

function bindEvents() {
  $$('.mode').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
  $$('.edit-level').forEach((button) => button.addEventListener('click', () => setEditLevel(button.dataset.editLevel)));
  $$('.tag-bank button').forEach((button) => button.addEventListener('click', () => appendTag(button.dataset.target, button.textContent)));
  $('#source-image').addEventListener('change', (event) => { $('#upload-name').textContent = event.target.files[0] ? event.target.files[0].name : 'NENHUM ARQUIVO NO BUFFER'; });
  $('#model').addEventListener('change', (event) => updateModelProfile(event.target.value));
  $('#settings-model').addEventListener('change', (event) => { $('#model').value = event.target.value; updateModelProfile(event.target.value); });
  $('#sampler').addEventListener('change', (event) => { if ($('#settings-sampler')) $('#settings-sampler').value = event.target.value; });
  $('#settings-sampler').addEventListener('change', (event) => { $('#sampler').value = event.target.value; });
  $('#generation-form').addEventListener('submit', submitJob);
  $('#open-catalog').addEventListener('click', () => { $('#catalog-dialog').showModal(); loadCatalog(); });
  $('#close-catalog').addEventListener('click', () => $('#catalog-dialog').close());
  $('#search-catalog').addEventListener('click', () => { state.catalogCursor = null; loadCatalog(); });
  ['#catalog-query', '#catalog-tag'].forEach((selector) => $(selector).addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    state.catalogCursor = null;
    loadCatalog();
  }));
  $('#catalog-adult').addEventListener('change', () => { state.catalogCursor = null; $('#next-catalog').disabled = true; });
  $('#next-catalog').addEventListener('click', () => loadCatalog({append: true}));
  $('#open-model-settings').addEventListener('click', openModelSettings);
  $('#open-model-settings-inline').addEventListener('click', openModelSettings);
  $('#close-model-settings').addEventListener('click', () => $('#settings-dialog').close());
  $('#open-model-store').addEventListener('click', openModelStore);
  $('#close-model-store').addEventListener('click', () => $('#model-store-dialog').close());
  $('#search-model-store').addEventListener('click', () => { state.modelStoreCursor = null; loadModelStore(); });
  $('#model-store-family').addEventListener('change', () => { state.modelStoreCursor = null; loadModelStore(); });
  $('#model-store-sort').addEventListener('change', () => { state.modelStoreCursor = null; loadModelStore(); });
  $('#model-store-adult').addEventListener('change', () => { state.modelStoreCursor = null; loadModelStore(); });
  ['#model-store-query', '#model-store-tag'].forEach((selector) => $(selector).addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); state.modelStoreCursor = null; loadModelStore(); } }));
  $('#next-model-store').addEventListener('click', () => loadModelStore({append: true}));
  const openPromptStore = () => { state.promptStoreCursor = null; $('#prompt-store-dialog').showModal(); loadPromptStore(); };
  $('#open-prompt-store').addEventListener('click', openPromptStore);
  $('#open-prompt-store-top').addEventListener('click', openPromptStore);
  $('#close-prompt-store').addEventListener('click', () => $('#prompt-store-dialog').close());
  $('#search-prompt-store').addEventListener('click', () => { state.promptStoreCursor = null; loadPromptStore(); });
  $('#prompt-store-query').addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); state.promptStoreCursor = null; loadPromptStore(); } });
  $('#prompt-store-sort').addEventListener('change', () => { state.promptStoreCursor = null; loadPromptStore(); });
  $('#prompt-store-adult').addEventListener('change', () => { state.promptStoreCursor = null; loadPromptStore(); });
  $('#random-prompt-store').addEventListener('click', () => { state.promptStoreCursor = null; loadPromptStore({random: true}); });
  $$('.prompt-filter-chip').forEach((button) => button.addEventListener('click', () => {
    const term = button.dataset.promptFilter;
    if (state.promptFilters.has(term)) state.promptFilters.delete(term); else state.promptFilters.add(term);
    button.classList.toggle('active', state.promptFilters.has(term));
    state.promptStoreCursor = null;
    loadPromptStore();
  }));
  $('#next-prompt-store').addEventListener('click', () => loadPromptStore({append: true}));
  $('#refresh-history').addEventListener('click', () => { refreshHistory({sync: true}); log('Solicitando sincronização do arquivo MEGA.'); });
  $('#close-image-dialog').addEventListener('click', () => $('#image-dialog').close());
  $('#image-dialog-remix').addEventListener('click', () => { if (state.previewJobId) remixHistoryJob(state.previewJobId); });
  $('#logout').addEventListener('click', async () => { try { await api('/api/logout', {method: 'POST'}); } finally { window.location.assign('/'); } });
  wireParameterReadouts();
  setEditLevel('medium', {silent: true});
  setMode('text2img');
}

bindEvents();
bootstrap();
