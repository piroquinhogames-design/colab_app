import { restoreLastSettings as applyLastSettings } from '/static/settings.js';

const state = {
  csrf: null,
  mode: 'text2img',
  selectedLoras: [],
  catalogCursor: null,
  activeJobId: null,
  pollTimer: null,
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

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

function setMode(mode) {
  state.mode = mode;
  $$('.mode').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
  $('#upload-zone').classList.toggle('hidden', mode !== 'img2img');
  $('.strength-control').classList.toggle('hidden', mode !== 'img2img');
  log(mode === 'img2img' ? 'Modo IMG→IMG selecionado; injete a imagem-base.' : 'Modo TXT→IMG selecionado.');
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
    grid.innerHTML = '<p class="catalog-empty">Nenhum LoRA Illustrious corresponde aos filtros atuais.</p>';
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
    const civitaiUrl = `https://civitai.com/models/${encodeURIComponent(item.id)}?modelVersionId=${encodeURIComponent(selected.id)}`;
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
    if (link) link.href = `https://civitai.com/models/${encodeURIComponent(picker.dataset.versionSelect)}?modelVersionId=${encodeURIComponent(picker.value)}`;
  }));
}

async function loadCatalog({append = false} = {}) {
  const button = $('#search-catalog');
  button.disabled = true;
  const params = new URLSearchParams({
    query: $('#catalog-query').value.trim(), tag: $('#catalog-tag').value.trim(),
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
      $$('[data-add-lora]').forEach((element) => element.onclick = null);
      $$('[data-add-lora]').forEach((element) => element.addEventListener('click', () => addLora(JSON.parse(element.dataset.addLora))));
    } else {
      renderCatalog(payload.items);
    }
    $('#next-catalog').disabled = !state.catalogCursor;
    const authState = payload.catalog_query?.authenticated ? 'TOKEN OK' : 'TOKEN AUSENTE';
    $('#catalog-note').textContent = `${payload.items.length} sinais encontrados // base Illustrious${payload.includes_adult ? ' // +18 INCLUÍDO' : ' // MODO PADRÃO'} // ${authState}`;
  } catch (error) {
    toast(error.message, true);
  } finally { button.disabled = false; }
}

function imageCard(job) {
  const prompt = job.params?.prompt || '';
  const loras = (job.params?.loras || []).map((item) => item.name).join(', ') || 'checkpoint puro';
  const mode = job.params?.mode === 'img2img' ? 'IMG→IMG' : 'TXT→IMG';
  const strength = job.params?.mode === 'img2img' ? ` // STR ${job.params?.strength ?? '--'}` : '';
  const settings = `${mode} // ${job.params?.width ?? '--'}×${job.params?.height ?? '--'} // ${job.params?.steps ?? '--'} STEPS // CFG ${job.params?.guidance ?? '--'}${strength}`;
  return `<article class="history-card">
    <img loading="lazy" src="/api/history/${encodeURIComponent(job.id)}/image" alt="Resultado com seed ${escapeHtml(job.params?.seed)}" />
    <div class="history-overlay"><p title="${escapeHtml(prompt)}">${escapeHtml(prompt)}</p><a class="download-link" href="/api/history/${encodeURIComponent(job.id)}/image?download=1">↓ PNG</a></div>
    <div class="history-meta"><strong>SEED ${escapeHtml(job.params?.seed)}</strong><span>${escapeHtml(settings)}</span><span>${escapeHtml(loras)} // ${escapeHtml(formatDate(job.completed_at || job.created_at))}</span></div>
  </article>`;
}

function renderHistory(items) {
  const grid = $('#history-grid');
  const complete = items.filter((item) => item.status === 'completed' && (item.filename || item.id));
  grid.innerHTML = complete.length ? complete.map(imageCard).join('') : '<div class="empty-history"><span>///</span><p>O arquivo ainda não contém sinais gerados.</p></div>';
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
    strength: Number($('#strength').value), loras: state.selectedLoras,
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
    $('#archive-readout').textContent = payload.archive.available ? `MEGA // ${payload.archive.folder}` : 'MEGA // INDISPONÍVEL';
    $('#archive-readout').style.color = payload.archive.available ? 'var(--acid)' : 'var(--danger)';
    if (!payload.archive.available) log(payload.archive.error || 'Arquivo MEGA ainda não foi conectado.');
    restoreLastSettings(payload.last_settings);
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
  return applyLastSettings(settings, { query: $, state, renderSelectedLoras, setMode, log });
}

function bindEvents() {
  $$('.mode').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
  $$('.tag-bank button').forEach((button) => button.addEventListener('click', () => appendTag(button.dataset.target, button.textContent)));
  $('#source-image').addEventListener('change', (event) => { $('#upload-name').textContent = event.target.files[0] ? event.target.files[0].name : 'NENHUM ARQUIVO NO BUFFER'; });
  $('#generation-form').addEventListener('submit', submitJob);
  $('#open-catalog').addEventListener('click', () => { $('#catalog-dialog').showModal(); loadCatalog(); });
  $('#close-catalog').addEventListener('click', () => $('#catalog-dialog').close());
  $('#search-catalog').addEventListener('click', () => { state.catalogCursor = null; loadCatalog(); });
  $('#catalog-adult').addEventListener('change', () => { state.catalogCursor = null; $('#next-catalog').disabled = true; });
  $('#next-catalog').addEventListener('click', () => loadCatalog({append: true}));
  $('#refresh-history').addEventListener('click', () => { refreshHistory({sync: true}); log('Solicitando sincronização do arquivo MEGA.'); });
  $('#logout').addEventListener('click', async () => { try { await api('/api/logout', {method: 'POST'}); } finally { window.location.assign('/'); } });
  wireParameterReadouts();
  setMode('text2img');
}

bindEvents();
bootstrap();
