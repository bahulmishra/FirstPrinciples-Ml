/**
 * FirstPrinciple ML — Frontend Application Logic
 * Communicates with the FastAPI backend at the same origin.
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════════
const state = {
  sessionId: null,
  columns: [],              // [{name, dtype, sample}]
  features: [],             // selected feature cols
  target: null,             // selected target col
  taskType: 'regression',
  selectedAlgorithms: [],
  preprocessDone: false,
  currentStep: 0,
  trainResult: null,
  previewRows: [],          // raw preview rows from upload (used for dist chart)
};

let _distMode = 'histogram'; // current distribution chart type

const API = '';             // FastAPI served at same origin

// ═══════════════════════════════════════════════════════════════════
//  Step navigation
// ═══════════════════════════════════════════════════════════════════
const NUM_STEPS = 5;

function goToStep(n) {
  if (n < 0 || n >= NUM_STEPS) return;
  document.querySelectorAll('.panel:not(#panelIntro)').forEach((p, i) => {
    if (i <= n) {
      p.classList.add('active');
    } else {
      p.classList.remove('active');
    }
  });
  for (let i = 0; i < NUM_STEPS; i++) {
    const dot = document.getElementById(`dot${i}`);
    if (!dot) continue;
    dot.classList.toggle('active', i === n);
    dot.classList.toggle('completed', i < n);
    if (i < NUM_STEPS - 1) {
      const line = document.getElementById(`line${i}${i + 1}`);
      if (line) line.classList.toggle('completed', i < n);
    }
  }
  state.currentStep = n;

  const targetPanel = document.getElementById(`panel${n}`);
  if (targetPanel) {
    targetPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Toast notifications
// ═══════════════════════════════════════════════════════════════════
let _toastTimer = null;
function showToast(msg, type = 'info', duration = 4000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast ${type}`;
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.add('hidden'), duration);
}

// ═══════════════════════════════════════════════════════════════════
//  API helpers
// ═══════════════════════════════════════════════════════════════════
async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { const j = await res.json(); detail = j.detail || JSON.stringify(j); } catch { }
    throw new Error(detail);
  }
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════
//  Step 0 — INGEST
// ═══════════════════════════════════════════════════════════════════

// Drag-and-drop
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => fileInput.files[0] && uploadFile(fileInput.files[0]));

async function uploadFile(file) {
  document.getElementById('uploadFilename').textContent = `📄 ${file.name}`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    showToast('Uploading…', 'info');
    const data = await apiFetch('/api/upload/file', { method: 'POST', body: fd });
    handleUploadResponse(data);
  } catch (e) {
    showToast(`Upload failed: ${e.message}`, 'error');
  }
}

async function loadFromUrl() {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) { showToast('Please enter a URL', 'error'); return; }
  try {
    showToast('Fetching URL…', 'info');
    const data = await apiFetch('/api/upload/url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    handleUploadResponse(data);
  } catch (e) {
    showToast(`URL fetch failed: ${e.message}`, 'error');
  }
}

async function loadSample(filename) {
  const url = `/api/sample-data/${filename}`;
  // Use the URL endpoint — pass it through our URL loader path for consistency
  try {
    showToast(`Loading sample: ${filename}…`, 'info');
    const response = await fetch(`/api/sample-data/${filename}`);
    const blob = await response.blob();
    const file = new File([blob], filename, { type: 'text/csv' });
    await uploadFile(file);
  } catch (e) {
    showToast(`Failed to load sample: ${e.message}`, 'error');
  }
}

function handleUploadResponse(data) {
  state.sessionId = data.session_id;
  state.columns = data.columns;
  state.features = [];
  state.target = null;
  // Store full preview data for distribution chart (fetch all rows via raw parse)
  state.previewRows = data.preview || [];

  document.getElementById('shapeLabel').textContent =
    `${data.shape.rows} rows × ${data.shape.cols} cols`;

  renderPreviewTable(data.preview, data.columns);
  document.getElementById('previewCard').classList.remove('hidden');
  renderFeatureSelect();
  renderTargetSelect();
  showToast('Dataset loaded successfully!', 'success');
}

function renderPreviewTable(rows, columns) {
  if (!rows || rows.length === 0) return;
  const cols = columns.map(c => c.name);
  let html = '<table><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>';
    cols.forEach(c => html += `<td>${row[c] ?? ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('previewTable').innerHTML = html;
}

// Load sample dataset list
async function loadSampleList() {
  try {
    const data = await apiFetch('/api/sample-data');
    const container = document.getElementById('sampleChips');
    container.innerHTML = '';
    if (!data.datasets || data.datasets.length === 0) {
      container.innerHTML = '<span class="chip-loader">No samples found</span>';
      return;
    }
    data.datasets.forEach(name => {
      const chip = document.createElement('div');
      chip.className = 'chip';
      chip.textContent = name;
      chip.onclick = () => loadSample(name);
      container.appendChild(chip);
    });
  } catch {
    document.getElementById('sampleChips').innerHTML = '<span class="chip-loader">Could not load samples</span>';
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Step 1 — CONFIGURE
// ═══════════════════════════════════════════════════════════════════

function setTask(type) {
  state.taskType = type;
  document.getElementById('btnRegression').classList.toggle('active', type === 'regression');
  document.getElementById('btnClassification').classList.toggle('active', type === 'classification');
  document.getElementById('taskInfo').innerHTML = type === 'regression'
    ? '<strong>Regression:</strong> Predict a continuous numeric value (e.g., house price, temperature).'
    : '<strong>Classification:</strong> Predict a discrete category/class label (e.g., species, sentiment).';

  // Show/hide algorithm sections in step 3
  document.getElementById('algoRegression').classList.toggle('hidden', type !== 'regression');
  document.getElementById('algoClassification').classList.toggle('hidden', type !== 'classification');

  // Reset algorithm selection
  document.querySelectorAll('.algo-card').forEach(c => {
    c.classList.remove('selected');
    c.querySelector('.algo-check').textContent = '□';
  });
  state.selectedAlgorithms = [];
}

function renderFeatureSelect() {
  const container = document.getElementById('featureList');
  container.innerHTML = '';
  state.columns.forEach(col => {
    const item = document.createElement('div');
    item.className = 'col-item';
    item.dataset.col = col.name;
    item.innerHTML = `
      <div class="col-check"></div>
      <span class="col-name">${col.name}</span>
      <span class="col-dtype">${col.dtype}</span>`;
    item.addEventListener('click', () => {
      const i = state.features.indexOf(col.name);
      if (i === -1) {
        state.features.push(col.name);
        item.classList.add('selected');
      } else {
        state.features.splice(i, 1);
        item.classList.remove('selected');
      }
      renderFeatureDistribution();
    });
    container.appendChild(item);
  });
}

// ═══════════════════════════════════════════════════════════════════
//  Feature Distribution Chart
// ═══════════════════════════════════════════════════════════════════

const DIST_PALETTE = [
  '#6366f1', '#34d399', '#f87171', '#fbbf24', '#818cf8',
  '#38bdf8', '#fb923c', '#a78bfa', '#4ade80', '#f472b6',
];

function switchDistMode(mode) {
  _distMode = mode;
  document.querySelectorAll('.dist-toggle').forEach(b => b.classList.remove('active'));
  const btnMap = { histogram: 'distBtnHistogram', box: 'distBtnBox' };
  if (document.getElementById(btnMap[mode])) {
    document.getElementById(btnMap[mode]).classList.add('active');
  }
  renderFeatureDistribution();
}

function renderFeatureDistribution() {
  const distCard = document.getElementById('distCard');
  const plotEl = document.getElementById('featureDistPlot');

  // Only numeric features that actually exist in previewRows
  const numericFeats = state.features.filter(f => {
    const col = state.columns.find(c => c.name === f);
    if (!col) return false;
    return col.dtype.includes('int') || col.dtype.includes('float');
  });

  if (numericFeats.length === 0) {
    distCard.classList.add('hidden');
    return;
  }
  distCard.classList.remove('hidden');

  // We need the full column data, not just 5-row preview.
  // We'll pull the values from the already-fetched preview rows for instant render,
  // but we first check if we have enough — if only preview rows exist we label it as sample.
  const usingPreview = state.previewRows.length > 0;
  const rows = state.previewRows;  // all rows FastAPI returned (typically first 5)

  // Build traces
  const layout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(248,250,252,0.8)',
    margin: { t: 20, r: 20, b: 60, l: 60 },
    font: { color: '#334155', family: 'Inter', size: 11 },
    legend: { bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.25 },
    barmode: 'overlay',
    height: 340,
    xaxis: { gridcolor: 'rgba(0,0,0,0.05)', zeroline: false },
    yaxis: {
      gridcolor: 'rgba(0,0,0,0.05)', zeroline: false,
      title: _distMode === 'histogram' ? 'Count' : ''
    },
  };

  // Fetch full column data from server using the session
  // Since we stored only 5 preview rows, fire a lightweight approach:
  // build from previewRows but warn user it's a sample.
  // A proper approach: add a /api/column-data endpoint. We'll do that —
  // but for instant feedback we also render from preview and then refresh once full data arrives.

  const traces = numericFeats.map((feat, i) => {
    const vals = rows.map(r => parseFloat(r[feat])).filter(v => !isNaN(v));
    const color = DIST_PALETTE[i % DIST_PALETTE.length];

    if (_distMode === 'histogram') {
      return {
        type: 'histogram',
        name: feat,
        x: vals,
        opacity: 0.65,
        marker: { color, line: { width: 0 } },
        nbinsx: 60,
        histnorm: 'probability density',
      };
    } else {
      return {
        type: 'box',
        name: feat,
        y: vals,
        marker: { color },
        boxmean: 'sd',
        opacity: 0.85,
      };
    }
  });

  Plotly.react(plotEl, traces, layout, { responsive: true, displayModeBar: false });

  // Now fetch the fuller data (all values from the raw uploaded data)
  _fetchAndRefreshDist(numericFeats);
}

async function _fetchAndRefreshDist(numericFeats) {
  if (!state.sessionId) return;
  // Pull full column data via a lightweight preprocess preview call if available
  // We expose the raw df via /api/column-values endpoint added below
  try {
    const res = await fetch(`${API}/api/column-values`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, columns: numericFeats }),
    });
    if (!res.ok) return;
    const data = await res.json();

    const plotEl = document.getElementById('featureDistPlot');
    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'rgba(248,250,252,0.8)',
      margin: { t: 20, r: 20, b: 60, l: 60 },
      font: { color: '#334155', family: 'Inter', size: 11 },
      legend: { bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.25 },
      barmode: 'overlay',
      height: 340,
      xaxis: { gridcolor: 'rgba(0,0,0,0.05)', zeroline: false },
      yaxis: {
        gridcolor: 'rgba(0,0,0,0.05)', zeroline: false,
        title: _distMode === 'histogram' ? 'Probability Density' : ''
      },
    };

    const traces = numericFeats.map((feat, i) => {
      const vals = (data.values[feat] || []).map(Number).filter(v => !isNaN(v));
      const color = DIST_PALETTE[i % DIST_PALETTE.length];

      if (_distMode === 'histogram') {
        return {
          type: 'histogram', name: feat, x: vals,
          opacity: 0.65, marker: { color, line: { width: 0 } }, nbinsx: 60,
          histnorm: 'probability density',
        };
      } else {
        return {
          type: 'box', name: feat, y: vals,
          marker: { color }, boxmean: 'sd', opacity: 0.85,
        };
      }
    });

    Plotly.react(plotEl, traces, layout, { responsive: true, displayModeBar: false });
  } catch { /* silently fail — preview data already shown */ }
}

function renderTargetSelect() {
  const container = document.getElementById('targetList');
  container.innerHTML = '';
  state.columns.forEach(col => {
    const item = document.createElement('div');
    item.className = 'col-item radio-item';
    item.dataset.col = col.name;
    item.innerHTML = `
      <div class="col-check"></div>
      <span class="col-name">${col.name}</span>
      <span class="col-dtype">${col.dtype}</span>`;
    item.addEventListener('click', () => {
      document.querySelectorAll('#targetList .col-item').forEach(el => el.classList.remove('selected'));
      item.classList.add('selected');
      state.target = col.name;
    });
    container.appendChild(item);
  });
}

async function configureAndPreprocess() {
  if (!state.sessionId) { showToast('Please upload a dataset first', 'error'); return; }
  if (state.features.length === 0) { showToast('Please select at least one feature (X)', 'error'); return; }
  if (!state.target) { showToast('Please select a target column (y)', 'error'); return; }
  if (state.features.includes(state.target)) { showToast('Target column must not be in features', 'error'); return; }

  const encoding = document.getElementById('encodingSelect').value;
  const imputation = document.getElementById('imputationSelect').value;
  const scaling = document.getElementById('scalingSelect').value;

  try {
    showToast('Configuring…', 'info');
    await apiFetch('/api/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        features: state.features,
        target: state.target,
        task_type: state.taskType,
        encoding_strategy: encoding,
        imputation_strategy: imputation,
        scaling_strategy: scaling,
      }),
    });

    showToast('Running preprocessing pipeline…', 'info');
    const result = await apiFetch('/api/preprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });

    renderPipelineLog(result, { encoding, imputation, scaling });
    renderCorrelationHeatmap(result.correlation_matrix);
    state.preprocessDone = true;
    goToStep(2);
    showToast('Preprocessing complete!', 'success');
  } catch (e) {
    showToast(`Preprocessing error: ${e.message}`, 'error', 7000);
  }
}

function renderPipelineLog(result, opts) {
  const log = document.getElementById('pipelineLog');
  const items = [
    { icon: '🔤', type: 'success', text: `<strong>Encoding:</strong> ${opts.encoding === 'onehot' ? 'One-Hot Encoding' : 'Label Encoding'} applied to all categorical (text) columns` },
    { icon: '🩹', type: 'success', text: `<strong>Imputation:</strong> Missing values handled via <em>${opts.imputation === 'mean' ? 'mean fill' : 'row drop'}</em>` },
    { icon: '📏', type: 'success', text: `<strong>Scaling:</strong> ${opts.scaling === 'zscore' ? 'Z-Score Standardization (x = (x−μ)/σ)' : 'Min-Max Normalization'}` },
    {
      icon: '✂️', type: result.dropped_cols.length > 0 ? 'warn' : 'success', text: result.dropped_cols.length > 0
        ? `<strong>Correlation Pruning:</strong> Removed ${result.dropped_cols.length} collinear feature(s): <code>${result.dropped_cols.join(', ')}</code>`
        : '<strong>Correlation Pruning:</strong> No multi-collinear pairs found (threshold |r| > 0.9)'
    },
    { icon: '✅', type: 'success', text: `<strong>Surviving features:</strong> ${result.surviving_features.join(', ')}` },
    { icon: '📊', type: 'success', text: `<strong>65:35 Split:</strong> Train ${result.train_shape[0]} samples · Test ${result.test_shape[0]} samples · ${result.train_shape[1]} features` },
  ];
  log.innerHTML = '';
  items.forEach(({ icon, type, text }) => {
    const div = document.createElement('div');
    div.className = `log-item log-${type}`;
    div.innerHTML = `<span class="log-icon">${icon}</span><div class="log-text">${text}</div>`;
    log.appendChild(div);
  });
}

function renderCorrelationHeatmap(corrData) {
  if (!corrData || !corrData.columns || corrData.columns.length === 0) {
    document.getElementById('corrHeatmap').innerHTML = '<p style="color:var(--text-3);font-size:0.85rem;padding:1rem">Not enough numeric features for correlation.</p>';
    return;
  }
  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { t: 20, r: 10, b: 60, l: 80 },
    font: { color: '#94a3b8', family: 'Inter' },
    height: 300,
  };
  Plotly.newPlot('corrHeatmap', [{
    type: 'heatmap',
    z: corrData.values,
    x: corrData.columns,
    y: corrData.columns,
    colorscale: [
      [0, '#e0e7ff'], [0.5, '#f8fafc'], [1, '#4f46e5']
    ],
    showscale: true,
    zmin: -1, zmax: 1,
  }], layout, { responsive: true, displayModeBar: false });
}

// ═══════════════════════════════════════════════════════════════════
//  Step 3 — TRAIN
// ═══════════════════════════════════════════════════════════════════

const FORMULAS = {
  linear_regression: 'θ = θ − α · (1/m) · Xᵀ(Xθ − y)  [MSE Gradient Descent]\nBias term: augment X with column of 1s',
  hard_svm: 'L = max(0, |y − (w·x + b)| − ε)  [Hinge Loss]\nSubgradient: ∂L/∂w via sign of residuals',
  soft_svm: 'min ½||w||² + C · Σξᵢ  [Slack variable C]\nHigher C → tighter margin, less tolerance',
  knn: 'd(x,y) = √Σ(xᵢ − yᵢ)²  [Euclidean Distance]\nLabel = majority vote of K nearest',
  fcm: 'u_ij = 1/Σ(d_ij/d_ik)^(2/m−1)  [Membership]\nc_j = Σuᵢⱼᵐxᵢ / Σuᵢⱼᵐ  [Centroid update]',
  pcm: 'τ_ij = 1/(1 + (d²ij/ηj)^(1/(m−1)))  [Typicality]\nη bootstrapped from FCM, outlier-robust',
};

const ALG_NAMES = {
  linear_regression: 'Linear Regression (GD)',
  hard_svm: 'Hard SVM',
  soft_svm: 'Soft SVM (Slack)',
  knn: 'K-Nearest Neighbors',
  fcm: 'Fuzzy C-Means',
  pcm: 'Possibilistic C-Means',
};

function toggleAlgo(el) {
  const alg = el.dataset.alg;
  const wasSelected = el.classList.contains('selected');
  el.classList.toggle('selected', !wasSelected);
  el.querySelector('.algo-check').textContent = wasSelected ? '□' : '✓';

  if (wasSelected) {
    state.selectedAlgorithms = state.selectedAlgorithms.filter(a => a !== alg);
  } else {
    state.selectedAlgorithms.push(alg);
  }

  // Update hyperparameter panel visibility
  const hasSoftSVM = state.selectedAlgorithms.includes('soft_svm');
  const hasKNN = state.selectedAlgorithms.includes('knn');
  const hasFuzzy = state.selectedAlgorithms.includes('fcm') || state.selectedAlgorithms.includes('pcm');
  const hasGD = state.selectedAlgorithms.some(a => ['linear_regression', 'hard_svm', 'soft_svm'].includes(a));

  document.getElementById('paramGD').classList.toggle('hidden', !hasGD);
  document.getElementById('paramC').classList.toggle('hidden', !hasSoftSVM);
  document.getElementById('paramKNN').classList.toggle('hidden', !hasKNN);
  document.getElementById('paramFuzzy').classList.toggle('hidden', !hasFuzzy);

  // Show formula for last selected
  const formulaBox = document.getElementById('formulaText');
  formulaBox.textContent = alg in FORMULAS ? FORMULAS[alg] : 'Select an algorithm to see its update rule';
}

async function runTraining() {
  if (!state.preprocessDone) { showToast('Please complete preprocessing first (Steps 1–3)', 'error'); return; }
  if (state.selectedAlgorithms.length === 0) { showToast('Please select at least one algorithm', 'error'); return; }

  // Helper to parse comma-separated numbers into an array
  const parseList = (str, fallback) => {
    if (!str || !str.trim()) return [fallback];
    const vals = str.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    return vals.length > 0 ? vals : [fallback];
  };

  const hp = {
    learning_rate: parseList(document.getElementById('paramLR').value, 0.01),
    epochs: parseList(document.getElementById('paramEpochs').value, 1000).map(Math.floor),
    tolerance: parseFloat(document.getElementById('paramTol').value) || 1e-4,
    k: parseList(document.getElementById('paramKnnK').value, 5).map(Math.floor),
    C: parseList(document.getElementById('paramCVal').value, 1.0),
    m: parseList(document.getElementById('paramM').value, 2.0),
    n_clusters: parseList(document.getElementById('paramClusters').value, 3).map(Math.floor),
  };

  const kFolds = parseInt(document.getElementById('paramK').value) || 5;

  // Progress animation
  const progressWrapper = document.getElementById('progressWrapper');
  const progressBar = document.getElementById('progressBar');
  const progressLabel = document.getElementById('progressLabel');
  const runBtn = document.getElementById('runBtn');

  progressWrapper.classList.remove('hidden');
  runBtn.classList.add('running');
  runBtn.disabled = true;

  let pct = 0;
  const progressInterval = setInterval(() => {
    pct = Math.min(pct + Math.random() * 8, 90);
    progressBar.style.width = pct + '%';
    const stages = ['Initializing…', 'Running K-Fold CV…', 'Optimizing weights…', 'Computing metrics…', 'Generating decision surface…'];
    progressLabel.textContent = stages[Math.floor((pct / 100) * stages.length)] || 'Computing…';
  }, 350);

  try {
    const result = await apiFetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        algorithms: state.selectedAlgorithms,
        k_folds: kFolds,
        hyperparams: hp,
      }),
    });

    clearInterval(progressInterval);
    progressBar.style.width = '100%';
    progressLabel.textContent = 'Done!';

    state.trainResult = result;
    setTimeout(() => {
      renderResults(result);
      goToStep(4);
      progressWrapper.classList.add('hidden');
      progressBar.style.width = '0%';
      runBtn.classList.remove('running');
      runBtn.disabled = false;
    }, 500);

    showToast('Training complete! 🎉', 'success');
  } catch (e) {
    clearInterval(progressInterval);
    progressWrapper.classList.add('hidden');
    runBtn.classList.remove('running');
    runBtn.disabled = false;
    showToast(`Training error: ${e.message}`, 'error', 8000);
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Step 4 — RESULTS
// ═══════════════════════════════════════════════════════════════════

function renderResults(data) {
  const isReg = data.task_type === 'regression';
  const metricLabel = isReg ? 'R² Score' : 'Accuracy';

  document.getElementById('taskBadge').textContent = isReg ? '📈 Regression' : '🔵 Classification';
  if (isReg) document.getElementById('surfaceTitle').textContent = 'Regression Surface (PCA Projection)';

  renderLeaderboard(data.leaderboard, data.best_algorithm, metricLabel);
  renderDecisionSurface(data.decision_surface, data.scatter_data, isReg, data.best_algorithm);
  renderCVChart(data.results, data.best_algorithm);
  renderMetricsCards(data.results, metricLabel);
  renderLossChart(data.results);
}

function renderLeaderboard(leaderboard, bestAlg, metricLabel) {
  const MEDALS = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'];
  let html = `<div style="display:grid;grid-template-columns:36px 1fr 200px 140px;gap:0.75rem;padding:0.5rem 1rem;font-size:0.75rem;color:var(--text-3);text-transform:uppercase;letter-spacing:0.05em">
    <div></div><div>Algorithm</div><div>${metricLabel}</div><div>Score Bar</div>
  </div>`;

  leaderboard.forEach((entry, i) => {
    const isWinner = entry.algorithm === bestAlg;
    const pct = Math.max(0, Math.min(100, entry.test_score * 100));
    const color = isWinner ? 'var(--green)' : i === 1 ? 'var(--accent)' : 'var(--text-3)';
    html += `
      <div class="leaderboard-row ${isWinner ? 'winner' : ''}">
        <div class="leaderboard-rank">${MEDALS[i] || (i + 1)}</div>
        <div class="leaderboard-name" style="color:${color}">
          ${ALG_NAMES[entry.algorithm] || entry.algorithm}${isWinner ? ' <span style="color:var(--green);font-size:0.8rem">★ Best</span>' : ''}
          <div style="font-size:0.7rem; color:var(--text-3); font-weight:normal; margin-top:2px;">
            Params: ${Object.entries(entry.best_hyperparameters || {}).map(([k, v]) => `${k}=${v}`).join(', ') || 'N/A'}
          </div>
        </div>
        <div class="leaderboard-score" style="color:${color}">${(entry.test_score * 100).toFixed(2)}% &nbsp;<span style="color:var(--text-3);font-size:0.78rem">CV: ${(entry.cv_mean * 100).toFixed(1)}%</span></div>
        <div class="score-bar"><div class="score-bar-fill" style="width:${Math.abs(pct)}%;background:${isWinner ? 'linear-gradient(90deg,var(--green),#10b981)' : 'linear-gradient(90deg,var(--accent),#8b5cf6)'}"></div></div>
      </div>`;
  });
  document.getElementById('leaderboardTable').innerHTML = html;
}

function renderDecisionSurface(surface, scatter, isReg, bestAlg) {
  if (!surface) {
    document.getElementById('surfacePlot').innerHTML =
      '<p style="color:var(--text-3);padding:2rem;text-align:center">Decision surface not available (single feature dataset)</p>';
    return;
  }

  const traces = [];

  if (isReg) {
    // Scatter of actual points
    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: scatter.x1,
      y: scatter.y,
      marker: { color: scatter.y, colorscale: 'Viridis', size: 6, opacity: 0.8 },
      name: 'Data Points',
    });
    // Regression surface as contour
    traces.push({
      type: 'contour',
      x: surface.x1,
      y: surface.x2,
      z: surface.z,
      colorscale: 'RdBu',
      opacity: 0.3,
      showscale: false,
      name: `${ALG_NAMES[bestAlg]} surface`,
    });
  } else {
    // Classification — heatmap background + scatter colored by class
    traces.push({
      type: 'heatmap',
      x: surface.x1,
      y: surface.x2,
      z: surface.z,
      colorscale: 'Portland',
      showscale: false,
      opacity: 0.4,
    });
    const classes = [...new Set(scatter.y)].sort();
    const palette = ['#6366f1', '#34d399', '#f87171', '#fbbf24', '#818cf8'];
    classes.forEach((cls, ci) => {
      const mask = scatter.y.map((v, i) => v === cls ? i : null).filter(i => i !== null);
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: mask.map(i => scatter.x1[i]),
        y: mask.map(i => scatter.x2[i]),
        name: `Class ${cls}`,
        marker: { color: palette[ci % palette.length], size: 7, opacity: 0.9, line: { width: 1, color: '#000' } },
      });
    });
  }

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(248,250,252,0.8)',
    margin: { t: 20, r: 20, b: 60, l: 60 },
    font: { color: '#334155', family: 'Inter', size: 11 },
    xaxis: { title: 'PC 1', gridcolor: 'rgba(0,0,0,0.05)', zeroline: false },
    yaxis: { title: 'PC 2', gridcolor: 'rgba(0,0,0,0.05)', zeroline: false },
    legend: { bgcolor: 'rgba(0,0,0,0)' },
    height: 400,
  };

  Plotly.newPlot('surfacePlot', traces, layout, { responsive: true, displayModeBar: false });
}

function renderCVChart(results, bestAlg) {
  const algs = Object.keys(results);
  const means = algs.map(a => results[a].cv_mean * 100);
  const stds = algs.map(a => results[a].cv_std * 100);
  const colors = algs.map(a => a === bestAlg ? '#34d399' : '#6366f1');

  Plotly.newPlot('cvPlot', [{
    type: 'bar',
    x: algs.map(a => ALG_NAMES[a] || a),
    y: means,
    error_y: { type: 'data', array: stds, visible: true, color: 'rgba(255,255,255,0.4)' },
    marker: { color: colors, opacity: 0.85 },
    text: means.map(m => m.toFixed(1) + '%'),
    textposition: 'auto',
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(248,250,252,0.8)',
    margin: { t: 20, r: 20, b: 80, l: 60 },
    font: { color: '#334155', family: 'Inter', size: 11 },
    yaxis: { title: 'CV Score (%)', gridcolor: 'rgba(0,0,0,0.05)', range: [0, 110] },
    xaxis: { tickangle: -20 },
    height: 400,
  }, { responsive: true, displayModeBar: false });
}

function renderMetricsCards(results, metricLabel) {
  const algs = Object.keys(results);
  let html = '<div class="metrics-grid">';
  algs.forEach(alg => {
    const r = results[alg];
    const score = (r.test_score * 100).toFixed(1);
    const cvMean = (r.cv_mean * 100).toFixed(1);
    const cvStd = (r.cv_std * 100).toFixed(1);
    html += `
      <div class="metric-card">
        <div class="metric-label">${ALG_NAMES[alg] || alg}</div>
        <div class="metric-value">${score}%</div>
        <div class="metric-sub">${metricLabel} on Test Set</div>
        <div class="metric-sub" style="margin-top:0.4rem">CV: ${cvMean}% ± ${cvStd}%</div>
        <div class="metric-sub" style="margin-top:0.4rem; font-size:0.7rem; color:var(--text-3);">
          <strong>Best Params:</strong> ${Object.entries(r.best_hyperparameters || {}).map(([k, v]) => `${k}=${v}`).join(', ')}
        </div>
      </div>`;
  });
  html += '</div>';
  document.getElementById('metricsTable').innerHTML = html;
}

function renderLossChart(results) {
  const hasLoss = Object.values(results).some(r => r.loss_history && r.loss_history.length > 0);
  if (!hasLoss) {
    document.getElementById('lossCard').classList.add('hidden');
    return;
  }
  document.getElementById('lossCard').classList.remove('hidden');
  const palette = ['#6366f1', '#34d399', '#f87171', '#fbbf24', '#818cf8'];
  const traces = Object.entries(results)
    .filter(([, r]) => r.loss_history && r.loss_history.length > 0)
    .map(([alg, r], i) => ({
      type: 'scatter',
      mode: 'lines',
      x: r.loss_history.map((_, j) => j),
      y: r.loss_history,
      name: ALG_NAMES[alg] || alg,
      line: { color: palette[i % palette.length], width: 2 },
    }));

  Plotly.newPlot('lossPlot', traces, {
    paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(13,14,30,0.5)',
    margin: { t: 20, r: 20, b: 60, l: 80 },
    font: { color: '#94a3b8', family: 'Inter', size: 11 },
    xaxis: { title: 'Epoch', gridcolor: 'rgba(255,255,255,0.05)' },
    yaxis: { title: 'Loss', gridcolor: 'rgba(255,255,255,0.05)' },
    legend: { bgcolor: 'rgba(0,0,0,0)' },
    height: 350,
  }, { responsive: true, displayModeBar: false });
}

function resetAll() {
  state.sessionId = null;
  state.columns = [];
  state.features = [];
  state.target = null;
  state.preprocessDone = false;
  state.selectedAlgorithms = [];
  state.trainResult = null;

  document.getElementById('previewCard').classList.add('hidden');
  document.getElementById('uploadFilename').textContent = '';
  document.getElementById('urlInput').value = '';
  document.getElementById('featureList').innerHTML = '';
  document.getElementById('targetList').innerHTML = '';
  document.getElementById('pipelineLog').innerHTML = '';
  document.getElementById('corrHeatmap').innerHTML = '';
  document.getElementById('surfacePlot').innerHTML = '';
  document.getElementById('cvPlot').innerHTML = '';
  document.getElementById('lossPlot').innerHTML = '';
  document.getElementById('metricsTable').innerHTML = '';
  document.getElementById('leaderboardTable').innerHTML = '';
  document.getElementById('progressWrapper').classList.add('hidden');
  document.getElementById('progressBar').style.width = '0%';

  document.querySelectorAll('.algo-card').forEach(c => {
    c.classList.remove('selected');
    c.querySelector('.algo-check').textContent = '□';
  });
  document.getElementById('paramGD').classList.remove('hidden');
  document.getElementById('paramC').classList.add('hidden');
  document.getElementById('paramKNN').classList.add('hidden');
  document.getElementById('paramFuzzy').classList.add('hidden');
  document.getElementById('formulaText').textContent = 'Select an algorithm to see its update rule';

  goToStep(0);
  showToast('Ready for a new experiment!', 'info');
}

// ═══════════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════════
(function init() {
  loadSampleList();
  // Default: show GD params
  document.getElementById('paramGD').classList.remove('hidden');
})();
