// NeuroGuard ML — frontend logic.
// Features: live analysis, example chips, session history, confidence bars,
// keyboard shortcut (Ctrl+Enter), share via URL, dark/light mode toggle.

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;
const MAX_HISTORY = 8;

let sessionHistory = [];
let lastResult = null;

function $(id) { return document.getElementById(id); }

// ---------------- Theme ----------------

const THEME_KEY = 'neuroguard-theme';

function applyTheme(theme) {
  document.body.classList.toggle('light', theme === 'light');
  const btn = $('theme-toggle');
  if (btn) btn.textContent = theme === 'light' ? '●' : '◐';
  // Re-render chart so axis/gridline colors update for the new theme
  renderPRChart();
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(saved);
  const btn = $('theme-toggle');
  if (btn) {
    btn.addEventListener('click', () => {
      const next = document.body.classList.contains('light') ? 'dark' : 'light';
      localStorage.setItem(THEME_KEY, next);
      applyTheme(next);
    });
  }
}

initTheme();

// ---------------- Keyboard shortcut ----------------

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    const form = $('analyze-form');
    if (form) form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  }
});

// ---------------- Chips ----------------

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    const text = chip.dataset.text;
    const input = $('text-input');
    if (!input) return;
    input.value = text;
    input.focus();
    $('analyze-form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
});

// ---------------- Analyze flow ----------------

const form = $('analyze-form');
const textInput = $('text-input');
const runBtn = $('run-btn');
const runHint = $('run-hint');
const errorBox = $('error-box');
const resultBlock = $('result-block');

// Auto-run from URL ?q= param
(function checkUrlParam() {
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  if (q && textInput) {
    textInput.value = q;
    setTimeout(() => {
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    }, 400);
  }
})();

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text) { showError('Please enter some text to analyze.'); return; }
  setLoading(true);
  hideError();
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Something went wrong.'); setLoading(false); return; }
    lastResult = data;
    renderResult(data);
    addToHistory(data);
    updateShareUrl(text);
  } catch (err) {
    showError('Could not reach the server. Check your connection.');
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  runBtn.disabled = isLoading;
  runBtn.textContent = isLoading ? 'ANALYZING...' : 'RUN ANALYSIS';
  runHint.textContent = isLoading ? 'running on cpu...' : '~1-3s on cpu · ctrl+enter';
  const status = $('sys-status');
  if (status) {
    status.textContent = isLoading ? 'BUSY' : 'READY';
    status.className = isLoading ? 'status-warn' : 'status-ok';
  }
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function hideError() { errorBox.hidden = true; }

// ---------------- Render result ----------------

function renderResult(data) {
  resultBlock.hidden = false;

  const banner = $('verdict-banner');
  const dot = $('verdict-dot');
  const text = $('verdict-text');
  const sub = $('verdict-sub');

  if (data.is_failure_risk) {
    banner.className = 'verdict-banner warn';
    dot.className = 'dot warn';
    text.className = 'verdict-text warn';
  } else {
    banner.className = 'verdict-banner ok';
    dot.className = 'dot ok';
    text.className = 'verdict-text';
  }
  text.textContent = data.warning;
  sub.textContent = `failure probability ${data.failure_probability.toFixed(3)} · threshold ${data.failure_threshold.toFixed(3)}`;

  // Gauge
  const gaugeFill = $('gauge-fill');
  const gaugeValue = $('gauge-value');
  const pct = Math.max(0, Math.min(1, data.failure_probability));
  const offset = GAUGE_CIRCUMFERENCE * (1 - pct);
  gaugeFill.style.transition = 'none';
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  void gaugeFill.offsetWidth;
  gaugeFill.style.transition = '';
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = data.is_failure_risk ? '#fbbf24' : '#4ade80';
  gaugeValue.textContent = `${Math.round(pct * 100)}%`;

  // Evidence rows + confidence bars
  const bertConf = data.bert_confidence;
  const vaderConf = Math.min(1, Math.abs(data.vader_score));
  const lrConf = data.lr_confidence;
  const entropy = data.bert_entropy;

  $('ev-bert').textContent = `${data.bert_label} (${(bertConf * 100).toFixed(1)}%)`;
  $('ev-bert').className = 'v ' + (data.bert_label === 'positive' ? 'pos' : 'neg');
  setBar('bar-bert', bertConf, data.bert_label === 'positive');

  $('ev-vader').textContent = data.vader_agrees ? `agrees · ${data.vader_label}` : `disagrees · ${data.vader_label}`;
  $('ev-vader').className = 'v ' + (data.vader_label === 'positive' ? 'pos' : 'neg');
  setBar('bar-vader', vaderConf, data.vader_label === 'positive');

  $('ev-lr').textContent = data.lr_agrees ? `agrees · ${data.lr_label}` : `disagrees · ${data.lr_label}`;
  $('ev-lr').className = 'v ' + (data.lr_label === 'positive' ? 'pos' : 'neg');
  setBar('bar-lr', lrConf, data.lr_label === 'positive');

  $('ev-entropy').textContent = entropy.toFixed(3);
  $('ev-entropy').className = 'v';
  const entropyBar = $('bar-entropy');
  if (entropyBar) {
    entropyBar.style.width = `${Math.round(entropy * 100)}%`;
  }

  resultBlock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setBar(id, value, isPositive) {
  const bar = $(id);
  if (!bar) return;
  bar.style.width = `${Math.round(value * 100)}%`;
  bar.style.background = isPositive ? 'var(--green)' : 'var(--red)';
}

// ---------------- Share URL ----------------

function updateShareUrl(text) {
  const url = new URL(window.location.href);
  url.searchParams.set('q', text);
  window.history.replaceState({}, '', url.toString());
}

// ---------------- Session history ----------------

function addToHistory(data) {
  sessionHistory.unshift(data);
  if (sessionHistory.length > MAX_HISTORY) sessionHistory = sessionHistory.slice(0, MAX_HISTORY);
  renderHistory();
}

function renderHistory() {
  const panel = $('history-panel');
  const list = $('history-list');
  if (!panel || !list) return;

  panel.hidden = sessionHistory.length === 0;
  list.innerHTML = '';

  sessionHistory.forEach((entry, i) => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <div class="history-dot ${entry.is_failure_risk ? 'warn' : 'ok'}"></div>
      <span class="history-text">${escapeHtml(entry.text)}</span>
      <span class="history-prob">${(entry.failure_probability * 100).toFixed(0)}%</span>
    `;
    item.addEventListener('click', () => {
      textInput.value = entry.text;
      renderResult(entry);
      resultBlock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
    list.appendChild(item);
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ---------------- PR chart ----------------

function renderPRChart() {
  const dataEl = $('metrics-data');
  const chartEl = $('pr-chart');
  if (!dataEl || !chartEl) return;

  let metrics;
  try { metrics = JSON.parse(dataEl.textContent); } catch (e) { return; }
  if (!metrics || !metrics.pr_curve || metrics.pr_curve.length === 0) return;

  const width = 400;
  const height = 200;
  const padding = { top: 10, right: 16, bottom: 32, left: 48 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  function xPos(recall) { return padding.left + recall * plotW; }
  function yPos(precision) { return padding.top + (1 - precision) * plotH; }

  const sorted = [...metrics.pr_curve].sort((a, b) => a.recall - b.recall);
  const pathD = sorted
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xPos(p.recall).toFixed(1)} ${yPos(p.precision).toFixed(1)}`)
    .join(' ');

  // Read current theme colors so chart looks right in both light and dark mode
  const rootStyle = getComputedStyle(document.documentElement);
  const colorBorder = getComputedStyle(document.body).getPropertyValue('--border').trim() || '#1a1a1a';
  const colorMuted = getComputedStyle(document.body).getPropertyValue('--text-muted').trim() || '#6b6b6b';
  const colorFaint = getComputedStyle(document.body).getPropertyValue('--text-faint').trim() || '#4a4a4a';
  const colorBg = getComputedStyle(document.body).getPropertyValue('--bg').trim() || '#0a0a0a';

  let svg = '';

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = yPos(t);
    svg += `<line x1="${padding.left}" y1="${y.toFixed(1)}" x2="${width - padding.right}" y2="${y.toFixed(1)}" stroke="${colorBorder}" stroke-width="1"/>`;
    svg += `<text x="${padding.left - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-size="9" fill="${colorMuted}" font-family="JetBrains Mono,monospace">${t}</text>`;
  });

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const x = xPos(t);
    svg += `<text x="${x.toFixed(1)}" y="${height - padding.bottom + 14}" text-anchor="middle" font-size="9" fill="${colorMuted}" font-family="JetBrains Mono,monospace">${t}</text>`;
  });

  svg += `<text x="${padding.left + plotW / 2}" y="${height - 2}" text-anchor="middle" font-size="9" fill="${colorFaint}" font-family="JetBrains Mono,monospace">recall</text>`;
  svg += `<text x="10" y="${padding.top + plotH / 2}" text-anchor="middle" font-size="9" fill="${colorFaint}" font-family="JetBrains Mono,monospace" transform="rotate(-90 10 ${padding.top + plotH / 2})">precision</text>`;

  svg += `<path d="${pathD}" fill="none" stroke="#4ade80" stroke-width="0.75"/>`;

  let closest = sorted[0];
  let closestDist = Infinity;
  for (const p of sorted) {
    const d = Math.abs(p.recall - metrics.recall) + Math.abs(p.precision - metrics.precision);
    if (d < closestDist) { closestDist = d; closest = p; }
  }
  const opX = xPos(closest.recall);
  const opY = yPos(closest.precision);

  svg += `<circle cx="${opX.toFixed(1)}" cy="${opY.toFixed(1)}" r="2.5" fill="#fbbf24"/>`;

  // Position label between the 0.75 and 1.0 gridlines (y midpoint = 0.875)
  // so it never overlaps a gridline regardless of where the threshold dot falls
  const legendX = width - padding.right - 2;
  const legendY = yPos(0.875);
  const labelText = `threshold ${metrics.threshold.toFixed(3)}`;
  const labelW = labelText.length * 4.6 + 10;
  svg += `<rect x="${(legendX - labelW - 8).toFixed(1)}" y="${(legendY - 6).toFixed(1)}" width="${(labelW + 4).toFixed(1)}" height="13" fill="${colorBg}" rx="1"/>`;
  svg += `<circle cx="${(legendX - labelW - 4).toFixed(1)}" cy="${legendY}" r="2.5" fill="#fbbf24"/>`;
  svg += `<text x="${(legendX - labelW).toFixed(1)}" y="${(legendY + 3.5).toFixed(1)}" text-anchor="start" font-size="7.5" fill="#fbbf24" font-family="JetBrains Mono,monospace">${labelText}</text>`;

  chartEl.innerHTML = svg;
}

renderPRChart();

// ---------------- Tab navigation ----------------

const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels = {
  analyze: $('tab-analyze'),
  diagnostics: $('tab-diagnostics'),
  ping: $('tab-ping')
};

tabButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    tabButtons.forEach((b) => {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
    });
    Object.entries(tabPanels).forEach(([name, panel]) => {
      if (panel) panel.hidden = name !== target;
    });
  });
});

// ---------------- Feedback form ----------------

const feedbackForm = $('feedback-form');
if (feedbackForm) {
  const feedbackEmail = $('feedback-email');
  const feedbackMessage = $('feedback-message');
  const feedbackBtn = $('feedback-btn');
  const feedbackErrorBox = $('feedback-error-box');
  const feedbackSuccessBox = $('feedback-success-box');

  feedbackForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = feedbackEmail.value.trim();
    const message = feedbackMessage.value.trim();
    feedbackErrorBox.hidden = true;
    feedbackSuccessBox.hidden = true;
    if (!email) { showFeedbackError('Please enter your email.'); return; }
    if (!message) { showFeedbackError('Please enter a message.'); return; }
    feedbackBtn.disabled = true;
    feedbackBtn.textContent = 'SENDING...';
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, message })
      });
      const data = await res.json();
      if (!res.ok) { showFeedbackError(data.error || 'Something went wrong.'); return; }
      feedbackSuccessBox.hidden = false;
      feedbackForm.reset();
    } catch (err) {
      showFeedbackError('Could not reach the server. Try again.');
    } finally {
      feedbackBtn.disabled = false;
      feedbackBtn.textContent = 'SEND';
    }
  });

  function showFeedbackError(msg) {
    feedbackErrorBox.textContent = msg;
    feedbackErrorBox.hidden = false;
  }
}
