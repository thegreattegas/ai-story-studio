/* ============================================================
   AI Story Studio — Frontend App
   ============================================================ */

'use strict';

// ─── State ──────────────────────────────────────────────────
const state = {
  stories: [],
  currentJobId: null,
  eventSource: null,
  isGenerating: false,
};

// Step weights for progress bar (must sum to 100)
const STEP_WEIGHTS = {
  story_writer:   18,
  scene_director: 14,
  media:          32,
  subtitles:      12,
  compositor:     16,
  reviewer:        8,
};
const STEPS = Object.keys(STEP_WEIGHTS);

// ─── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initStars();
  loadStories();
  setupInputEnter();
});

// ─── Stars canvas ────────────────────────────────────────────
function initStars() {
  const canvas = document.getElementById('starCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const STARS = Array.from({ length: 140 }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: Math.random() * 1.4 + 0.3,
    o: Math.random() * 0.6 + 0.1,
    speed: Math.random() * 0.004 + 0.001,
    phase: Math.random() * Math.PI * 2,
  }));

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const t = performance.now() / 1000;
    for (const s of STARS) {
      const alpha = s.o * (0.5 + 0.5 * Math.sin(t * s.speed * 30 + s.phase));
      ctx.beginPath();
      ctx.arc(s.x * canvas.width, s.y * canvas.height, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(196,181,253,${alpha.toFixed(3)})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  draw();
}

// ─── Load stories from API ───────────────────────────────────
async function loadStories() {
  try {
    const res  = await fetch('/api/stories');
    const data = await res.json();
    state.stories = data.stories || [];
    renderStories();
  } catch (err) {
    console.error('Failed to load stories:', err);
    renderStories();
  }
}

// ─── Render story grid ───────────────────────────────────────
function renderStories() {
  const grid    = document.getElementById('storiesGrid');
  const empty   = document.getElementById('emptyState');
  const section = document.getElementById('storiesSection');
  const count   = document.getElementById('sectionCount');
  const navCount = document.getElementById('navCount');

  if (state.stories.length === 0) {
    section.style.display = 'none';
    empty.style.display   = 'block';
    navCount.textContent  = '';
    return;
  }

  section.style.display = 'block';
  empty.style.display   = 'none';
  count.textContent     = `${state.stories.length} ${state.stories.length === 1 ? 'story' : 'stories'}`;
  navCount.textContent  = `${state.stories.length} stories`;

  grid.innerHTML = state.stories.map((story, i) => buildCard(story, i)).join('');
}

function buildCard(story, index) {
  const duration = story.duration_sec
    ? `${Math.round(story.duration_sec)}s`
    : '';
  const date = story.created_at
    ? new Date(story.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    : '';

  const thumb = story.thumbnail || '';
  const imgHtml = thumb
    ? `<img class="story-card-img" src="${esc(thumb)}" alt="${esc(story.title)}" loading="lazy" />`
    : `<div class="story-card-img" style="background:linear-gradient(135deg,#1c0a4a,#2d0060);"></div>`;

  // Use data-index to avoid putting JSON inside an HTML attribute (breaks on quotes)
  return `
    <div class="story-card" data-index="${index}" onclick="playStoryByIndex(${index})">
      ${imgHtml}
      <div class="story-card-overlay"></div>
      <div class="story-card-play">▶</div>
      <div class="story-card-footer">
        <div class="story-card-title">${esc(story.title)}</div>
        <div class="story-card-meta">
          ${duration ? `<span class="story-card-badge">${esc(duration)}</span>` : ''}
          ${date     ? `<span class="story-card-badge">${esc(date)}</span>`     : ''}
        </div>
      </div>
    </div>`;
}

function playStoryByIndex(index) {
  const story = state.stories[index];
  if (story) openVideoByData(story);
}

// ─── Generation ──────────────────────────────────────────────
function startGeneration() {
  const input  = document.getElementById('promptInput');
  const prompt = input.value.trim();

  if (!prompt) {
    input.focus();
    input.style.border = '1.5px solid rgba(236,72,153,0.6)';
    setTimeout(() => { input.style.border = ''; }, 1500);
    return;
  }

  if (state.isGenerating) return;

  doGenerate(prompt);
}

async function doGenerate(prompt) {
  state.isGenerating = true;
  setGenBarBusy(true);
  openProgressModal(prompt);
  resetSteps();

  try {
    const res  = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prompt }),
    });

    if (res.status === 409) {
      const err = await res.json();
      alert(err.detail || 'Already generating. Please wait.');
      cancelGeneration();
      return;
    }
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start generation');
    }

    const { job_id } = await res.json();
    state.currentJobId = job_id;
    listenToJob(job_id);

  } catch (err) {
    console.error(err);
    showProgressError(err.message);
  }
}

// ─── SSE streaming ───────────────────────────────────────────
function listenToJob(jobId) {
  if (state.eventSource) {
    state.eventSource.close();
  }

  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state.eventSource = es;

  es.onmessage = (e) => {
    try {
      handleProgressEvent(JSON.parse(e.data));
    } catch (err) {
      console.warn('SSE parse error:', err);
    }
  };

  es.onerror = () => {
    if (state.isGenerating) {
      showProgressError('Connection lost. The generation may still be running — refresh to check.');
    }
    es.close();
  };
}

function handleProgressEvent(event) {
  if (event.type === 'ping') return;

  if (event.type === 'title') {
    document.getElementById('progressTitle').textContent = `✦ "${event.title}"`;
    return;
  }

  if (event.type === 'step') {
    updateStep(event.step, event.status, event.label);
    updateProgressBar();
    return;
  }

  if (event.type === 'complete') {
    updateProgressBar(100);
    document.getElementById('progressTitle').textContent = `✦ "${event.title}" is ready!`;
    state.eventSource && state.eventSource.close();

    // Brief delay so user sees 100% before closing
    setTimeout(() => {
      closeProgressModal();
      loadStories();
      // Auto-play the new story
      if (event.video) {
        openVideoByData({
          title:       event.title,
          summary:     event.summary || '',
          video:       event.video,
          thumbnail:   event.thumbnail,
          duration_sec: event.duration_sec,
        });
      }
    }, 1200);
    return;
  }

  if (event.type === 'error') {
    showProgressError(event.message);
    return;
  }
}

// ─── Progress step UI ────────────────────────────────────────
function resetSteps() {
  for (const step of STEPS) {
    const el = document.getElementById(`step-${step}`);
    if (!el) continue;
    el.className = 'step';
    const statusEl = el.querySelector('.step-status');
    statusEl.className = 'step-status step-pending';
    statusEl.innerHTML = '<span class="step-dot"></span>';
  }
  updateProgressBar(0);
  document.getElementById('progressTitle').textContent = 'Crafting your story…';
}

function updateStep(stepId, status, label) {
  const el = document.getElementById(`step-${stepId}`);
  if (!el) return;

  el.className = `step is-${status}`;

  const detailEl = document.getElementById(`detail-${stepId}`);
  if (detailEl && label) {
    detailEl.textContent = label;
  }

  const statusEl = el.querySelector('.step-status');
  statusEl.className = 'step-status';
  statusEl.innerHTML = '';
}

function updateProgressBar(forcePct) {
  let pct = forcePct;
  if (pct === undefined) {
    let done = 0;
    for (const step of STEPS) {
      const el = document.getElementById(`step-${step}`);
      if (el && el.classList.contains('is-done')) {
        done += STEP_WEIGHTS[step];
      } else if (el && el.classList.contains('is-running')) {
        done += STEP_WEIGHTS[step] * 0.5;
      }
    }
    pct = Math.min(Math.round(done), 99);
  }

  document.getElementById('progressBarFill').style.width = `${pct}%`;
  document.getElementById('progressPct').textContent     = `${pct}%`;
}

function showProgressError(message) {
  document.getElementById('progressTitle').textContent = '⚠ Generation failed';
  const bar = document.getElementById('progressBarFill');
  bar.style.background = 'linear-gradient(90deg, #dc2626, #ef4444)';
  bar.style.width = '100%';
  document.getElementById('progressPct').textContent = 'Error';

  const stepsEl = document.getElementById('progressSteps');
  const errDiv  = document.createElement('div');
  errDiv.style.cssText = 'margin-top:16px;padding:12px 16px;background:rgba(220,38,38,0.1);border:1px solid rgba(220,38,38,0.3);border-radius:10px;font-size:13px;color:#fca5a5;';
  errDiv.textContent = message || 'An unexpected error occurred.';
  stepsEl.appendChild(errDiv);

  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'Close';
  closeBtn.style.cssText = 'margin-top:16px;width:100%;padding:10px;background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.3);border-radius:8px;color:#c4b5fd;cursor:pointer;font-family:inherit;font-size:13px;';
  closeBtn.onclick = closeProgressModal;
  stepsEl.appendChild(closeBtn);

  cancelGeneration();
}

function cancelGeneration() {
  state.isGenerating = false;
  state.currentJobId = null;
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  setGenBarBusy(false);
}

// ─── Gen bar helpers ─────────────────────────────────────────
function setGenBarBusy(busy) {
  const input  = document.getElementById('promptInput');
  const btn    = document.getElementById('generateBtn');
  const text   = btn.querySelector('.gen-btn-text');
  const spinner = btn.querySelector('.gen-btn-spinner');

  input.disabled  = busy;
  btn.disabled    = busy;
  text.style.display   = busy ? 'none'  : '';
  spinner.style.display = busy ? 'flex' : 'none';
}

// ─── Progress modal open/close ───────────────────────────────
function openProgressModal(prompt) {
  document.getElementById('progressPrompt').textContent = `"${prompt}"`;
  document.getElementById('progressBackdrop').style.display = 'flex';
}

function closeProgressModal() {
  document.getElementById('progressBackdrop').style.display = 'none';
  cancelGeneration();
  // Re-enable input and clear it on success
  document.getElementById('promptInput').value = '';
}

// ─── Video modal ─────────────────────────────────────────────
function openVideoByData(story) {
  const player   = document.getElementById('videoPlayer');
  const errorBox = document.getElementById('videoError');
  const errorMsg = document.getElementById('videoErrorMsg');

  // Reset error state
  errorBox.style.display = 'none';
  player.style.display   = 'block';

  // Populate info
  document.getElementById('videoTitle').textContent   = story.title || '';
  document.getElementById('videoSummary').textContent = story.summary || '';
  document.getElementById('videoDuration').textContent = story.duration_sec
    ? `${Math.round(story.duration_sec)}s` : '';
  document.getElementById('videoDate').textContent = story.created_at
    ? new Date(story.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
    : '';

  // Set source and show modal BEFORE loading so layout is ready
  player.src = story.video || '';
  document.getElementById('videoBackdrop').style.display = 'flex';

  // Show the video path in console for debugging
  console.log('Playing video:', player.src);

  // Handle loading errors
  player.onerror = function () {
    const code = player.error ? player.error.code : '?';
    const msg  = player.error ? player.error.message : 'Unknown error';
    console.error('Video load error — code:', code, 'msg:', msg, 'src:', player.src);
    player.style.display   = 'none';
    errorMsg.textContent   = `Could not load video (error ${code}). Check server is running.`;
    errorBox.style.display = 'flex';
  };

  player.load();
  // No autoplay — browser blocks it for audio videos.
  // Native controls let the user press play.
}

function closeVideo(event) {
  // Close on backdrop click but not on modal itself
  if (event && event.target !== document.getElementById('videoBackdrop')) return;
  _closeVideoModal();
}

function _closeVideoModal() {
  const player = document.getElementById('videoPlayer');
  player.pause();
  player.src = '';
  document.getElementById('videoBackdrop').style.display = 'none';
}

// Close on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (document.getElementById('videoBackdrop').style.display !== 'none') {
      _closeVideoModal();
    }
  }
});

// ─── Helpers ────────────────────────────────────────────────
function focusPrompt() {
  document.getElementById('promptInput').focus();
  document.getElementById('genBar').scrollIntoView({ behavior: 'smooth' });
}

function useExample(el) {
  document.getElementById('promptInput').value = el.textContent;
  focusPrompt();
}

function setupInputEnter() {
  document.getElementById('promptInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      startGeneration();
    }
  });
}

// ─── Improve modal ──────────────────────────────────────────
const IMPROVE_STEPS = ['qa', 'design', 'frontend', 'backend', 'review'];
const IMPROVE_WEIGHTS = { qa: 10, design: 25, frontend: 30, backend: 25, review: 10 };

function openImproveModal() {
  // Reset to control view
  document.getElementById('improveControls').style.display = 'flex';
  document.getElementById('improveSteps').style.display    = 'none';
  document.getElementById('improveBarWrap').style.display  = 'none';
  document.getElementById('improveReviewOut').style.display = 'none';
  document.getElementById('improveRunBtn').disabled = false;
  document.getElementById('improveTitle').textContent = 'Improve Website';
  resetImproveSteps();
  document.getElementById('improveBackdrop').style.display = 'flex';
}

function closeImproveModal() {
  document.getElementById('improveBackdrop').style.display = 'none';
}

function resetImproveSteps() {
  for (const s of IMPROVE_STEPS) {
    const el = document.getElementById(`istep-${s}`);
    if (!el) continue;
    el.className = 'step';
    const st = el.querySelector('.step-status');
    st.className = 'step-status step-pending';
    st.innerHTML = '<span class="step-dot"></span>';
  }
  const bar = document.getElementById('improveBarFill');
  if (bar) { bar.style.width = '0%'; bar.style.background = ''; }
  const pct = document.getElementById('improvePct');
  if (pct) pct.textContent = '0%';
}

async function runImprove() {
  const target      = document.getElementById('improveTarget').value;
  const instruction = document.getElementById('improveInstruction').value.trim();

  document.getElementById('improveControls').style.display = 'none';
  document.getElementById('improveSteps').style.display    = 'flex';
  document.getElementById('improveBarWrap').style.display  = 'flex';
  document.getElementById('improveRunBtn').disabled = true;
  document.getElementById('improveTitle').textContent = 'Agents working…';
  resetImproveSteps();

  try {
    const res = await fetch('/api/improve', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ target, instruction }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start improve job');
    }
    const { job_id } = await res.json();
    listenToImproveJob(job_id);
  } catch (err) {
    document.getElementById('improveTitle').textContent = '⚠ Error: ' + err.message;
  }
}

function listenToImproveJob(jobId) {
  const es = new EventSource(`/api/improve/${jobId}/stream`);

  es.onmessage = (e) => {
    try { handleImproveEvent(JSON.parse(e.data)); }
    catch (err) { console.warn('improve SSE parse error:', err); }
  };
  es.onerror = () => {
    document.getElementById('improveTitle').textContent = '⚠ Connection lost';
    es.close();
  };
}

function handleImproveEvent(event) {
  if (event.type === 'ping') return;

  if (event.type === 'web_step') {
    updateImproveStep(event.step, event.status, event.label);
    updateImproveBar();
    return;
  }

  if (event.type === 'web_complete') {
    updateImproveBar(100);
    const modified = event.files_modified || [];
    document.getElementById('improveTitle').textContent =
      modified.length ? `Done — ${modified.join(', ')} updated` : 'Done — no files changed';

    if (event.review_notes) {
      document.getElementById('improveReviewOut').style.display = 'block';
      document.getElementById('improveReviewText').textContent = event.review_notes;
    }

    // If CSS was changed, reload styles without full page reload
    if (modified.includes('style.css')) {
      const links = document.querySelectorAll('link[rel="stylesheet"]');
      links.forEach(l => { l.href = l.href.replace(/\?.*$/, '') + '?v=' + Date.now(); });
    }
    return;
  }

  if (event.type === 'error') {
    document.getElementById('improveTitle').textContent = '⚠ ' + event.message;
  }
}

function updateImproveStep(stepId, status, label) {
  const el = document.getElementById(`istep-${stepId}`);
  if (!el) return;
  el.className = `step is-${status}`;
  const d = document.getElementById(`idetail-${stepId}`);
  if (d && label) d.textContent = label;
  const st = el.querySelector('.step-status');
  st.className = 'step-status';
  st.innerHTML = '';
}

function updateImproveBar(forcePct) {
  let pct = forcePct;
  if (pct === undefined) {
    let done = 0;
    for (const s of IMPROVE_STEPS) {
      const el = document.getElementById(`istep-${s}`);
      if (!el) continue;
      if (el.classList.contains('is-done'))    done += IMPROVE_WEIGHTS[s];
      else if (el.classList.contains('is-running')) done += IMPROVE_WEIGHTS[s] * 0.5;
    }
    pct = Math.min(Math.round(done), 99);
  }
  const bar = document.getElementById('improveBarFill');
  const txt = document.getElementById('improvePct');
  if (bar) bar.style.width = `${pct}%`;
  if (txt) txt.textContent = `${pct}%`;
}

// Close improve modal on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (document.getElementById('improveBackdrop').style.display !== 'none') {
      closeImproveModal();
    }
  }
});

// ─── Helpers ────────────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
