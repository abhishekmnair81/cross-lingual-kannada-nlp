/**
 * IndicNER Web Application - Frontend Controller
 * Modern interactive client for Dravidian Multilingual NER (Kannada & Telugu)
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const inputText = document.getElementById('inputText');
  const wordCharCount = document.getElementById('wordCharCount');
  const scriptBadge = document.getElementById('scriptBadge');
  const confSlider = document.getElementById('confSlider');
  const sliderValDisplay = document.getElementById('sliderValDisplay');
  const checkPER = document.getElementById('checkPER');
  const checkLOC = document.getElementById('checkLOC');
  const checkORG = document.getElementById('checkORG');
  const extractBtn = document.getElementById('extractBtn');
  const clearBtn = document.getElementById('clearBtn');
  const copyBtn = document.getElementById('copyBtn');
  const toast = document.getElementById('toast');

  // KPI Elements
  const kpiTotalEntities = document.getElementById('kpiTotalEntities');
  const kpiPER = document.getElementById('kpiPER');
  const kpiLOC = document.getElementById('kpiLOC');
  const kpiORG = document.getElementById('kpiORG');

  // Results & Subtabs Elements
  const highlightContainer = document.getElementById('highlightContainer');
  const tokenTableBody = document.getElementById('tokenTableBody');
  const subwordTableBody = document.getElementById('subwordTableBody');
  const jsonPayload = document.getElementById('jsonPayload');
  const subtabBtns = document.querySelectorAll('.subtab-btn');
  const exampleChips = document.querySelectorAll('.example-chip');

  // State cache for live re-filtering
  let lastApiResponse = null;

  // -------------------------------------------------------------
  // 1. Text Analysis & Script Detection
  // -------------------------------------------------------------
  function detectScript(text) {
    if (!text || !text.trim()) {
      return { script: 'Ready', className: 'script-idle' };
    }
    let knCount = 0;
    let teCount = 0;
    for (const char of text) {
      const code = char.charCodeAt(0);
      if (code >= 0x0C80 && code <= 0x0CFF) knCount++;
      else if (code >= 0x0C00 && code <= 0x0C7F) teCount++;
    }

    if (knCount > 0 && knCount >= teCount) {
      return { script: 'Kannada (In-Language)', className: 'script-kn' };
    } else if (teCount > 0) {
      return { script: 'Telugu (Zero-Shot Cross-Lingual)', className: 'script-te' };
    } else {
      return { script: 'Latin / Other Script', className: 'script-idle' };
    }
  }

  function updateTextStats() {
    const text = inputText.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).filter(Boolean).length : 0;
    wordCharCount.textContent = `${words} ${words === 1 ? 'word' : 'words'} • ${chars} ${chars === 1 ? 'character' : 'characters'}`;

    const { script, className } = detectScript(text);
    scriptBadge.textContent = script;
    scriptBadge.className = `script-tag ${className}`;
  }

  inputText.addEventListener('input', updateTextStats);

  // -------------------------------------------------------------
  // 2. Slider & Controls
  // -------------------------------------------------------------
  confSlider.addEventListener('input', () => {
    const pct = Math.round(parseFloat(confSlider.value) * 100);
    sliderValDisplay.textContent = `${pct}%`;
    if (lastApiResponse) {
      renderResults(lastApiResponse);
    }
  });

  function getSelectedEntityTypes() {
    const types = [];
    if (checkPER.checked) types.push('PER');
    if (checkLOC.checked) types.push('LOC');
    if (checkORG.checked) types.push('ORG');
    return types;
  }

  [checkPER, checkLOC, checkORG].forEach((cb) => {
    cb.addEventListener('change', () => {
      if (lastApiResponse) {
        renderResults(lastApiResponse);
      }
    });
  });

  // -------------------------------------------------------------
  // 3. Example Chips
  // -------------------------------------------------------------
  exampleChips.forEach((chip) => {
    chip.addEventListener('click', (e) => {
      e.preventDefault();
      const sampleText = chip.getAttribute('data-text');
      if (sampleText) {
        inputText.value = sampleText;
        updateTextStats();
        // Trigger extraction automatically for instant feedback
        runExtraction();
      }
    });
  });

  // -------------------------------------------------------------
  // 4. Subtab Switching
  // -------------------------------------------------------------
  subtabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      subtabBtns.forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.subtab-content').forEach((c) => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });

  // -------------------------------------------------------------
  // 5. Toast Notifications
  // -------------------------------------------------------------
  function showToast(message) {
    toast.textContent = message;
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 2200);
  }

  copyBtn.addEventListener('click', () => {
    if (!lastApiResponse) {
      showToast('No results to copy yet!');
      return;
    }
    const textToCopy = JSON.stringify(lastApiResponse, null, 2);
    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast('Results JSON copied to clipboard!');
    }).catch(() => {
      showToast('Failed to copy to clipboard');
    });
  });

  // -------------------------------------------------------------
  // 6. Clear Button
  // -------------------------------------------------------------
  clearBtn.addEventListener('click', () => {
    inputText.value = '';
    updateTextStats();
    lastApiResponse = null;

    // Reset KPIs
    kpiTotalEntities.textContent = '0';
    kpiPER.textContent = '0';
    kpiLOC.textContent = '0';
    kpiORG.textContent = '0';

    // Reset highlights
    highlightContainer.innerHTML = `
      <div class="highlight-placeholder">
        Enter or select a sentence above and click "Extract Named Entities"
      </div>
    `;

    // Reset tables
    tokenTableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-light); padding: 24px;">No tokens analyzed yet</td>
      </tr>
    `;

    subwordTableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-light); padding: 24px;">No subwords analyzed yet</td>
      </tr>
    `;

    // Reset JSON
    jsonPayload.textContent = '{}';
  });

  // -------------------------------------------------------------
  // 7. API Call & Prediction Execution
  // -------------------------------------------------------------
  async function runExtraction() {
    const text = inputText.value.trim();
    if (!text) {
      showToast('Please enter or select a sentence first.');
      inputText.focus();
      return;
    }

    const minConf = parseFloat(confSlider.value);
    const selectedTypes = getSelectedEntityTypes();

    // Set UI to loading state
    extractBtn.disabled = true;
    const originalBtnHtml = extractBtn.innerHTML;
    extractBtn.innerHTML = `
      <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="10" stroke-width="4" stroke-opacity="0.25"></circle>
        <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <span>Extracting Entities...</span>
    `;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          min_confidence: minConf,
          selected_types: selectedTypes,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      lastApiResponse = data;
      renderResults(data);
    } catch (err) {
      console.error('Inference error:', err);
      showToast(`Error: ${err.message}`);
    } finally {
      extractBtn.disabled = false;
      extractBtn.innerHTML = originalBtnHtml;
    }
  }

  extractBtn.addEventListener('click', runExtraction);

  // -------------------------------------------------------------
  // 8. Results Rendering Pipeline
  // -------------------------------------------------------------
  function renderResults(data) {
    if (!data || !data.tokens) return;

    const minConf = parseFloat(confSlider.value);
    const selectedTypes = getSelectedEntityTypes();

    // Map each token's active status based on current filters
    const tokens = data.tokens;
    const tags = data.tags || [];
    const confidences = data.confidences || [];

    // Filter entities based on active threshold and types
    let totalEnt = 0;
    let perCount = 0;
    let locCount = 0;
    let orgCount = 0;

    // Process entity spans for highlighting
    // Construct merged entities list from tokens/tags
    const spans = [];
    let currentSpan = null;

    tokens.forEach((token, idx) => {
      const tag = tags[idx] || 'O';
      const conf = confidences[idx] !== undefined ? confidences[idx] : 1.0;
      const isEntity = tag !== 'O';
      const entType = isEntity ? tag.split('-')[1] : null;
      const meetsConf = conf >= minConf;
      const meetsType = entType ? selectedTypes.includes(entType) : false;
      const isActive = isEntity && meetsConf && meetsType;

      if (isActive && tag.startsWith('B-')) {
        if (currentSpan) spans.push(currentSpan);
        currentSpan = {
          type: entType,
          tokens: [token],
          confs: [conf],
          start: idx,
          end: idx + 1,
        };
      } else if (isActive && tag.startsWith('I-') && currentSpan && currentSpan.type === entType) {
        currentSpan.tokens.push(token);
        currentSpan.confs.push(conf);
        currentSpan.end = idx + 1;
      } else if (isActive && tag.startsWith('I-') && (!currentSpan || currentSpan.type !== entType)) {
        if (currentSpan) spans.push(currentSpan);
        currentSpan = {
          type: entType,
          tokens: [token],
          confs: [conf],
          start: idx,
          end: idx + 1,
        };
      } else {
        if (currentSpan) {
          spans.push(currentSpan);
          currentSpan = null;
        }
      }
    });
    if (currentSpan) spans.push(currentSpan);

    // Compute KPI counts from active spans
    spans.forEach((s) => {
      totalEnt++;
      if (s.type === 'PER') perCount++;
      else if (s.type === 'LOC') locCount++;
      else if (s.type === 'ORG') orgCount++;
    });

    kpiTotalEntities.textContent = totalEnt;
    kpiPER.textContent = perCount;
    kpiLOC.textContent = locCount;
    kpiORG.textContent = orgCount;

    // Render Highlights
    if (spans.length === 0 && tokens.length > 0) {
      // Just plain tokens
      highlightContainer.innerHTML = `<div>${tokens.map(escapeHtml).join(' ')}</div>`;
    } else {
      let html = '';
      let cursor = 0;

      spans.forEach((span) => {
        // Tokens before this span
        if (span.start > cursor) {
          const beforeTokens = tokens.slice(cursor, span.start);
          html += beforeTokens.map(escapeHtml).join(' ') + ' ';
        }

        // Entity badge
        const spanText = escapeHtml(span.tokens.join(' '));
        const avgConf = span.confs.reduce((a, b) => a + b, 0) / span.confs.length;
        const pctConf = Math.round(avgConf * 100);

        html += `<span class="entity-tag ${span.type}" title="${span.type} Entity &bull; Confidence: ${pctConf}%">${spanText} <span class="tag-badge">${span.type}</span></span> `;

        cursor = span.end;
      });

      // Remaining tokens
      if (cursor < tokens.length) {
        const remaining = tokens.slice(cursor);
        html += remaining.map(escapeHtml).join(' ');
      }

      highlightContainer.innerHTML = html.trim();
    }

    // Render Token Diagnostic Table
    let tableHtml = '';
    tokens.forEach((token, idx) => {
      const tag = tags[idx] || 'O';
      const conf = confidences[idx] !== undefined ? confidences[idx] : 1.0;
      const isEntity = tag !== 'O';
      const entType = isEntity ? tag.split('-')[1] : null;
      const meetsConf = conf >= minConf;
      const meetsType = entType ? selectedTypes.includes(entType) : false;

      let statusHtml = '';
      if (!isEntity) {
        statusHtml = `<span class="status-badge status-bg">Background</span>`;
      } else if (meetsConf && meetsType) {
        statusHtml = `<span class="status-badge status-detected">Active (${entType})</span>`;
      } else {
        const reasons = [];
        if (!meetsConf) reasons.push('Low Conf');
        if (!meetsType) reasons.push('Type Disabled');
        statusHtml = `<span class="status-badge status-filtered">Filtered (${reasons.join(', ')})</span>`;
      }

      const confPct = (conf * 100).toFixed(1);

      tableHtml += `
        <tr>
          <td style="color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">${idx + 1}</td>
          <td style="font-weight: 600;">${escapeHtml(token)}</td>
          <td>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; ${tag !== 'O' ? 'color: var(--color-accent);' : 'color: var(--text-light);'}">
              ${tag}
            </span>
          </td>
          <td>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">${confPct}%</span>
              <div style="flex: 1; height: 5px; background: #E2E8F0; border-radius: 9999px; overflow: hidden; min-width: 50px;">
                <div style="width: ${confPct}%; height: 100%; background: ${conf >= 0.7 ? '#10B981' : conf >= 0.4 ? '#F59E0B' : '#EF4444'};"></div>
              </div>
            </div>
          </td>
          <td>${statusHtml}</td>
        </tr>
      `;
    });
    tokenTableBody.innerHTML = tableHtml;

    // Render Subwords Breakdown Table
    if (data.subwords && Array.isArray(data.subwords)) {
      let subHtml = '';
      data.subwords.forEach((item) => {
        const subPieces = (item.subwords || []).map((s) => `<span style="display: inline-block; padding: 2px 6px; background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; margin: 1px;">${escapeHtml(s)}</span>`).join(' ');

        subHtml += `
          <tr>
            <td style="color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">${item.index}</td>
            <td style="font-weight: 600;">${escapeHtml(item.word)}</td>
            <td>${subPieces}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${item.count}</td>
            <td style="font-family: 'JetBrains Mono', monospace; color: var(--color-accent);">${escapeHtml(item.first_subword)}</td>
          </tr>
        `;
      });
      subwordTableBody.innerHTML = subHtml;
    }

    // Update JSON Payload Display
    const filteredPayload = {
      text: data.text,
      language: data.language,
      transfer_mode: data.transfer_mode,
      threshold_applied: minConf,
      entity_types_filter: selectedTypes,
      summary: {
        total_entities: totalEnt,
        per_count: perCount,
        loc_count: locCount,
        org_count: orgCount,
      },
      entities: spans.map((s) => ({
        entity: s.type,
        text: s.tokens.join(' '),
        tokens: s.tokens,
        start_token: s.start,
        end_token: s.end,
        average_confidence: parseFloat((s.confs.reduce((a, b) => a + b, 0) / s.confs.length).toFixed(4)),
      })),
      tokens: data.tokens,
      tags: data.tags,
      confidences: data.confidences,
    };
    jsonPayload.textContent = JSON.stringify(filteredPayload, null, 2);
  }

  // -------------------------------------------------------------
  // 9. Utility Functions
  // -------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Initial update
  updateTextStats();
});
