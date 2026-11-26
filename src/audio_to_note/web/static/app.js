const state = {
  env: null,
  config: null,
  prompts: [],
  jobs: [],
  selectedFile: null,
  currentJobId: null,
  currentJobData: null,
  eventSource: null,
  streamedSummary: "",
};

const $ = (id) => document.getElementById(id);

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = await res.text();
    try { msg = JSON.parse(msg).detail || msg; } catch {}
    throw new Error(msg || `HTTP ${res.status}`);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Lightweight safe Markdown renderer (headings, tables, lists, code, bold)
function renderMarkdown(md) {
  if (!md) return '<p class="muted">No content yet.</p>';
  const lines = escapeHtml(md).split("\n");
  let html = "";
  let inList = false;
  let inTable = false;

  const inline = (s) =>
    s.replace(/`([^`]+)`/g, "<code>$1</code>")
     .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
     .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*\|.*\|\s*$/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      const cells = line.trim().slice(1, -1).split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-+:?$/.test(c))) continue;
      if (!inTable) {
        html += "<table><thead><tr>" + cells.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
        inTable = true;
      } else {
        html += "<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
      }
      continue;
    }
    if (inTable) { html += "</tbody></table>"; inTable = false; }

    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) {
      if (inList) { html += "</ul>"; inList = false; }
      const lvl = h[1].length;
      html += `<h${lvl}>${inline(h[2])}</h${lvl}>`;
      continue;
    }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(li[1])}</li>`;
      continue;
    }
    if (inList && !line.trim()) { html += "</ul>"; inList = false; continue; }
    if (!line.trim()) { html += "<br/>"; continue; }
    html += `<p>${inline(line)}</p>`;
  }
  if (inList) html += "</ul>";
  if (inTable) html += "</tbody></table>";
  return html;
}

function showView(viewId) {
  $("view-new-job").classList.toggle("hidden", viewId !== "new-job");
  $("view-job-detail").classList.toggle("hidden", viewId !== "job-detail");
}

function setBadge(el, statusObj) {
  el.className = "status-badge";
  if (!statusObj) return;
  if (statusObj.available) {
    el.classList.add("ready");
    el.textContent = "Ready";
  } else {
    el.classList.add("warn");
    el.textContent = statusObj.status.replace("_", " ");
  }
}

function updateASRStatusUI() {
  const pid = $("asr-provider").value;
  const st = (state.env?.asr || []).find((x) => x.id === pid);
  setBadge($("asr-status-badge"), st);
  const hint = $("asr-hint");
  if (st && !st.available && st.hint) {
    hint.textContent = `${st.detail} — Setup: ${st.hint}`;
    hint.classList.remove("hidden");
  } else {
    hint.classList.add("hidden");
  }
  const dl = $("asr-detected-models");
  dl.innerHTML = (state.env?.models?.gguf || []).map((f) => `<option value="${escapeHtml(f)}">`).join("");
}

function updateDiarStatusUI() {
  const pid = $("diar-provider").value;
  const st = (state.env?.diarization || []).find((x) => x.id === pid);
  setBadge($("diar-status-badge"), st);
  const hint = $("diar-hint");
  if (st && !st.available && st.hint) {
    hint.textContent = `${st.detail} — Setup: ${st.hint}`;
    hint.classList.remove("hidden");
  } else {
    hint.classList.add("hidden");
  }
}

function updateLLMStatusUI() {
  const pid = $("llm-provider").value;
  const st = (state.env?.llm || []).find((x) => x.id === pid);
  setBadge($("llm-status-badge"), st);
}

function syncPromptEditor() {
  const pid = $("prompt-select").value;
  const tpl = state.prompts.find((p) => p.id === pid);
  if (tpl) {
    $("prompt-system-editor").value = tpl.system_prompt || "";
    $("prompt-user-editor").value = tpl.user_prompt || "";
  }
}

async function loadInitial() {
  const [env, cfg, hw, prompts, jobs] = await Promise.all([
    api("/api/environment"),
    api("/api/config"),
    api("/api/hotwords"),
    api("/api/prompts"),
    api("/api/jobs"),
  ]);
  state.env = env;
  state.config = cfg;
  state.prompts = prompts;
  state.jobs = jobs;

  $("app-version").textContent = `v${env.version}`;
  $("quick-badges").innerHTML = (env.tools || [])
    .map((t) => `<span class="status-badge ${t.available ? "ready" : "warn"}">${escapeHtml(t.name)}: ${t.available ? "OK" : "Missing"}</span>`)
    .join("");

  // Populate ASR & Diarization dropdowns
  $("asr-provider").innerHTML = (env.asr || [])
    .map((a) => `<option value="${a.id}">${a.available ? "✓" : "○"} ${escapeHtml(a.name)}</option>`)
    .join("");
  $("asr-provider").value = cfg.asr?.provider || "crispasr";
  $("asr-language").value = cfg.asr?.language || "auto";
  updateASRStatusUI();

  $("diar-provider").innerHTML = (env.diarization || [])
    .map((d) => `<option value="${d.id}">${d.available ? "✓" : "○"} ${escapeHtml(d.name)}</option>`)
    .join("");
  $("diar-provider").value = cfg.diarization?.provider || "none";
  updateDiarStatusUI();

  // Hotwords
  $("hotwords-editor").value = hw.text || "";

  // LLM presets & fields
  const presets = env.llm_presets || {};
  $("llm-preset").innerHTML =
    '<option value="custom">Custom</option>' +
    Object.entries(presets)
      .map(([k, v]) => `<option value="${k}">${escapeHtml(v.label)}</option>`)
      .join("");
  $("llm-provider").value = cfg.llm?.provider || "openai_compat";
  $("llm-preset").value = cfg.llm?.preset || "ollama";
  $("llm-base-url").value = cfg.llm?.providers?.openai_compat?.base_url || "";
  $("llm-api-key").value = cfg.llm?.providers?.openai_compat?.api_key || "";
  $("llm-model").value = cfg.llm?.model || "";
  $("output-language").value = cfg.pipeline?.output_language || "auto";
  $("toggle-polish").checked = !!cfg.pipeline?.polish_enabled;
  $("toggle-summarize").checked = cfg.pipeline?.summarize_enabled !== false;
  updateLLMStatusUI();

  // Prompts
  const sumPrompts = prompts.filter((p) => p.stage !== "polish");
  const promptOpts = sumPrompts
    .map((p) => `<option value="${p.id}">${escapeHtml(p.name)}</option>`)
    .join("");
  $("prompt-select").innerHTML = promptOpts;
  $("resum-prompt").innerHTML = promptOpts;
  $("prompt-select").value = cfg.pipeline?.default_prompt || "meeting_minutes";
  syncPromptEditor();

  renderJobList();
}

function renderJobList() {
  const el = $("job-list");
  if (!state.jobs.length) {
    el.innerHTML = '<div class="empty-hint">No jobs yet</div>';
    return;
  }
  el.innerHTML = state.jobs
    .map(
      (j) => `
      <div class="job-item ${j.id === state.currentJobId ? "active" : ""}" data-job-id="${escapeHtml(j.id)}">
        <div class="job-item-title">${escapeHtml(j.title || j.id)}</div>
        <div class="job-item-meta">
          <span>${escapeHtml(j.status)}</span>
          <span>${escapeHtml((j.created_at || "").slice(0, 16).replace("T", " "))}</span>
        </div>
      </div>`
    )
    .join("");
  el.querySelectorAll(".job-item").forEach((node) => {
    node.addEventListener("click", () => openJob(node.dataset.jobId));
  });
}

async function openJob(jobId) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.currentJobId = jobId;
  state.streamedSummary = "";
  renderJobList();
  showView("job-detail");

  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  renderJobDetail(data);

  if (data.status === "queued" || data.status === "running") {
    subscribeJobEvents(jobId);
  }
}

function formatSeconds(sec) {
  const total = Math.max(0, Math.floor(sec || 0));
  const h = String(Math.floor(total / 3600)).padStart(2, "0");
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function renderJobDetail(data) {
  state.currentJobData = data;
  $("job-detail-title").textContent = data.title || data.id;
  const badge = $("job-detail-status");
  badge.className = `status-badge ${data.status}`;
  badge.textContent = data.status;

  const snap = data.config_snapshot || {};
  const dur = data.metrics?.duration_seconds ? `${Math.round(data.metrics.duration_seconds)}s` : "";
  $("job-detail-meta").textContent = `${data.id} • ASR: ${snap.asr_provider || "-"} • Diarize: ${snap.diarizer_provider || "none"} • LLM: ${snap.llm_model || "-"} ${dur ? "• Duration: " + dur : ""}`;

  const pct = Math.round((data.progress || 0) * 100);
  $("progress-pct").textContent = `${pct}%`;
  $("progress-bar").style.width = `${pct}%`;
  $("progress-stage-label").textContent = data.message || data.stage || data.status;

  const errBox = $("job-error-box");
  if (data.error) {
    errBox.textContent = data.error;
    errBox.classList.remove("hidden");
  } else {
    errBox.classList.add("hidden");
  }

  // Transcript pane
  const mode = $("transcript-view-mode").value;
  const editMode = $("transcript-edit-mode").value;
  const hasPolished = !!(data.content?.polished_md);
  $("transcript-view-mode").querySelector('option[value="polished"]').disabled = !hasPolished;
  const trText = mode === "polished" && hasPolished ? data.content.polished_md : data.content?.transcript_md || "";
  const segments = mode === "polished" && data.polished_segments?.length ? data.polished_segments : (data.segments || []);

  const speakers = data.speakers || [];
  const aliases = data.speaker_aliases || {};
  const speakerSuggestions = new Set(speakers);
  Object.values(aliases).forEach((v) => { if (v) speakerSuggestions.add(v); });
  $("job-speakers-datalist").innerHTML = Array.from(speakerSuggestions)
    .map((s) => `<option value="${escapeHtml(s)}">`)
    .join("");

  const segEditor = $("transcript-segments-editor");
  const mdEditor = $("transcript-md-editor");
  const preOut = $("transcript-output");
  const saveBtn = $("btn-save-transcript");

  if (editMode === "segments" && segments.length > 0) {
    segEditor.classList.remove("hidden");
    mdEditor.classList.add("hidden");
    preOut.classList.add("hidden");
    saveBtn.classList.remove("hidden");
    segEditor.innerHTML = segments
      .map(
        (seg, idx) => `<div class="seg-row" data-idx="${idx}" data-start="${seg.start}" data-end="${seg.end}">
          <span class="seg-ts">${formatSeconds(seg.start)}</span>
          <input type="text" class="seg-speaker" list="job-speakers-datalist" value="${escapeHtml(seg.speaker || "")}" />
          <textarea class="seg-text" rows="2">${escapeHtml(seg.text || "")}</textarea>
        </div>`
      )
      .join("");
  } else if (editMode === "markdown" && trText) {
    segEditor.classList.add("hidden");
    mdEditor.classList.remove("hidden");
    preOut.classList.add("hidden");
    saveBtn.classList.remove("hidden");
    mdEditor.value = trText;
  } else {
    segEditor.classList.add("hidden");
    mdEditor.classList.add("hidden");
    preOut.classList.remove("hidden");
    saveBtn.classList.add("hidden");
    preOut.textContent = trText || "Waiting for transcript...";
  }

  // Speaker rename chips
  const bar = $("speaker-rename-bar");
  if (speakers.length) {
    bar.classList.remove("hidden");
    bar.innerHTML =
      '<span class="muted small">Batch rename speakers:</span>' +
      speakers
        .map(
          (spk) =>
            `<span class="speaker-chip"><span>${escapeHtml(spk)}&rarr;</span><input type="text" data-spk="${escapeHtml(spk)}" value="${escapeHtml(aliases[spk] || "")}" placeholder="Name"/></span>`
        )
        .join("") +
      '<button class="btn btn-secondary btn-sm" id="btn-apply-speakers">Apply</button>';
    $("btn-apply-speakers").addEventListener("click", applySpeakerAliases);
  } else {
    bar.classList.add("hidden");
  }

  // Summary pane
  const sumMd = data.content?.summary_md || state.streamedSummary;
  $("summary-output").innerHTML = renderMarkdown(sumMd);
  $("link-dl-transcript").href = `/api/jobs/${encodeURIComponent(data.id)}/export/${hasPolished && mode === "polished" ? "polished.md" : "transcript.md"}`;
  $("link-dl-summary").href = `/api/jobs/${encodeURIComponent(data.id)}/export/summary.md`;
}

function subscribeJobEvents(jobId) {
  const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
  state.eventSource = es;
  es.onmessage = async (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "llm_token") {
      state.streamedSummary += ev.token || "";
      $("summary-output").innerHTML = renderMarkdown(state.streamedSummary);
      return;
    }
    if (ev.progress !== undefined) {
      const pct = Math.round(ev.progress * 100);
      $("progress-pct").textContent = `${pct}%`;
      $("progress-bar").style.width = `${pct}%`;
    }
    if (ev.message) {
      $("progress-stage-label").textContent = ev.message;
    }
    if (ev.type === "stage_done" || ev.type === "job_done" || ev.type === "job_error") {
      const fresh = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      renderJobDetail(fresh);
      state.jobs = await api("/api/jobs");
      renderJobList();
      if (ev.type === "job_done" || ev.type === "job_error") {
        es.close();
        state.eventSource = null;
      }
    }
  };
}

async function applySpeakerAliases() {
  const aliases = {};
  $("speaker-rename-bar").querySelectorAll("input[data-spk]").forEach((inp) => {
    if (inp.value.trim()) aliases[inp.dataset.spk] = inp.value.trim();
  });
  const res = await api(`/api/jobs/${encodeURIComponent(state.currentJobId)}/speakers`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ aliases }),
  });
  if (state.currentJobData) {
    state.currentJobData.speaker_aliases = res.speaker_aliases;
    state.currentJobData.content.transcript_md = res.transcript_md;
    if (res.polished_md) state.currentJobData.content.polished_md = res.polished_md;
    renderJobDetail(state.currentJobData);
  }
}

async function startPipeline(runSummarize) {
  const localPath = $("local-path-input").value.trim();
  if (!state.selectedFile && !localPath) {
    alert("Please drop/choose an audio or video file, or enter a local file path.");
    return;
  }
  const numSpk = $("num-speakers").value.trim();
  const options = {
    title: $("job-title-input").value.trim(),
    asr_provider: $("asr-provider").value,
    asr_language: $("asr-language").value,
    asr_model: $("asr-model-override").value.trim() || null,
    diarizer_provider: $("diar-provider").value,
    num_speakers: numSpk ? parseInt(numSpk, 10) : null,
    hotwords_text: $("hotwords-editor").value,
    run_polish: $("toggle-polish").checked,
    run_summarize: runSummarize,
    llm_provider: $("llm-provider").value,
    llm_base_url: $("llm-base-url").value.trim(),
    llm_api_key: $("llm-api-key").value.trim(),
    llm_model: $("llm-model").value.trim(),
    prompt_id: $("prompt-select").value,
    custom_system_prompt: $("prompt-system-editor").value,
    custom_user_prompt: $("prompt-user-editor").value,
    output_language: $("output-language").value,
  };

  const form = new FormData();
  form.append("options", JSON.stringify(options));
  if (state.selectedFile) {
    form.append("file", state.selectedFile);
  } else {
    form.append("local_path", localPath);
  }

  const res = await api("/api/jobs", { method: "POST", body: form });
  state.jobs = await api("/api/jobs");
  await openJob(res.job_id);
}

async function browseTo(path) {
  const data = await api(`/api/fs/browse?path=${encodeURIComponent(path || "")}`);
  $("browse-path-bar").value = data.current;
  $("btn-browse-up").dataset.parent = data.parent || "";
  const list = $("browse-list");
  list.innerHTML =
    data.dirs
      .map((d) => `<div class="browse-row" data-type="dir" data-path="${escapeHtml(d.path)}"><span>📁 ${escapeHtml(d.name)}</span><span class="muted">folder</span></div>`)
      .join("") +
    data.files
      .map((f) => `<div class="browse-row" data-type="file" data-path="${escapeHtml(f.path)}"><span>📄 ${escapeHtml(f.name)}</span><span class="muted">${f.size_mb} MB</span></div>`)
      .join("");
  list.querySelectorAll(".browse-row").forEach((row) => {
    row.addEventListener("click", () => {
      if (row.dataset.type === "dir") {
        browseTo(row.dataset.path);
      } else {
        $("local-path-input").value = row.dataset.path;
        state.selectedFile = null;
        $("selected-file-label").innerHTML = `Selected local path: <strong>${escapeHtml(row.dataset.path)}</strong>`;
        $("modal-browse").classList.add("hidden");
      }
    });
  });
}

function bindEvents() {
  $("btn-new-job").addEventListener("click", () => {
    state.currentJobId = null;
    renderJobList();
    showView("new-job");
  });

  $("asr-provider").addEventListener("change", updateASRStatusUI);
  $("diar-provider").addEventListener("change", updateDiarStatusUI);
  $("llm-provider").addEventListener("change", updateLLMStatusUI);
  $("prompt-select").addEventListener("change", syncPromptEditor);
  $("toggle-summarize").addEventListener("change", () => {
    $("llm-config-section").classList.toggle("hidden", !$("toggle-summarize").checked && !$("toggle-polish").checked);
  });

  $("llm-preset").addEventListener("change", () => {
    const preset = state.env?.llm_presets?.[$("llm-preset").value];
    if (preset) {
      $("llm-base-url").value = preset.base_url;
      if (preset.default_model) $("llm-model").value = preset.default_model;
    }
  });

  // Dropzone
  const dz = $("dropzone");
  $("btn-pick-upload").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    if (e.target.files?.[0]) {
      state.selectedFile = e.target.files[0];
      $("local-path-input").value = "";
      $("selected-file-label").innerHTML = `Upload ready: <strong>${escapeHtml(state.selectedFile.name)}</strong>`;
    }
  });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files?.[0]) {
      state.selectedFile = e.dataTransfer.files[0];
      $("local-path-input").value = "";
      $("selected-file-label").innerHTML = `Upload ready: <strong>${escapeHtml(state.selectedFile.name)}</strong>`;
    }
  });

  $("btn-save-hotwords").addEventListener("click", async () => {
    await api("/api/hotwords", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("hotwords-editor").value }),
    });
    $("hotwords-save-msg").textContent = "Saved!";
    setTimeout(() => ($("hotwords-save-msg").textContent = ""), 2000);
  });

  $("btn-save-prompt").addEventListener("click", async () => {
    const customName = $("prompt-save-name").value.trim();
    const activeId = $("prompt-select").value;
    const activeTpl = state.prompts.find((p) => p.id === activeId);
    const payload = {
      id: customName ? customName.toLowerCase().replace(/[^a-z0-9]+/g, "_") : activeId,
      name: customName || activeTpl?.name || activeId,
      description: activeTpl?.description || "Custom prompt",
      stage: "summarize",
      system_prompt: $("prompt-system-editor").value,
      user_prompt: $("prompt-user-editor").value,
    };
    const res = customName
      ? await api("/api/prompts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
      : await api(`/api/prompts/${encodeURIComponent(activeId)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    state.prompts = await api("/api/prompts");
    const opts = state.prompts.filter((p) => p.stage !== "polish").map((p) => `<option value="${p.id}">${escapeHtml(p.name)}</option>`).join("");
    $("prompt-select").innerHTML = opts;
    $("resum-prompt").innerHTML = opts;
    $("prompt-select").value = res.prompt.id;
    $("prompt-save-name").value = "";
    alert(`Saved prompt: ${res.prompt.name}`);
  });

  $("btn-test-llm").addEventListener("click", async () => {
    $("llm-test-msg").textContent = "Testing...";
    const res = await api("/api/llm/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: $("llm-provider").value,
        base_url: $("llm-base-url").value.trim(),
        api_key: $("llm-api-key").value.trim(),
      }),
    });
    $("llm-test-msg").textContent = res.ok ? `✓ ${res.detail}` : `✗ ${res.detail}`;
  });

  $("btn-fetch-models").addEventListener("click", async () => {
    const res = await api("/api/llm/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: $("llm-provider").value,
        base_url: $("llm-base-url").value.trim(),
        api_key: $("llm-api-key").value.trim(),
      }),
    });
    if (res.ok && res.models?.length) {
      $("llm-models-datalist").innerHTML = res.models.map((m) => `<option value="${escapeHtml(m)}">`).join("");
      if (!$("llm-model").value.trim()) $("llm-model").value = res.models[0];
      $("llm-test-msg").textContent = `Found ${res.models.length} models`;
    } else {
      $("llm-test-msg").textContent = res.detail || "No models returned";
    }
  });

  $("btn-start-pipeline").addEventListener("click", () => startPipeline($("toggle-summarize").checked));
  $("btn-transcribe-only").addEventListener("click", () => startPipeline(false));

  // Job detail actions
  $("transcript-view-mode").addEventListener("change", () => state.currentJobData && renderJobDetail(state.currentJobData));
  $("transcript-edit-mode").addEventListener("change", () => state.currentJobData && renderJobDetail(state.currentJobData));
  $("btn-save-transcript").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    const target = $("transcript-view-mode").value;
    const editMode = $("transcript-edit-mode").value;
    const payload = { target };
    if (editMode === "segments") {
      payload.segments = Array.from($("transcript-segments-editor").querySelectorAll(".seg-row")).map((row) => ({
        start: parseFloat(row.dataset.start || "0"),
        end: parseFloat(row.dataset.end || "0"),
        speaker: row.querySelector(".seg-speaker").value.trim() || "SPEAKER_UNKNOWN",
        text: row.querySelector(".seg-text").value.trim(),
      }));
    } else if (editMode === "markdown") {
      payload.markdown = $("transcript-md-editor").value;
    } else {
      return;
    }
    const res = await api(`/api/jobs/${encodeURIComponent(state.currentJobId)}/transcript`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (state.currentJobData) {
      state.currentJobData.speakers = res.speakers;
      if (target === "polished") {
        state.currentJobData.polished_segments = res.segments;
        state.currentJobData.content.polished_md = res.polished_md;
      } else {
        state.currentJobData.segments = res.segments;
        state.currentJobData.content.transcript_md = res.transcript_md;
      }
      renderJobDetail(state.currentJobData);
    }
    $("transcript-save-msg").textContent = "Saved!";
    setTimeout(() => ($("transcript-save-msg").textContent = ""), 2000);
  });
  $("btn-copy-transcript").addEventListener("click", () => {
    const mode = $("transcript-view-mode").value;
    const txt = $("transcript-edit-mode").value === "markdown"
      ? $("transcript-md-editor").value
      : (mode === "polished" ? state.currentJobData?.content?.polished_md : state.currentJobData?.content?.transcript_md) || "";
    navigator.clipboard.writeText(txt);
  });
  $("btn-copy-summary").addEventListener("click", () => navigator.clipboard.writeText(state.currentJobData?.content?.summary_md || state.streamedSummary || ""));

  $("btn-resummarize").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    state.streamedSummary = "";
    await api(`/api/jobs/${encodeURIComponent(state.currentJobId)}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt_id: $("resum-prompt").value,
        llm_model: $("resum-model").value.trim() || null,
        run_polish: $("resum-polish").checked,
        run_summarize: true,
      }),
    });
    subscribeJobEvents(state.currentJobId);
  });

  $("btn-delete-job").addEventListener("click", async () => {
    if (!state.currentJobId || !confirm("Delete this job and its output files?")) return;
    await api(`/api/jobs/${encodeURIComponent(state.currentJobId)}`, { method: "DELETE" });
    state.currentJobId = null;
    state.jobs = await api("/api/jobs");
    renderJobList();
    showView("new-job");
  });

  // Modals
  document.querySelectorAll("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => $(btn.dataset.close).classList.add("hidden"));
  });

  $("btn-browse-audio").addEventListener("click", () => {
    $("modal-browse").classList.remove("hidden");
    browseTo($("local-path-input").value.trim());
  });
  $("btn-browse-up").addEventListener("click", () => browseTo($("btn-browse-up").dataset.parent));
  $("btn-browse-go").addEventListener("click", () => browseTo($("browse-path-bar").value));

  $("btn-env").addEventListener("click", async () => {
    const env = await api("/api/environment");
    state.env = env;
    const renderSection = (title, items) =>
      `<h3>${title}</h3>` +
      items
        .map(
          (i) => `<div style="margin:6px 0;padding:8px;border:1px solid var(--border);border-radius:7px">
            <strong>${escapeHtml(i.name)}</strong>
            <span class="status-badge ${i.available ? "ready" : "warn"}" style="margin-left:8px">${i.available ? "READY" : escapeHtml(i.status)}</span>
            <div class="muted small">${escapeHtml(i.detail)}</div>
            ${i.hint ? `<div class="small" style="color:#fde68a">Hint: ${escapeHtml(i.hint)}</div>` : ""}
          </div>`
        )
        .join("");
    $("env-modal-body").innerHTML =
      renderSection("Tools & Hardware", env.tools) +
      renderSection("ASR Engines", env.asr) +
      renderSection("Speaker Diarization", env.diarization) +
      renderSection("LLM Summarizers", env.llm);
    $("modal-env").classList.remove("hidden");
  });

  $("btn-settings").addEventListener("click", async () => {
    const cfg = await api("/api/config");
    state.config = cfg;
    $("set-crisp-bin").value = cfg.asr?.providers?.crispasr?.runtime_bin || "";
    $("set-crisp-model").value = cfg.asr?.providers?.crispasr?.model || "";
    $("set-oasr-url").value = cfg.asr?.providers?.openai_compat?.base_url || "";
    $("set-oasr-key").value = cfg.asr?.providers?.openai_compat?.api_key || "";
    $("set-sh-seg").value = cfg.diarization?.providers?.sherpa_onnx?.segmentation_model || "";
    $("set-sh-emb").value = cfg.diarization?.providers?.sherpa_onnx?.embedding_model || "";
    $("set-hf-token").value = cfg.diarization?.providers?.pyannote?.hf_token || "";
    $("set-llm-url").value = cfg.llm?.providers?.openai_compat?.base_url || "";
    $("set-llm-key").value = cfg.llm?.providers?.openai_compat?.api_key || "";
    $("set-llm-model").value = cfg.llm?.model || "";
    $("set-llm-ctx").value = cfg.llm?.max_context_tokens || 12000;
    $("modal-settings").classList.remove("hidden");
  });

  $("btn-save-settings").addEventListener("click", async () => {
    const payload = {
      asr: {
        providers: {
          crispasr: { runtime_bin: $("set-crisp-bin").value.trim(), model: $("set-crisp-model").value.trim() },
          openai_compat: { base_url: $("set-oasr-url").value.trim(), api_key: $("set-oasr-key").value.trim() },
        },
      },
      diarization: {
        providers: {
          sherpa_onnx: { segmentation_model: $("set-sh-seg").value.trim(), embedding_model: $("set-sh-emb").value.trim() },
          pyannote: { hf_token: $("set-hf-token").value.trim() },
        },
      },
      llm: {
        model: $("set-llm-model").value.trim(),
        max_context_tokens: parseInt($("set-llm-ctx").value || "12000", 10),
        providers: {
          openai_compat: { base_url: $("set-llm-url").value.trim(), api_key: $("set-llm-key").value.trim() },
        },
      },
    };
    await api("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("modal-settings").classList.add("hidden");
    await loadInitial();
  });
}

bindEvents();
loadInitial().catch((err) => console.error(err));
