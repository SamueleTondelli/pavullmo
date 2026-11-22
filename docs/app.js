(() => {
  "use strict";

  const THEME_STORAGE_KEY = "pavullmo-tracker-theme";

  const state = {
    data: null,
    versionsMap: new Map(),
    runCountByVersion: new Map(),
    bestBench: null,
    bestTest16k: null,
    versionsSearch: "",
    runsSearch: "",
    runsVersionId: "",
    runsBenchOnly: false,
    runsTestOnly: false,
    runsSortField: "date",
    runsSortDir: "desc",
    expandedRuns: new Set(),
    ablationsSearch: "",
    ablationsCategory: "",
    ablationsStatus: "",
  };

  function escapeHtml(value) {
    if (value === null || value === undefined) {
      return "";
    }
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatLoss(val) {
    return typeof val === "number" ? val.toFixed(4) : "—";
  }

  function formatPpl(val) {
    if (typeof val !== "number") {
      return "";
    }
    const ppl = Math.exp(val);
    return `PPL ${ppl >= 100 ? ppl.toFixed(1) : ppl.toFixed(2)}`;
  }

  function formatPct(val, digits = 2) {
    return typeof val === "number" ? `${(val * 100).toFixed(digits)}%` : "—";
  }

  function applyTheme(theme) {
    const resolved = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolved);
    const label = document.getElementById("theme-label");
    if (label) {
      label.textContent = resolved === "dark" ? "Light mode" : "Dark mode";
    }
    try {
      localStorage.setItem(THEME_STORAGE_KEY, resolved);
    } catch (_) {
      // Ignore storage failures in restricted contexts.
    }
  }

  function initTheme() {
    let saved = null;
    try {
      saved = localStorage.getItem(THEME_STORAGE_KEY);
    } catch (_) {
      saved = null;
    }
    if (!saved && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      saved = "dark";
    }
    applyTheme(saved || "light");

    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      toggle.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme");
        applyTheme(current === "dark" ? "light" : "dark");
      });
    }
  }

  async function loadTrackerData() {
    if (window.PAVULLMO_TRACKER_DATA && typeof window.PAVULLMO_TRACKER_DATA === "object") {
      return window.PAVULLMO_TRACKER_DATA;
    }
    const response = await fetch("data/tracker.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} loading data/tracker.json`);
    }
    return response.json();
  }

  function computeIndexes(data) {
    state.versionsMap.clear();
    state.runCountByVersion.clear();

    for (const ver of data.model_versions) {
      state.versionsMap.set(ver.id, ver);
      state.runCountByVersion.set(ver.id, 0);
    }

    let bestBenchRun = null;
    let bestTestRun = null;

    for (const run of data.training_runs) {
      state.runCountByVersion.set(
        run.version_id,
        (state.runCountByVersion.get(run.version_id) || 0) + 1
      );
      if (run.benchmarks && typeof run.benchmarks.overall_bidir === "number") {
        if (!bestBenchRun || run.benchmarks.overall_bidir > bestBenchRun.benchmarks.overall_bidir) {
          bestBenchRun = run;
        }
      }
      const ver = state.versionsMap.get(run.version_id);
      const is16k = ver && ver.tokenizer && ver.tokenizer.vocab_size === 16384;
      const isFullTest =
        typeof run.test_loss === "number" &&
        run.test_loss_scope &&
        run.test_loss_scope.includes("full 10M");
      if (is16k && isFullTest) {
        if (!bestTestRun || run.test_loss < bestTestRun.test_loss) {
          bestTestRun = run;
        }
      }
    }

    state.bestBench = bestBenchRun;
    state.bestTest16k = bestTestRun;
  }

  function renderHeaderAndKpis() {
    const data = state.data;
    document.getElementById("project-title").textContent = `${data.project} — Experiment Tracker`;
    document.getElementById("header-updated").textContent = `Updated ${data.updated_at}`;
    const repoLink = document.getElementById("repo-link");
    if (repoLink && data.repo_url) {
      repoLink.href = data.repo_url;
    }

    document.getElementById("benchmark-note-box").textContent = data.benchmark_note;
    document.getElementById("nav-summary-text").textContent =
      `${data.model_versions.length} versions · ${data.training_runs.length} runs · ${data.ablations.length} ablations`;

    document.getElementById("kpi-versions").textContent = String(data.model_versions.length);
    document.getElementById("kpi-runs").textContent = String(data.training_runs.length);
    const benchRunsCount = data.training_runs.filter((r) => r.benchmarks).length;
    document.getElementById("kpi-runs-sub").textContent = `${benchRunsCount} with 30-bench eval`;

    if (state.bestBench) {
      document.getElementById("kpi-best-bench").textContent = formatPct(
        state.bestBench.benchmarks.overall_bidir
      );
      document.getElementById("kpi-best-bench-sub").textContent = state.bestBench.name;
    }

    if (state.bestTest16k) {
      document.getElementById("kpi-best-test").textContent = `${formatLoss(
        state.bestTest16k.test_loss
      )}`;
      document.getElementById("kpi-best-test-sub").textContent =
        `${state.bestTest16k.name} (${formatPpl(state.bestTest16k.test_loss)})`;
    }

    document.getElementById("kpi-ablations").textContent = String(data.ablations.length);
    const promotedCount = data.ablations.filter((a) => a.status === "Promoted").length;
    document.getElementById("kpi-ablations-sub").textContent = `${promotedCount} promoted`;
    document.getElementById("kpi-updated").textContent = data.updated_at;
  }

  function populateVersionSelect() {
    const select = document.getElementById("runs-version-select");
    const current = state.runsVersionId;
    select.innerHTML = '<option value="">All model versions</option>';
    for (const ver of state.data.model_versions) {
      const option = document.createElement("option");
      option.value = ver.id;
      option.textContent = `${ver.id} — ${ver.name} (${ver.commit})`;
      select.appendChild(option);
    }
    select.value = current;
  }

  function renderVersionsTable() {
    const tbody = document.getElementById("versions-tbody");
    const query = state.versionsSearch.trim().toLowerCase();
    const repoUrl = state.data.repo_url || "";

    const filtered = state.data.model_versions.filter((ver) => {
      if (!query) return true;
      const haystack = [
        ver.id,
        ver.name,
        ver.commit,
        ver.branch,
        ver.params_label,
        ver.positional_encoding,
        ver.attention_variant,
        ver.optimizer_summary,
        ver.notes,
        ver.tokenizer.type,
        ver.tokenizer.artifact,
        String(ver.tokenizer.vocab_size),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });

    document.getElementById("versions-count").textContent =
      `${filtered.length} of ${state.data.model_versions.length} versions`;

    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-cell">No matching model versions.</td></tr>';
      return;
    }

    tbody.innerHTML = filtered
      .map((ver) => {
        const arch = ver.architecture;
        const tok = ver.tokenizer;
        const opt = ver.optimizer_details;
        const runCount = state.runCountByVersion.get(ver.id) || 0;
        const commitUrl = repoUrl ? `${repoUrl}/commit/${encodeURIComponent(ver.commit)}` : "#";

        return `
          <tr class="version-row" id="ver-${escapeHtml(ver.id)}">
            <td>
              <button type="button" class="version-trigger" data-filter-version="${escapeHtml(ver.id)}">
                ${escapeHtml(ver.name)}
              </button>
              <div class="sub-meta">
                <span class="mono">${escapeHtml(ver.id)}</span> ·
                <a class="commit-link" href="${escapeHtml(commitUrl)}" target="_blank" rel="noopener noreferrer" title="View commit on GitHub">${escapeHtml(ver.commit)}</a>
              </div>
              <div class="sub-soft">${escapeHtml(ver.commit_date)} · ${escapeHtml(ver.branch)}</div>
            </td>
            <td class="num-col">
              <div class="primary-title">${escapeHtml(ver.params_label)}</div>
              <div class="sub-soft">${ver.total_params.toLocaleString("en-US")}</div>
            </td>
            <td>
              <div class="primary-title">${arch.layers}L × ${arch.embed_dim}d</div>
              <div class="sub-meta">${escapeHtml(arch.ffn_type)} FFN ${arch.ffn_dim}</div>
              <div class="sub-soft">ctx ${arch.seq_len} · ${arch.tied_embeddings ? "tied emb" : "untied"}</div>
            </td>
            <td>
              <div class="primary-title">${tok.vocab_size.toLocaleString("en-US")}</div>
              <div class="sub-meta">${escapeHtml(tok.type)}</div>
              <div class="sub-soft"><code>${escapeHtml(tok.artifact)}</code></div>
            </td>
            <td>
              <div>${escapeHtml(ver.positional_encoding)}</div>
            </td>
            <td>
              <div>${escapeHtml(ver.attention_variant)}</div>
            </td>
            <td>
              <div class="primary-title">${escapeHtml(opt.optimizers)}</div>
              <div class="sub-meta">${escapeHtml(opt.lr)} · ${escapeHtml(opt.schedule)}</div>
              <div class="sub-soft">WD: ${escapeHtml(opt.weight_decay)} · ${escapeHtml(opt.batch_size)} · init: ${escapeHtml(opt.init)}${opt.ema && opt.ema !== "None" ? ` · EMA: ${escapeHtml(opt.ema)}` : ""}</div>
            </td>
            <td class="num-col">
              <button type="button" class="run-count-btn" data-filter-version="${escapeHtml(ver.id)}" title="Filter training runs to ${escapeHtml(ver.id)}">
                ${runCount}
              </button>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  function getRunSortValue(run, field) {
    if (field === "date") {
      return run.date;
    }
    if (field === "test_loss") {
      return typeof run.test_loss === "number" ? run.test_loss : null;
    }
    if (field === "val_loss") {
      return typeof run.val_loss === "number" ? run.val_loss : null;
    }
    if (field === "benchmarks") {
      return run.benchmarks && typeof run.benchmarks.overall_bidir === "number"
        ? run.benchmarks.overall_bidir
        : null;
    }
    return run.date;
  }

  function renderRunsTable() {
    const tbody = document.getElementById("runs-tbody");
    const query = state.runsSearch.trim().toLowerCase();

    const filtered = state.data.training_runs.filter((run) => {
      if (state.runsVersionId && run.version_id !== state.runsVersionId) {
        return false;
      }
      if (state.runsBenchOnly && !run.benchmarks) {
        return false;
      }
      if (state.runsTestOnly && typeof run.test_loss !== "number") {
        return false;
      }
      if (!query) {
        return true;
      }
      const ver = state.versionsMap.get(run.version_id);
      const haystack = [
        run.id,
        run.name,
        run.date,
        run.dataset_name,
        run.tokens_label,
        run.version_id,
        ver ? ver.name : "",
        ver ? ver.commit : "",
        run.notes,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });

    filtered.sort((a, b) => {
      const va = getRunSortValue(a, state.runsSortField);
      const vb = getRunSortValue(b, state.runsSortField);
      if (va === null && vb === null) {
        return b.date.localeCompare(a.date);
      }
      if (va === null) return 1;
      if (vb === null) return -1;
      const cmp = typeof va === "string" ? va.localeCompare(vb) : va - vb;
      if (cmp === 0) {
        return b.date.localeCompare(a.date);
      }
      return state.runsSortDir === "asc" ? cmp : -cmp;
    });

    document.getElementById("runs-count").textContent =
      `${filtered.length} of ${state.data.training_runs.length} runs`;

    const chipRow = document.getElementById("active-version-chip");
    const chipLabel = document.getElementById("active-version-label");
    if (state.runsVersionId) {
      const ver = state.versionsMap.get(state.runsVersionId);
      chipLabel.textContent = ver ? `${ver.id} (${ver.name})` : state.runsVersionId;
      chipRow.hidden = false;
    } else {
      chipRow.hidden = true;
    }

    document.querySelectorAll(".sort-btn").forEach((btn) => {
      const ind = btn.querySelector(".sort-ind");
      if (!ind) return;
      if (btn.dataset.sort === state.runsSortField) {
        ind.textContent = state.runsSortDir === "asc" ? "▲" : "▼";
      } else {
        ind.textContent = "↕";
      }
    });

    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-cell">No matching training runs.</td></tr>';
      return;
    }

    const rowsHtml = [];
    for (const run of filtered) {
      const ver = state.versionsMap.get(run.version_id);
      const isExpanded = state.expandedRuns.has(run.id);
      const isBestBench = state.bestBench && state.bestBench.id === run.id;
      const isBestTest = state.bestTest16k && state.bestTest16k.id === run.id;

      const testHtml =
        typeof run.test_loss === "number"
          ? `<div class="metric-main ${isBestTest ? "metric-best" : ""}">${formatLoss(run.test_loss)}</div>
             <div class="sub-soft">${formatPpl(run.test_loss)}${run.test_loss_scope ? `<span class="scope-tag">${escapeHtml(run.test_loss_scope)}</span>` : ""}</div>`
          : '<span class="muted-dash">—</span>';

      const valHtml =
        typeof run.val_loss === "number"
          ? `<div class="metric-main">${formatLoss(run.val_loss)}</div>
             <div class="sub-soft">${formatPpl(run.val_loss)}${run.val_loss_scope ? `<span class="scope-tag">${escapeHtml(run.val_loss_scope)}</span>` : ""}</div>
             ${typeof run.val_loss_prefix === "number" ? `<div class="sub-soft">prefix/alt: ${formatLoss(run.val_loss_prefix)}</div>` : ""}`
          : '<span class="muted-dash">—</span>';

      let benchHtml = '<span class="muted-dash">—</span>';
      if (run.benchmarks) {
        const b = run.benchmarks;
        benchHtml = `
          <div class="bench-wrap">
            <span class="bench-overall ${isBestBench ? "metric-best" : ""}">${formatPct(b.overall_bidir)} overall</span>
            <div class="bench-cats">
              <span class="bench-cat">ICL ${formatPct(b.icl, 1)}</span>
              <span class="bench-cat">Know ${formatPct(b.knowledge, 1)}</span>
              <span class="bench-cat">Lang ${formatPct(b.language_manipulation, 1)}</span>
              <span class="bench-cat">Math ${formatPct(b.math, 1)}</span>
              <span class="bench-cat">M-L ${formatPct(b.math_languages, 1)}</span>
              <span class="bench-cat">Reas ${formatPct(b.reasoning, 1)}</span>
            </div>
          </div>
        `;
      }

      rowsHtml.push(`
        <tr class="run-row ${isExpanded ? "is-expanded" : ""}" data-run-id="${escapeHtml(run.id)}">
          <td>
            <div class="primary-title">
              <span class="expand-caret">${isExpanded ? "▾" : "▸"}</span>
              ${escapeHtml(run.name)}
            </div>
            <div class="sub-meta">
              <span class="mono">${escapeHtml(run.date)}</span> ·
              <span class="token-tag">${escapeHtml(run.tokens_label)}</span>
            </div>
          </td>
          <td class="num-col">${testHtml}</td>
          <td class="num-col">${valHtml}</td>
          <td>${benchHtml}</td>
          <td>
            <div class="primary-title">${escapeHtml(run.dataset_name)}</div>
          </td>
          <td>
            <button type="button" class="version-pill" data-jump-version="${escapeHtml(run.version_id)}" title="Jump to ${escapeHtml(run.version_id)} in Model Versions table">
              <span class="version-pill-id">${escapeHtml(run.version_id)}</span>
              <span class="sub-soft">${ver ? `${escapeHtml(ver.commit)} · ${escapeHtml(ver.params_label)}` : ""}</span>
            </button>
          </td>
        </tr>
      `);

      if (isExpanded) {
        rowsHtml.push(renderRunDetailRow(run, ver));
      }
    }

    tbody.innerHTML = rowsHtml.join("");
  }

  function renderRunDetailRow(run, ver) {
    const b = run.benchmarks;
    const catDefs = b
      ? [
          ["ICL", b.icl],
          ["Knowledge", b.knowledge],
          ["Language Manip.", b.language_manipulation],
          ["Math", b.math],
          ["Math-Languages", b.math_languages],
          ["Reasoning", b.reasoning],
        ]
      : [];

    const barsHtml = b
      ? `<div class="detail-block">
          <h4>30-Task Benchmark Breakdown (Macro Bidirectional Accuracy)</h4>
          <div class="bench-bars">
            ${catDefs
              .map(
                ([label, val]) => `
              <div class="bench-bar-row">
                <span>${escapeHtml(label)}</span>
                <div class="bench-bar-track">
                  <div class="bench-bar-fill" style="width: ${Math.max(0, Math.min(100, val * 100)).toFixed(1)}%"></div>
                </div>
                <span class="num-col">${formatPct(val, 2)}</span>
              </div>
            `
              )
              .join("")}
          </div>
        </div>`
      : `<div class="detail-block">
          <h4>30-Task Benchmarks</h4>
          <p class="sub-meta">No synthetic benchmark suite evaluation recorded for this run.</p>
        </div>`;

    return `
      <tr class="run-detail-row">
        <td colspan="6">
          <div class="detail-grid">
            <div class="detail-block">
              <h4>Run Details &amp; Notes</h4>
              <div class="detail-kv">
                <span><strong>Run ID:</strong> <code>${escapeHtml(run.id)}</code></span>
                <span><strong>Train Loss:</strong> ${typeof run.train_loss === "number" ? `${formatLoss(run.train_loss)} (${formatPpl(run.train_loss)})` : "—"}</span>
                <span><strong>Commit:</strong> <code>${ver ? escapeHtml(ver.commit) : "—"}</code></span>
              </div>
              <p class="detail-notes">${escapeHtml(run.notes || "No additional notes.")}</p>
            </div>
            ${barsHtml}
          </div>
        </td>
      </tr>
    `;
  }

  function renderAblationsTable() {
    const tbody = document.getElementById("ablations-tbody");
    const query = state.ablationsSearch.trim().toLowerCase();

    const filtered = state.data.ablations.filter((abl) => {
      if (state.ablationsCategory && abl.category !== state.ablationsCategory) {
        return false;
      }
      if (state.ablationsStatus && abl.status !== state.ablationsStatus) {
        return false;
      }
      if (!query) return true;
      const haystack = [
        abl.id,
        abl.date,
        abl.title,
        abl.category,
        abl.status,
        abl.testing,
        abl.results,
        abl.conclusion,
        abl.source,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });

    document.getElementById("ablations-count").textContent =
      `${filtered.length} of ${state.data.ablations.length} ablations`;

    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No matching ablations.</td></tr>';
      return;
    }

    tbody.innerHTML = filtered
      .map((abl) => {
        const statusClass = `status-${abl.status.toLowerCase()}`;
        return `
          <tr>
            <td>
              <div class="primary-title mono">${escapeHtml(abl.id)}</div>
              <div class="sub-meta mono">${escapeHtml(abl.date)}</div>
              <span class="cat-badge">${escapeHtml(abl.category)}</span>
            </td>
            <td>
              <div class="primary-title">${escapeHtml(abl.title)}</div>
              <div class="sub-soft"><code>${escapeHtml(abl.source)}</code></div>
            </td>
            <td>${escapeHtml(abl.testing)}</td>
            <td>${escapeHtml(abl.results)}</td>
            <td>
              <span class="status-badge ${escapeHtml(statusClass)}">${escapeHtml(abl.status)}</span>
              <div class="sub-meta">${escapeHtml(abl.conclusion)}</div>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  function filterRunsByVersion(versionId) {
    state.runsVersionId = versionId;
    document.getElementById("runs-version-select").value = versionId;
    renderRunsTable();
    const runsSec = document.getElementById("runs");
    if (runsSec) {
      runsSec.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function jumpToVersion(versionId) {
    state.versionsSearch = "";
    document.getElementById("versions-search").value = "";
    renderVersionsTable();

    const row = document.getElementById(`ver-${versionId}`);
    if (row) {
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.classList.add("is-highlighted");
      setTimeout(() => row.classList.remove("is-highlighted"), 1800);
    }
  }

  function bindEvents() {
    document.getElementById("versions-search").addEventListener("input", (e) => {
      state.versionsSearch = e.target.value;
      renderVersionsTable();
    });

    document.getElementById("versions-reset").addEventListener("click", () => {
      state.versionsSearch = "";
      document.getElementById("versions-search").value = "";
      renderVersionsTable();
    });

    document.getElementById("versions-tbody").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-filter-version]");
      if (btn) {
        filterRunsByVersion(btn.dataset.filterVersion);
      }
    });

    document.getElementById("runs-search").addEventListener("input", (e) => {
      state.runsSearch = e.target.value;
      renderRunsTable();
    });

    document.getElementById("runs-version-select").addEventListener("change", (e) => {
      state.runsVersionId = e.target.value;
      renderRunsTable();
    });

    document.getElementById("runs-bench-only").addEventListener("change", (e) => {
      state.runsBenchOnly = e.target.checked;
      renderRunsTable();
    });

    document.getElementById("runs-test-only").addEventListener("change", (e) => {
      state.runsTestOnly = e.target.checked;
      renderRunsTable();
    });

    document.getElementById("clear-version-chip").addEventListener("click", () => {
      state.runsVersionId = "";
      document.getElementById("runs-version-select").value = "";
      renderRunsTable();
    });

    document.getElementById("runs-reset").addEventListener("click", () => {
      state.runsSearch = "";
      state.runsVersionId = "";
      state.runsBenchOnly = false;
      state.runsTestOnly = false;
      state.runsSortField = "date";
      state.runsSortDir = "desc";
      document.getElementById("runs-search").value = "";
      document.getElementById("runs-version-select").value = "";
      document.getElementById("runs-bench-only").checked = false;
      document.getElementById("runs-test-only").checked = false;
      renderRunsTable();
    });

    document.querySelectorAll(".sort-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const field = btn.dataset.sort;
        if (state.runsSortField === field) {
          state.runsSortDir = state.runsSortDir === "asc" ? "desc" : "asc";
        } else {
          state.runsSortField = field;
          state.runsSortDir = field === "test_loss" || field === "val_loss" ? "asc" : "desc";
        }
        renderRunsTable();
      });
    });

    document.getElementById("runs-tbody").addEventListener("click", (e) => {
      const jumpBtn = e.target.closest("[data-jump-version]");
      if (jumpBtn) {
        e.stopPropagation();
        jumpToVersion(jumpBtn.dataset.jumpVersion);
        return;
      }
      const row = e.target.closest("tr.run-row");
      if (row && row.dataset.runId) {
        const id = row.dataset.runId;
        if (state.expandedRuns.has(id)) {
          state.expandedRuns.delete(id);
        } else {
          state.expandedRuns.add(id);
        }
        renderRunsTable();
      }
    });

    document.getElementById("ablations-search").addEventListener("input", (e) => {
      state.ablationsSearch = e.target.value;
      renderAblationsTable();
    });

    document.getElementById("ablations-status-select").addEventListener("change", (e) => {
      state.ablationsStatus = e.target.value;
      renderAblationsTable();
    });

    document.getElementById("ablations-category-pills").addEventListener("click", (e) => {
      const pill = e.target.closest(".pill-btn");
      if (!pill) return;
      state.ablationsCategory = pill.dataset.category || "";
      document.querySelectorAll("#ablations-category-pills .pill-btn").forEach((b) => {
        b.classList.toggle("active", b === pill);
      });
      renderAblationsTable();
    });

    document.getElementById("ablations-reset").addEventListener("click", () => {
      state.ablationsSearch = "";
      state.ablationsCategory = "";
      state.ablationsStatus = "";
      document.getElementById("ablations-search").value = "";
      document.getElementById("ablations-status-select").value = "";
      document.querySelectorAll("#ablations-category-pills .pill-btn").forEach((b) => {
        b.classList.toggle("active", (b.dataset.category || "") === "");
      });
      renderAblationsTable();
    });
  }

  async function init() {
    initTheme();
    try {
      const data = await loadTrackerData();
      state.data = data;
      computeIndexes(data);
      renderHeaderAndKpis();
      populateVersionSelect();
      bindEvents();
      renderVersionsTable();
      renderRunsTable();
      renderAblationsTable();
    } catch (err) {
      const banner = document.getElementById("load-error");
      banner.textContent = `Failed to load tracker data: ${err.message}`;
      banner.hidden = false;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
