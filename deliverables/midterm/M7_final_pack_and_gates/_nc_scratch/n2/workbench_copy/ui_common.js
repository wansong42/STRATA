/* =========================================================================
   STRATA 沙盒 · 共享组件层（图标 / 四态主状态 / 横幅 / 回执 / 空态 / 哈希 / toast）
   交互入口统一为 [data-action]，由 bindActions 委托；验证入口统一为 [data-testid]。
   ========================================================================= */
"use strict";
window.UI = (function () {
  const CORE = window.SandboxCore;
  const COPY = CORE.COPY;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------------- 图标（1.6px stroke，统一 16 viewBox） ---------------- */
  const PATHS = {
    plus: "M8 3v10M3 8h10",
    folder: "M2 4.5h4l1.5 2H14v6H2z",
    import: "M8 2v8m0 0L5 7m3 3l3-3M3 12h10",
    save: "M3 3h8l2 2v8H3zM5 3v3h5V3M5 13V9h6v4",
    close: "M4 4l8 8M12 4l-8 8",
    reopen: "M3 8a5 5 0 1 1 1.5 3.5M3 8V4.5M3 8h3.5",
    eye: "M2 8s2.5-4 6-4 6 4 6 4-2.5 4-6 4-6-4-6-4z M8 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z",
    check: "M3 8.5l3.2 3.2L13 5",
    undo: "M6 4L3 7l3 3M3 7h7a3.5 3.5 0 0 1 0 7H8",
    redo: "M10 4l3 3-3 3M13 7H6a3.5 3.5 0 0 0 0 7h2",
    play: "M5 3.5l7 4.5-7 4.5z",
    spark: "M8 2l1.4 4.6L14 8l-4.6 1.4L8 14l-1.4-4.6L2 8l4.6-1.4z",
    download: "M8 2v8m0 0L5 7m3 3l3-3M3 13h10",
    copy: "M6 6h7v7H6zM4 10V3h7",
    expand: "M3 6l5 5 5-5",
    collapse: "M3 10l5-5 5 5",
    list: "M3 4h10M3 8h10M3 12h6",
    fit: "M8 2v3M8 11v3M2 8h3M11 8h3M8 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z",
    camera: "M2 5h3l1.5-2h3L11 5h3v8H2zM8 7a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z",
    alert: "M8 2.5L14 13H2zM8 6.5v3M8 11.4v.2",
    error: "M5 2h6l3 3v6l-3 3H5l-3-3V5zM8 5v4M8 11.4v.2",
    info: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM8 7v4M8 4.6v.2",
    okc: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM5.5 8l2 2 3.5-4",
    question: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM6.4 6.4a1.7 1.7 0 1 1 2.4 1.6c-.6.3-.8.7-.8 1.2M8 11.4v.2",
    clock: "M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM8 4.5V8l2.5 1.5",
    layers: "M8 2l6 3-6 3-6-3zM2 8.5l6 3 6-3M2 11.5l6 3 6-3",
    cube: "M8 2l5.5 3v6L8 14l-5.5-3V5zM8 8l5.5-3M8 8L2.5 5M8 8v6",
    shield: "M8 2l5 2v4c0 3-2.5 5-5 6-2.5-1-5-3-5-6V4z",
    link: "M6.5 9.5l3-3M5 11l-1.5 1.5a2.4 2.4 0 0 1-3.4-3.4L3 6.5M11 5l1.5-1.5a2.4 2.4 0 0 1 3.4 3.4L13 9.5",
    search: "M7 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM10 10l3.5 3.5",
    arrow: "M3 8h9m0 0L9 5m3 3L9 11",
    refresh: "M13 8a5 5 0 1 1-1.5-3.5M13 8V4.5M13 8h-3.5",
    file: "M4 2h5l3 3v9H4zM9 2v3h3",
    hash: "M6 2L4 14M12 2l-2 12M2.5 5.5h11M2 10.5h11",
    filter: "M2 3h12l-4.5 5v4L6.5 13V8z",
    split: "M8 2v12M2 8h4m4 0h4",
    merge: "M3 3l5 5-5 5M13 3l-5 5 5 5",
    lock: "M4 7h8v6H4zM6 7V5a2 2 0 0 1 4 0v2",
    dot: "M8 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z",
  };
  function icon(name, cls = "ic") {
    return `<svg class="${cls}" viewBox="0 0 16 16" aria-hidden="true"><path d="${PATHS[name] || PATHS.dot}"/></svg>`;
  }

  /* ---------------- 主状态推导：唯一权威 + 下一步 ---------------- */
  function statusOf(S) {
    const job = S.jobs[S.jobs.length - 1];
    if (!S.project_open) {
      return { tone: "info", icon: "folder", text: COPY.empty_title, next: "重开已保存版本，或新建合成项目开始。", actions: [{ action: "reopen", testid: "status-reopen", label: COPY.reopen, kind: "primary" }, { action: "open-synthetic", testid: "status-new", label: COPY.new_synthetic, kind: "" }] };
    }
    if (S.saveError) {
      return { tone: "err", icon: "error", text: COPY.save_failed_title, next: "旧 head 已保留，会话可继续编辑；重试或查看日志定位根因。", actions: [{ action: "retry-save", testid: "status-retry-save", label: COPY.retry_save, kind: "danger" }, { action: "view-log", testid: "status-view-log", label: COPY.view_log, kind: "" }] };
    }
    if (S.preview && CORE.previewIsLive()) {
      /* W170 W169C-02：预览待提交期间，若项目有未保存修改，主状态行保留「有未保存修改 / 尚未落盘」
         警告，不被预览横幅完全替换。预览本身不改项目，落盘状态仍以服务端 dirty 为准。 */
      const unsaved = S.dirty ? ` 有未保存修改：当前 v${S.project_version} 尚未落盘；保存后生成可重开版本。` : "";
      return { tone: "warn", icon: "eye", text: "预览待提交 · 项目未改" + (S.dirty ? " · 尚未落盘" : ""), next: `基版本 v${S.preview.base_version} · 核对差异后提交，或取消预览释放令牌。${unsaved}`, actions: [{ action: "commit", testid: "status-commit", label: COPY.commit, kind: "primary" }, { action: "cancel-preview", testid: "status-cancel-preview", label: COPY.cancel_preview, kind: "" }] };
    }
    if (S.activeJobId && job && job.status === "running") {
      return { tone: "info", icon: "clock", text: `求解进行中 · ${job.phase} · ${Math.round(job.progress * 100)}%`, next: "独立进程运行，完成前不发布候选；可随时取消。", actions: [{ action: "cancel-job", testid: "status-cancel-job", label: COPY.cancel_job, kind: "" }] };
    }
    if (job && ["failed", "cancelled", "timeout"].includes(job.status)) {
      return { tone: "warn", icon: "alert", text: `任务回退 · ${job.error?.code || job.status}`, next: "候选未发布，项目不变；可改用规则求解或调整范围重试。", actions: [{ action: "solve", testid: "status-solve", label: COPY.solve, kind: "" }] };
    }
    if (S.exportReceipt && S.exportReceipt.current && !S.dirty) {
      return { tone: "ok", icon: "okc", text: "当前版本导出已核对", next: `${S.exportReceipt.enumeration_count} 项读回一致 · 版本 v${S.project_version}；可在回执卡下载交付件。`, actions: [] };
    }
    if (S.dirty) {
      return { tone: "warn", icon: "alert", text: COPY.dirty, next: `当前 v${S.project_version} 尚未落盘；保存后生成可重开版本。`, actions: [{ action: "save", testid: "status-save", label: COPY.save, kind: "primary" }] };
    }
    return { tone: "ok", icon: "okc", text: `${COPY.clean} · v${S.project_version}`, next: "选择对象查看证据，或发起编辑预览。", actions: [] };
  }

  function statuslineHTML(S) {
    const st = statusOf(S);
    const acts = st.actions.map((a) => `<button class="btn sm ${a.kind}" data-action="${a.action}" data-testid="${a.testid}">${esc(a.label)}</button>`).join("");
    return `<div class="statusline ${st.tone}" data-testid="primary-status" role="status">
      ${icon(st.icon, "ic sl-icon")}
      <span class="sl-text">${esc(st.text)}</span>
      <span class="sl-next">${esc(st.next)}</span>
      <span class="sl-actions">${acts}</span>
    </div>`;
  }

  /* ---------------- 横幅 / 回执 / 空态 ---------------- */
  function saveErrorBanner(S) {
    if (!S.saveError) return "";
    const e = S.saveError;
    return `<div class="banner err" data-testid="save-error-banner" role="alert">
      ${icon("error", "ic b-icon")}
      <div class="b-main">
        <div class="b-title">${esc(COPY.save_failed_title)}<span class="b-code">${esc(e.code)}</span></div>
        <div class="b-body">${esc(e.message)}</div>
        <div class="b-body">${esc(COPY.save_failed_rollback)}</div>
        <div class="b-actions">
          <button class="btn sm danger" data-action="retry-save" data-testid="retry-save-btn">${icon("refresh")}${esc(COPY.retry_save)}</button>
          <button class="btn sm" data-action="view-log" data-testid="view-log-btn">${icon("list")}${esc(COPY.view_log)}</button>
        </div>
      </div>
    </div>`;
  }

  function saveReceiptHTML(S) {
    const r = S.saveReceipt;
    if (!r) return "";
    return `<div class="receipt" data-testid="save-receipt">
      <div class="r-head">${icon("okc")}保存回执 · verified</div>
      <dl class="dl">
        <dt>saved</dt><dd class="mono">true</dd>
        <dt>版本</dt><dd class="mono">v${r.project_version}</dd>
        <dt>文件数</dt><dd class="num">${r.file_count}</dd>
        <dt>哈希</dt><dd>${hashlineHTML(r.content_hash, { testid: "save-receipt-hash" })}</dd>
        <dt>时刻</dt><dd class="mono">${esc(r.at)}</dd>
      </dl>
    </div>`;
  }

  function exportReceiptHTML(S) {
    const r = S.exportReceipt;
    if (!r) return "";
    const tone = r.verified && r.current ? "" : " neutral";
    return `<div class="receipt${tone}" data-testid="export-receipt">
      <div class="r-head">${icon(r.verified && r.current ? "okc" : "alert")}导出读回 · ${r.verified && r.current ? "PASS" : "FAIL"}</div>
      <dl class="dl">
        <dt>枚举</dt><dd class="num">${r.enumeration_count} 项</dd>
        <dt>版本一致</dt><dd class="mono">${r.current}</dd>
        <dt>版本</dt><dd class="mono">v${r.project_version}</dd>
        <dt>export_id</dt><dd>${hashlineHTML(r.export_id, { testid: "export-id" })}</dd>
        <dt>导出目录</dt><dd data-testid="export-receipt-dir-cell">${r.export_dir ? hashlineHTML(r.export_dir, { testid: "export-receipt-dir" }) : '<span class="mono">—</span>'}</dd>
      </dl>
      <div class="btn-row tight" style="margin-top:8px">
        <button class="btn sm" data-action="download-project" data-testid="download-project">${icon("download")}project.json</button>
        <button class="btn sm" data-action="download-manifest" data-testid="download-manifest">${icon("download")}manifest.json</button>
        <button class="btn sm" data-action="download-readback" data-testid="download-readback">${icon("download")}readback.json</button>
      </div>
    </div>`;
  }

  function emptyHTML({ title, hint, actions = [], mark = "cube" }) {
    return `<div class="empty" data-testid="empty-state">
      ${icon(mark, "ic e-mark")}
      <div class="e-title">${esc(title)}</div>
      <div class="e-hint">${esc(hint)}</div>
      <div class="e-actions">${actions.map((a) => `<button class="btn ${a.kind || ""}" data-action="${a.action}" data-testid="${a.testid}">${esc(a.label)}</button>`).join("")}</div>
    </div>`;
  }

  /* ---------------- 哈希 / 长 ID ---------------- */
  function hashlineHTML(value, { testid = "", expanded = false } = {}) {
    return `<span class="hashline" data-value="${esc(value)}" data-expanded="${expanded}" ${testid ? `data-testid="${testid}"` : ""}>
      <span class="hv mono" title="${esc(value)}">${esc(value)}</span>
      <button class="icon-btn" data-action="hash-expand" title="展开 / 收起完整值" aria-expanded="${expanded}" aria-label="${expanded ? "收起完整值" : "展开完整值"}">${icon("expand")}</button>
      <button class="icon-btn" data-action="copy" title="复制完整值" aria-label="复制完整值">${icon("copy")}</button>
    </span>`;
  }

  /* ---------------- 支持索引（截断字段的可展开渲染） ----------------
     W171 W170C-03：<=8 全量；>8 给前 8 + … 与「展开全部 N 个」文本级控件（独立确认用文本探针）。
     来源 SHA 的 hashline 展开是 CSS 换行，与此处文本级展开是两套机制，互不替代。 */
  function supportIndicesHTML(indices, testid) {
    const list = Array.isArray(indices) ? indices : [];
    const tid = testid ? ` data-testid="${esc(testid)}"` : "";
    if (!list.length) return `<span class="mono">—</span>`;
    if (list.length <= 8) return `<span class="mono wrap-any"${tid}>${esc(list.join(", "))}</span>`;
    const full = list.join(", ");
    const preview = list.slice(0, 8).join(", ") + " …";
    return `<span class="idxline" data-expanded="false" data-count="${list.length}" data-value="${esc(full)}"${tid}>`
      + `<span class="idx-preview mono">${esc(preview)}</span>`
      + `<span class="idx-full mono wrap-any" hidden>${esc(full)}</span>`
      + `<button class="idx-toggle" data-action="indices-expand" aria-expanded="false">展开全部 ${list.length} 个</button>`
      + `</span>`;
  }

  /* ---------------- toast ---------------- */
  let stack = null;
  function toast(msg, tone = "info") {
    if (!stack) { stack = document.createElement("div"); stack.className = "toast-stack"; document.body.appendChild(stack); }
    const el = document.createElement("div");
    el.className = "toast " + tone;
    el.innerHTML = `${icon(tone === "ok" ? "okc" : tone === "err" ? "error" : tone === "warn" ? "alert" : "info", "ic t-icon")}
      <div class="t-body">${esc(msg)}</div>
      <button class="icon-btn t-close" aria-label="关闭提示">${icon("close")}</button>`;
    el.querySelector(".t-close").addEventListener("click", () => el.remove());
    stack.appendChild(el);
    setTimeout(() => el.remove(), 3600);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("已复制到剪贴板", "ok");
      return true;
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      ta.remove();
      toast(ok ? "已复制到剪贴板" : "复制失败，请手动选择文本", ok ? "ok" : "warn");
      return ok;
    }
  }

  /* ---------------- 动作委托 ---------------- */
  function bindActions(root, handlers) {
    root.addEventListener("click", (e) => {
      const t = e.target.closest("[data-action]");
      if (!t || !root.contains(t)) return;
      const action = t.dataset.action;
      if (action === "copy") {
        const holder = t.closest("[data-value]") || t.parentElement.closest("[data-value]");
        if (holder) copyText(holder.dataset.value);
        return;
      }
      if (action === "hash-expand") {
        const holder = t.closest(".hashline");
        if (holder) {
          const on = holder.dataset.expanded === "true";
          holder.dataset.expanded = String(!on);
          t.innerHTML = icon(on ? "expand" : "collapse");
          /* W172 OBS-B：aria 随翻转同步（on 是翻转前旧态）。渲染参数派生天然一致，
             不引入跨重绘展开态记忆（refresh 重绘重置 data-expanded 是既有行为）。 */
          t.setAttribute("aria-expanded", String(!on));
          t.setAttribute("aria-label", on ? "展开完整值" : "收起完整值");
        }
        return;
      }
      if (action === "indices-expand") {
        const holder = t.closest(".idxline");
        if (holder) {
          const next = holder.dataset.expanded !== "true";
          holder.dataset.expanded = String(next);
          const preview = holder.querySelector(".idx-preview");
          const full = holder.querySelector(".idx-full");
          if (preview) preview.hidden = next;
          if (full) full.hidden = !next;
          t.setAttribute("aria-expanded", String(next));
          t.textContent = next ? "收起" : ("展开全部 " + (holder.dataset.count || "") + " 个");
        }
        return;
      }
      const fn = handlers[action];
      if (fn) fn(t, e);
    });
  }

  /* ---------------- 评审 chrome ---------------- */
  function mountChrome(variant, state) {
    const params = new URLSearchParams(location.search);
    if (params.get("chrome") !== "1") return;
    const bar = document.createElement("div");
    bar.className = "sandbox-chrome";
    const variants = CORE.SEED.sandbox.variants;
    const states = CORE.SEED.sandbox.states;
    // URL 参数进 inline onclick，先白名单收敛，杜绝属性注入
    const vid = variants.some((v) => v.id === variant) ? variant : "a";
    const sid = states.some((s) => s.id === state) ? state : "loaded";
    variant = vid; state = sid;
    bar.innerHTML = `
      <span class="sc-label">候选效果</span>
      <span class="sc-note">不是 PF05 功能验收 · 最终选择由甲方作出</span>
      <span class="seg" role="tablist" aria-label="方案">
        ${variants.map((v) => `<button role="tab" aria-selected="${v.id === variant}" data-testid="chrome-variant" data-variant="${v.id}" onclick="location.href='variant_${v.id}.html?state=${state}&chrome=1'">${v.id.toUpperCase()} — ${esc(v.name)}</button>`).join("")}
      </span>
      <span class="seg" role="tablist" aria-label="状态">
        ${states.map((s) => `<button role="tab" aria-selected="${s.id === state}" data-testid="chrome-state" data-state="${s.id}">${esc(s.label)}</button>`).join("")}
      </span>
      <span class="spacer"></span>
      <a href="index.html" style="color:#f18b6a">总览</a>`;
    bar.addEventListener("click", (e) => {
      const b = e.target.closest("[data-state]");
      if (b && window.__SANDBOX__) window.__SANDBOX__.setState(b.dataset.state);
    });
    document.body.prepend(bar);
  }

  /* ---------------- 证据档案（三套共用同一份字段与口径） ---------------- */
  function evidenceHTML(S, id) {
    const ev = CORE.getEvidence(id);
    if (!ev) return `<div class="empty" style="min-height:120px"><div class="e-hint">未选择对象。从对象列表或三维中拾取一个稳定 ID。</div></div>`;
    const o = ev.object;
    const p = S.project;
    const unitUnverified = p.unit !== "m";
    const rqUndefined = p.real_quality === "undefined";
    const src = ev.source_files[0];
    const stateTxt = o.kind === "relation"
      ? `${o.state}${o.state === "unknown" ? " · " + (o.unknown_reason || "") : ""}`
      : (o.interpretation_status || "—");
    return `
      <dl class="dl">
        <dt>稳定 ID</dt><dd>${hashlineHTML(o.id, { testid: "object-id" })}</dd>
        <dt>类型</dt><dd>${esc(kindLabel[o.kind] || o.kind)}</dd>
        <dt>对象版本</dt><dd class="mono">v${o.version}</dd>
        <dt>解释 / 判定</dt><dd class="mono wrap-any">${esc(stateTxt)}</dd>
        ${o.member_fragment_ids ? `<dt>成员 Fragment</dt><dd class="mono wrap-any">${esc(o.member_fragment_ids.join(", ") || "—")}</dd>` : ""}
        ${o.from ? `<dt>两端</dt><dd class="mono">${esc(o.from)} → ${esc(o.to)}</dd>` : ""}
        <dt>复核</dt><dd><span class="badge ${reviewTone[S.review[o.id] || o.review] || ""}"><i class="dot"></i>${esc(reviewLabel[S.review[o.id] || o.review] || (S.review[o.id] || o.review))}</span></dd>
      </dl>
      <hr class="divider">
      <div class="block-title">支持证据<span class="tag">${esc(p.evidence_policy)}</span></div>
      <dl class="dl" style="margin-top:6px">
        <dt>单位</dt><dd>${unitUnverified ? `<span class="chip warn">${esc(COPY.unit_unverified)}</span> <span class="mono wrap-any">${esc(p.unit)}</span>` : `<span class="mono">${esc(p.unit)}</span>`}</dd>
        <dt>坐标</dt><dd class="mono wrap-any">${esc(p.coordinate_frame)}</dd>
        <dt>许可</dt><dd class="mono wrap-any">${esc(p.license)}</dd>
        <dt>来源</dt><dd class="mono wrap-any">${esc(src.path)}</dd>
        <dt>来源 SHA</dt><dd>${hashlineHTML(src.sha256, { testid: "source-sha" })}<span class="dim num"> ${fmtBytes(src.byte_count)}</span></dd>
        <dt>支持索引</dt><dd>${supportIndicesHTML(o.support_indices, "support-indices")}</dd>
        <dt>真实质量</dt><dd>${rqUndefined ? `<span class="chip unk">${esc(COPY.real_quality_undefined)}</span>` : `<span class="mono wrap-any">${esc(p.real_quality)}</span>`}</dd>
      </dl>
      ${o.conflict_note ? `<div class="banner err" style="margin-top:8px;padding:8px 10px">${icon("alert", "ic b-icon")}<div class="b-main"><div class="b-title" style="font-size:12px">边界解释冲突</div><div class="b-body">${esc(o.conflict_note)}</div></div></div>` : ""}
      ${o.kind === "relation" && o.state === "unknown" ? `<div class="banner unk" style="margin-top:8px;padding:8px 10px;background:var(--unk-soft);border-color:color-mix(in srgb,var(--unk) 40%,transparent);color:var(--unk)">${icon("question", "ic b-icon")}<div class="b-main"><div class="b-title" style="font-size:12px">unknown · ${esc(o.unknown_reason || "")}</div><div class="b-body">${esc(o.note || "")}</div></div></div>` : ""}
      ${o.note && o.kind !== "relation" ? `<p class="muted" style="margin-top:8px;font-size:11px">${esc(o.note)}</p>` : ""}
    `;
  }

  /* ---------------- 杂项 ---------------- */
  const fmtBytes = (n) => n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB" : (n / 1048576).toFixed(1) + " MB";
  const kindLabel = { surface: "面", fragment: "片", relation: "关系", protected: "保护" };
  const reviewLabel = { reviewed: "已复核", pending: "待复核", conflict: "冲突", unknown: "unknown" };
  const reviewTone = { reviewed: "ok", pending: "warn", conflict: "err", unknown: "unk" };

  function download(name, content) {
    const blob = new Blob([content], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return { esc, icon, statusOf, statuslineHTML, saveErrorBanner, saveReceiptHTML, exportReceiptHTML, emptyHTML, hashlineHTML, supportIndicesHTML, evidenceHTML, toast, copyText, bindActions, mountChrome, fmtBytes, kindLabel, reviewLabel, reviewTone, download };
})();

/* 哈希交叉断言钩子：Python 常量 = JS 重算 = Node 重算（验证脚本调用） */
window.__SANDBOX_SEAL__ = function () {
  const C = window.SandboxCore;
  return {
    abc: C.sha256Hex("abc"),
    canonical_hashes: C.SEAL.sha256,
    recomputed: {
      v1: C.sha256Hex(C.SEAL.canonical.v1),
      v2_commit: C.sha256Hex(C.SEAL.canonical.v2_commit),
      v2_solve: C.sha256Hex(C.SEAL.canonical.v2_solve),
    },
  };
};