/* =========================================================================
   STRATA · 纸本仪器版（候选 C）· 视图层
   只负责布局与渲染；动作语义复用 actions.js，状态与哈希复用 core.js。
   设计语言：纸本颗粒 / 发丝线 / 墨色 / 单一矿物绿；主状态唯一权威在状态行；
   事务台与底栏合并为 48px 命令栏；常态状态行不放重复动作。
   ========================================================================= */
"use strict";
(function () {
  const CORE = window.SandboxCore;
  const UI = window.UI;
  const S = CORE.S;
  const COPY = CORE.COPY;
  /* F1 伴随修正：领域标签原文是 "真实质量 undefined"（W160 的真实质量指标本就未定义）。
     右栏要求不出现裸 undefined，这里只在显示层换成中文，数据字段本身不变。 */
  const REAL_QUALITY_UNDEFINED = "真实质量未定义";
  const SEED = CORE.SEED;
  const esc = UI.esc;
  const icon = UI.icon;
  const $ = (s) => document.querySelector(s);

  const params = new URLSearchParams(location.search);

  const view = {
    filter: "all", query: "", logOpen: false, inspectorOpen: false, ledgerOpen: false,
    menu: null, modal: null, saving: false, reopening: false, undoRedoBusy: false,
    /* W172 长耗时动作忙态统一：action→值（open 家族值=catalog id，portable-open 值=包路径，其余 true）。 */
    busy: {},
    edit: { action: "SET_INTERPRETATION", targets: "", payloadText: "", payloadDirty: false, drafts: {} },
  };
  let scene = null;
  let lastProjectRef = undefined;
  let lastReceiptSig = "";
  let lastSig = "";
  let rafPending = false;

  /* ---------------- 提示：不可交互，永不抢点击 ---------------- */
  UI.toast = function (msg, tone = "info") {
    const root = $("#toast-root");
    const el = document.createElement("div");
    el.className = "toast " + tone;
    el.innerHTML = icon(tone === "ok" ? "okc" : tone === "err" ? "error" : tone === "warn" ? "alert" : "info") + `<div>${esc(msg)}</div>`;
    root.appendChild(el);
    while (root.children.length > 3) root.firstChild.remove();
    /* W169：错误提示留 8 秒。保存/预览在真实 ROI 上要等 10 秒以上，
       3.2 秒的 toast 会在操作者抬头前消失，被误读成"没有报错"（W168-D01）。 */
    setTimeout(() => el.remove(), tone === "err" ? 8000 : 3200);
  };

  /* 复制：clipboard API 在权限被静默拒绝时可能永不 settle，超时回退 execCommand */
  async function copyText(text) {
    const fallback = () => {
      const ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      ta.remove();
      UI.toast(ok ? "已复制到剪贴板" : "复制失败，请展开后手动选择文本", ok ? "ok" : "warn");
    };
    try {
      await Promise.race([
        navigator.clipboard.writeText(text),
        new Promise((_, rej) => setTimeout(() => rej(new Error("clipboard timeout")), 1200)),
      ]);
      UI.toast("已复制到剪贴板", "ok");
    } catch { fallback(); }
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest('[data-action="copy"]');
    if (!t) return;
    e.stopPropagation();
    const holder = t.closest("[data-value]") || (t.parentElement && t.parentElement.closest("[data-value]"));
    if (holder) copyText(holder.dataset.value);
  }, true);

  /* ---------------- 小件 ---------------- */
  const GLYPHS = {
    surface: '<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 5.2L8 2.6l5 2.6-1.4 6.2H4.4z"/></svg>',
    fragment: '<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3.2l5 9.2H3z"/></svg>',
    relation: '<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 12L12.4 4.4M12.4 4.4H8.2M12.4 4.4v4.2"/></svg>',
    protected: '<svg class="ic" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.6 3.6h8.8v8.8H3.6z" stroke-dasharray="2.4 2"/></svg>',
  };
  const TONE_VAR = { ok: "var(--ok)", warn: "var(--warn)", err: "var(--err)", unk: "var(--unk)" };
  const reviewOf = (o) => S.review[o.id] || o.review || "reviewed";
  const badge = (tone, label) => `<span class="badge ${tone}"><i class="dot"></i>${esc(label)}</span>`;
  const hashline = (v, testid) => UI.hashlineHTML(v, testid ? { testid } : {});
  const fmtTime = (iso) => String(iso || "").replace("T", " ").slice(0, 19);
  /* 证据状态计数：取值只来自 GET /api/state.evidence_status_counts，四项固定顺序。
     W170 W169C-01：每项加中文标识（当前/过期/冲突/超范围）与含义 tooltip，并另给可见图例；
     后端状态机语义零改动；data-* 机器可读属性与 data-testid 原样保留（独立确认轮据此读数）。 */
  const CURRENCY_KEYS = ["current", "stale", "conflict", "out_of_scope"];
  const CURRENCY_LABELS = { current: "当前", stale: "过期", conflict: "冲突", out_of_scope: "超范围" };
  const CURRENCY_MEANING = {
    current: "证据与当前几何和版本一致，可计入求解",
    stale: "证据所依据的输入已变化，需重算后方可用",
    conflict: "证据与当前判定冲突，需人工复核",
    out_of_scope: "证据超出当前求解范围，不计入"
  };
  function evidenceCountsHTML() {
    const counts = S.evidence_counts || {};
    const attrs = CURRENCY_KEYS.map(function (key) {
      const value = counts[key] === undefined ? "" : counts[key];
      return ' data-' + key.replace(/_/g, "-") + '="' + esc(value) + '"';
    }).join("");
    const items = CURRENCY_KEYS.map(function (key) {
      const value = counts[key] === undefined ? "—" : counts[key];
      const tip = CURRENCY_LABELS[key] + "（" + key + "）：" + CURRENCY_MEANING[key];
      return '<span class="sl-ev-item" title="' + esc(tip) + '"><b>' + CURRENCY_LABELS[key] + '</b> ' + key + '=' + esc(value) + '</span>';
    }).join('<span class="sl-ev-sep">·</span>');
    const legendTip = "证据当前性计数（GET /api/state evidence_status_counts）。"
      + CURRENCY_KEYS.map(function (key) { return CURRENCY_LABELS[key] + " " + key + "：" + CURRENCY_MEANING[key]; }).join("；");
    return '<span class="sl-ev" data-testid="evidence-status-counts" title="' + esc(legendTip) + '"' + attrs + '>' + items + '</span>';
  }
  /* W170 W169C-01：可见中文图例（独立一行，不挤占状态计量行），含义与计数同源同口径。 */
  function evidenceLegendHTML() {
    const items = CURRENCY_KEYS.map(function (key) {
      return '<span class="sl-ev-legend-item"><b>' + CURRENCY_LABELS[key] + '</b> ' + key + '：' + esc(CURRENCY_MEANING[key]) + '</span>';
    }).join('<span class="sl-ev-sep">·</span>');
    return '<div class="sl-ev-legend" data-testid="evidence-currency-legend"><span class="sl-ev-legend-title">证据当前性图例</span>' + items + '</div>';
  }
  function statusMetaHTML() {
    const hash = S.content_hash || "—";
    return '<span class="sl-meta">'
      + '<span class="sl-ver"><i class="sl-ver-mark">v</i><b data-testid="project-version">' + S.project_version + '</b></span>'
      + '<span class="sl-hash" data-testid="content-hash" title="' + esc(hash) + '">' + esc(hash) + '</span>'
      + (S.read_only_legacy ? '<span class="sl-ro" data-testid="read-only-human">只读 v1（未迁移）</span>' : '')
      + '<span class="sl-flag" data-testid="read-only-flag">read_only_legacy=' + S.read_only_legacy + '</span>'
      + evidenceCountsHTML()
      + '<span class="sl-snap" data-testid="evidence-snapshot-id" title="' + esc(S.evidence_snapshot_id || "") + '">' + esc(S.evidence_snapshot_id || "—") + '</span>'
      + '</span>';
  }

  /* W169 F2/F3：只读 v1 的常驻中文引导。
     判定只认服务状态 read_only_legacy，不在本地推断；迁移成功后本横幅自然消失。 */
  function readonlyBannerHTML() {
    if (!S.project_open || !S.read_only_legacy) return "";
    return '<span class="rb-mark">只读 v1</span>'
      + '<span class="rb-text">当前为<b>只读 v1 副本</b>：三维证据与清单可以看，但预览、提交与保存都会被拒绝。'
      + '编辑或保存前，请先点顶栏「<b>迁移</b>」把它升级为可编辑副本；迁移不改版本号，内容哈希会变。</span>'
      + '<button class="btn sm primary rb-btn" data-action="migrate" data-testid="readonly-banner-migrate"' + (view.busy.migrate ? ' disabled' : '') + '>' + (view.busy.migrate ? '迁移中…' : '迁移') + '</button>';
  }

  /* 回执卡：数值一律来自服务响应（save receipt / export readback），无本地推断。 */
  function receiptsHTML() {
    const parts = [];
    const save = S.saveReceipt;
    if (save) {
      parts.push('<div class="receipt-block" data-testid="save-receipt-api">'
        + '<div class="receipt-head">保存回执 · 服务响应</div><dl class="dl">'
        + '<dt>receipt.verified</dt><dd class="mono" data-testid="save-receipt-verified">' + save.verified + '</dd>'
        + '<dt>bundle_id</dt><dd class="mono" data-testid="save-receipt-bundle">' + esc(save.bundle_id || "") + '</dd>'
        + '<dt>版本</dt><dd class="mono">v' + save.project_version + '</dd>'
        + '<dt>文件数</dt><dd class="mono">' + save.file_count + '</dd>'
        + '<dt>哈希</dt><dd class="mono">' + esc(save.content_hash || "") + '</dd>'
        + '</dl></div>');
    }
    const exp = S.exportReceipt;
    if (exp) {
      parts.push('<div class="receipt-block" data-testid="export-receipt-api">'
        + '<div class="receipt-head">导出回执 · 新进程读回</div><dl class="dl">'
        + '<dt>readback.verified</dt><dd class="mono" data-testid="export-readback-verified">' + exp.verified + '</dd>'
        + '<dt>enumeration_count</dt><dd class="mono" data-testid="export-enumeration">' + exp.enumeration_count + '</dd>'
        + '<dt>版本一致</dt><dd class="mono">' + exp.current + '</dd>'
        + '<dt>export_id</dt><dd class="mono">' + esc(exp.export_id || "") + '</dd>'
        /* W170 W169C-04：导出目录在盘路径原文（可复制），来自服务响应 export_dir，非本地推断。 */
        + '<dt>导出目录（在盘路径）</dt><dd data-testid="export-dir-cell">' + (exp.export_dir ? hashline(exp.export_dir, "export-dir") : '<span class="mono">—</span>') + '</dd>'
        + '</dl></div>');
    }
    return parts.join("");
  }

  /* 登记资料入口：条目来自 GET /api/catalog，构建缺失的条目标注 disabled。 */
  function catalogButtonsHTML() {
    const entries = ((S.catalog && S.catalog.entries) || []).filter(function (entry) {
      return entry.id !== "synthetic_contract" && entry.id !== "internal_pf02";
    });
    let body = entries.map(function (entry) {
      /* W172 忙态统一：open 家族飞行中全部禁用；被点按钮文案「打开中…」。 */
      const flag = (entry.build_present === false || openFamilyBusy()) ? " disabled" : "";
      const label = view.busy.open === entry.id ? "打开中…" : (entry.label || entry.id);
      return '<button class="btn ghost" data-action="open-catalog" data-catalog="' + esc(entry.id) + '"'
        + ' data-testid="open-catalog-' + esc(entry.id) + '" title="' + esc(entry.scope || "") + '"' + flag + '>'
        + esc(label) + '</button>';
    }).join("");
    /* W163：W154 判定已在盘，入口按真实判定显示；点击进便携面板（导出/打开/重定位）。 */
    const w154 = (S.portableStatus && S.portableStatus.w154) || null;
    const w154Label = w154 && w154.present
      ? ("W154 便携包 · " + (w154.verdict || "已就绪"))
      : "W154 便携包 · 判定未读到";
    body += '<button class="btn ghost" data-action="portable-panel" data-testid="portable-import"'
      + ' title="导出 / 打开 / 重定位 W154 .ffportable（GET /api/portable/status）">' + esc(w154Label) + '</button>';
    return '<div class="w162-registry" data-testid="registry-entries"><span class="micro">登记资料</span><div class="w162-registry-row">' + body + '</div></div>';
  }

  /* W171 W170C-04：后端拒绝码 → 中文可操作说明（code 仍保留在括号里，英文原文留在日志详情）。
     未知 code 回退英文 message；LEGACY_READ_ONLY 走专用分支。 */
  const REJECT_ZH = {
    BAD_TARGET: "目标对象不对或缺失（动作要求的对象类型与所选不符）",
    BAD_SPLIT: "拆分载荷不合法（子集/点数/成员数不满足拆分要求）",
    /* W172 OBS-C：后端 BAD_SUPPORT 只查空/重复（edit_engine.py:1732-1733），不查「是否属于该片支持点」。 */
    BAD_SUPPORT: "支持索引不合法（空或重复）",
    FINITE_DOMAIN_EMPTY: "该选择的有限支持域为空（选点太稀疏，围不出凹包域）；请改用更密集的子集或从骨架起步",
    /* W172 OBS-C：以下两码已只读核实后端字面量后新增。
       SUPPORT_INDEX_OUT_OF_RANGE：workbench_service.py:1159-1163（索引越界观测点云）。
       INSUFFICIENT_SUPPORT：support_geometry.py:222-226 与 232-237（去重后 < MIN_SUPPORT_POINTS=3）。 */
    SUPPORT_INDEX_OUT_OF_RANGE: "索引超出观测点云范围",
    INSUFFICIENT_SUPPORT: "去重后支持点不足 3 个",
    SELECT_SUPPORT_BREAKS_SURFACE_COPLANARITY: "所选支持点会破坏所在面的共面性",
    SPLIT_BREAKS_SURFACE_COPLANARITY: "该拆分会破坏面的共面性",
    SPLIT_REQUIRES_COPLANAR_CHILD_SUPPORT: "拆分要求子片支持共面"
  };

  /* W172 OBS-A：预览第三出口（POST 已发）的 failText 按 code 分流，不再谎称「未发起」。
     fail() 返回对象只有 code/message/path（core_api.js fail），无 status 字段，故按 code 判。
     客户端两处预检（目标为空 / JSON 解析失败）确实未发 POST，保持「预览未发起」。
     只改预览三处调用点实参，不动 toastResult 共享语义（save/commit/migrate 等共用）。 */
  function previewFailText(result) {
    const code = (result && result.code) || "";
    if (code === "NETWORK_ERROR" || code === "RESPONSE_NOT_JSON") return "预览请求失败";
    if (code === "PREVIEW_MISSING") return "预览响应异常";
    return "预览被拒绝";
  }

  /* W172 长耗时动作忙态统一（复用 W169 保存侧范式：守卫标志 + 控件禁用/文案 + 事前 toast）。
     busyStart 返回 false 表示同名动作已在飞行（重入守卫，防重复提交）。
     refresh 渲染得到的控件（topbar/cmdbar/扉页/菜单）在渲染函数内按 view.busy 给忙态；
     模态内控件（便携面板）refresh 不重渲染 modal-root，另用直接 DOM 忙态（domBusy）。 */
  function busyStart(name, value, preToast) {
    if (view.busy[name]) return false;
    view.busy[name] = value === undefined ? true : value;
    refresh(true);
    if (preToast) UI.toast(preToast, "info");
    return true;
  }
  function busyEnd(name) { delete view.busy[name]; refresh(true); }

  /* 直接 DOM 忙态（模态用）：存旧值再改；finally 里若按钮仍在盘（isConnected）恢复旧值。
     完成路径 openModal 全量重渲染时 view.busy 已清，新渲染自然正常。
     禁飞行中整体重渲染 #portable-body（会丢 portable-new-path 输入值）。 */
  function domBusy(el, text, title) {
    if (!el) return null;
    const old = { textContent: el.textContent, disabled: el.disabled, title: el.getAttribute("title") };
    el.disabled = true;
    if (text !== undefined) el.textContent = text;
    if (title !== undefined) el.setAttribute("title", title);
    return { el: el, old: old };
  }
  function domBusyEnd(saved) {
    if (!saved) return;
    const list = Array.isArray(saved) ? saved : [saved];
    for (const item of list) {
      if (!item || !item.el || !item.el.isConnected) continue;
      item.el.disabled = item.old.disabled;
      item.el.textContent = item.old.textContent;
      if (item.old.title === null) item.el.removeAttribute("title");
      else item.el.setAttribute("title", item.old.title);
    }
  }

  function toastResult(result, okText, failText) {
    if (result && result.ok) { UI.toast(okText, "ok"); return true; }
    const code = (result && result.code) || "UNKNOWN";
    const message = (result && result.message) || "";
    /* W169 F2：只读 v1 的拒绝不是"系统故障"，文案必须直说下一步点哪里；
       原始 code 保留在括号里，接口语义不改（后端拒绝仍然是正确行为）。 */
    if (code === "LEGACY_READ_ONLY") {
      UI.toast(failText + " · 只读 v1 不可编辑/保存 · 请先点顶栏「迁移」（" + code + "）", "err");
      return false;
    }
    UI.toast(failText + " · " + (REJECT_ZH[code] || message || "服务拒绝该请求") + "（" + code + "）", "err");
    return false;
  }

  function downloadExportFile(name) {
    const file = (S.exportFiles || []).filter(function (item) { return item.name === name; })[0];
    if (!file) { UI.toast("导出件不可用 · 先执行导出", "warn"); return; }
    const anchor = document.createElement("a");
    anchor.href = file.url;
    anchor.download = name;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    UI.toast(name + " · 由服务 /download 提供", "ok");
  }

  async function afterOpen() {
    if (!S.selectedId && S.project && S.project.objects.length) await CORE.selectObject(S.project.objects[0].id);
    refresh(true);
  }

  function lastBundleId() {
    if (S.bundle_id) return S.bundle_id;
    return (S.library && S.library.length) ? S.library[0].bundle_id : null;
  }

  function pickImportFile() {
    const input = document.createElement("input");
    input.type = "file"; input.accept = ".json,application/json"; input.style.display = "none";
    input.addEventListener("change", function () {
      const file = input.files && input.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async function () {
        /* W172 忙态统一：import 飞行中与 open 家族互禁（渲染函数按 view.busy 给 disabled）。 */
        if (!busyStart("import")) return;
        try {
          const result = await CORE.importProject(String(reader.result));
          toastResult(result, "导入成功 · 严格契约校验通过", "导入被拒绝");
          refresh(true);
        } finally { busyEnd("import"); }
      };
      reader.readAsText(file);
      input.value = "";
    });
    document.body.appendChild(input);
    input.click();
    setTimeout(function () { input.remove(); }, 0);
  }

  /* 动作语义全部落到真实 /api/*：没有本地提交、没有本地回执、没有展示态注入。 */
  function createApiActions(ctx2) {
    return {
      "open-catalog": async function (target) {
        const id = (target && target.dataset && target.dataset.catalog) || "";
        /* W172 忙态统一：open 家族共享 busy key=open（值=catalog id），飞行中家族互禁。 */
        if (!busyStart("open", id, "正在打开 " + id + " · 真实资料集场景加载约需十几秒")) return;
        try {
          const result = await CORE.openCatalog(id);
          toastResult(result, "已打开 · " + id, "打开被拒绝");
          await afterOpen();
        } finally { busyEnd("open"); }
      },
      "open-synthetic": async function () {
        if (!busyStart("open", "synthetic_contract", "正在打开 synthetic_contract · 真实资料集场景加载约需十几秒")) return;
        try {
          const result = await CORE.openCatalog("synthetic_contract");
          toastResult(result, "已打开合成项目", "打开被拒绝");
          await afterOpen();
        } finally { busyEnd("open"); }
      },
      "load-internal": async function () {
        if (!busyStart("open", "internal_pf02", "正在打开 internal_pf02 · 真实资料集场景加载约需十几秒")) return;
        try {
          const result = await CORE.openCatalog("internal_pf02");
          toastResult(result, "已载入内部 PF02 · 单位未核", "载入被拒绝");
          await afterOpen();
        } finally { busyEnd("open"); }
      },
      "import-json": pickImportFile,
      /* W163：便携入口全部走真实 /api/portable/*，页面不生成、不改写包内文件。 */
      "portable-panel": async function () {
        /* W172 静默失败修复（普查 A17）：ok:false 不再无声，仍 openModal+refresh 如实呈现。 */
        const result = await CORE.portableStatus();
        if (!result.ok) toastResult(result, "", "便携状态读取失败");
        openModal("portable");
        refresh(true);
      },
      "portable-export": async function () {
        if (!busyStart("portable-export", undefined, "便携导出已发起 · 构建与自校验中…")) return;
        /* 模态内控件：refresh 不重渲染 modal-root，用直接 DOM 忙态；渲染函数亦按 view.busy 感知。 */
        const saved = domBusy(document.querySelector('[data-testid="portable-export-btn"]'), "导出中…");
        try {
          const result = await CORE.portableExport();
          busyEnd("portable-export");
          domBusyEnd(saved);
          if (result.ok) UI.toast("便携包已导出并重开 · " + (result.receipt.package_id || ""), "ok");
          else UI.toast("便携导出未通过 · " + (result.code || "FAILED"), "err");
          openModal("portable");
          refresh(true);
        } finally { busyEnd("portable-export"); domBusyEnd(saved); }
      },
      "portable-open": async function (target) {
        const pkg = target && target.dataset && target.dataset.package;
        if (!busyStart("portable-open", pkg)) return;
        UI.toast("正在打开便携包 · 全量核哈希", "warn");
        /* 被点按钮「打开中…」，全部候选飞行中禁用（直接 DOM + 渲染感知）。 */
        const saved = [domBusy(target, "打开中…")];
        document.querySelectorAll('[data-action="portable-open"]').forEach(function (el) {
          if (el !== target) saved.push(domBusy(el));
        });
        try {
          const result = await CORE.portableOpen(pkg);
          busyEnd("portable-open");
          domBusyEnd(saved);
          if (result.ok) UI.toast("便携包已打开 · " + result.status, "ok");
          else UI.toast("便携包未通过 · " + (result.status || result.code || "FAILED"), "err");
          openModal("portable");
          refresh(true);
        } finally { busyEnd("portable-open"); domBusyEnd(saved); }
      },
      "portable-relocate": async function () {
        /* W172 W171K-01：飞行 17.7s 无反馈 + 可重复提交。守卫 + 按钮忙态 + 事前 toast。
           禁飞行中整体重渲染 #portable-body（会丢 portable-new-path 输入值），用直接 DOM。 */
        if (!busyStart("portable-relocate", undefined, "重定位已发起 · 全文件哈希校验约需十几秒，请勿重复点击")) return;
        const saved = [domBusy(document.querySelector('[data-testid="portable-relocate-btn"]'), "重定位中…", "全文件哈希校验约需十几秒，请勿重复点击")];
        document.querySelectorAll('[data-action="portable-relocate-fill"]').forEach(function (el) { saved.push(domBusy(el)); });
        try {
          const detail = S.portableDetail || {};
          const assetEl = document.getElementById("portable-asset-id");
          const pathEl = document.getElementById("portable-new-path");
          const result = await CORE.portableRelocate(detail.package,
            assetEl ? assetEl.value.trim() : "", pathEl ? pathEl.value.trim() : "");
          busyEnd("portable-relocate");
          domBusyEnd(saved);
          if (result.ok) UI.toast("重定位完成 · " + (result.status || "RELOCATED") + " · 全文件哈希校验通过", "ok");
          else UI.toast("重定位被拒绝 · " + (result.code || "REFUSED"), "err");
          openModal("portable");
          refresh(true);
        } finally { busyEnd("portable-relocate"); domBusyEnd(saved); }
      },
      "portable-relocate-fill": function (target) {
        const assetEl = document.getElementById("portable-asset-id");
        if (assetEl && target && target.dataset) assetEl.value = target.dataset.asset || "";
        const pathEl = document.getElementById("portable-new-path");
        if (pathEl) pathEl.focus();
      },
      "portable-refresh": async function () {
        /* W172 静默失败修复（普查 A21）：ok:false 不再无声。 */
        const result = await CORE.portableStatus();
        if (!result.ok) toastResult(result, "", "便携状态读取失败");
        openModal("portable");
        refresh(true);
      },
      "portable-detach": async function () {
        /* W172 静默失败修复（普查 A22）：ok:false 不再无声。 */
        const result = await CORE.portableClose();
        if (!result.ok) toastResult(result, "", "关闭包视图失败");
        openModal("portable");
        refresh(true);
      },
      "close-project": async function () {
        const result = await CORE.closeProject();
        toastResult(result, "项目已关闭", "关闭被拒绝");
        refresh(true);
      },
      "reopen-project": async function (target) {
        /* W169 D03：走稳健恢复（无可解析 id 时按需重新拉取），不再因瞬时内存态为空而卡 v0。 */
        if (view.reopening) return;
        view.reopening = true; refresh(true);
        try {
          const explicit = (target && target.dataset && target.dataset.bundle) || null;
          const result = await CORE.reopenSaved(explicit);
          if (result.ok) { UI.toast("已重开保存版本 · v" + S.project_version, "ok"); await afterOpen(); }
          else { toastResult(result, "", "重开被拒绝"); refresh(true); }
        } finally { view.reopening = false; refresh(true); }
      },
      "reopen": async function () {
        if (view.reopening) return;
        view.reopening = true; refresh(true);
        try {
          const result = await CORE.reopenSaved(null);
          if (result.ok) { UI.toast("已重开保存版本 · v" + S.project_version, "ok"); await afterOpen(); }
          else { toastResult(result, "", "重开被拒绝"); refresh(true); }
        } finally { view.reopening = false; refresh(true); }
      },
      "migrate": async function () {
        if (!busyStart("migrate")) return;
        try {
          const beforeHash = S.content_hash || "";
          const beforeReadOnly = S.read_only_legacy;
          const result = await CORE.migrate();
          if (!result.ok) { toastResult(result, "", "迁移被拒绝"); refresh(true); return; }
          UI.toast("迁移完成 · 现在可以编辑与保存了（read_only_legacy " + beforeReadOnly + " → " + S.read_only_legacy
            + " · hash " + beforeHash.slice(0, 12) + "… → " + (S.content_hash || "").slice(0, 12) + "…）", "ok");
          refresh(true);
        } finally { busyEnd("migrate"); }
      },
      "preview": async function () {
        const spec = ctx2.getPreviewSpec();
        if (!spec.targets.length) { UI.toast("预览未发起 · 目标为空", "warn"); return; }
        /* W172 忙态统一：守卫放在预检之后；Ctrl+Enter 经覆盖层同路受守卫。 */
        if (!busyStart("preview")) return;
        try {
          const result = await CORE.preview({ action: spec.action, targets: spec.targets, payload: spec.payload, base_version: S.project_version });
          toastResult(result, "预览已生成 · 服务确认项目不变", previewFailText(result));
          refresh(true);
        } finally { busyEnd("preview"); }
      },
      "commit": async function () {
        if (!busyStart("commit")) return;
        try {
          const result = await CORE.commit();
          toastResult(result, "已提交 · v" + S.project_version, "提交被拒绝");
          refresh(true);
        } finally { busyEnd("commit"); }
      },
      "cancel-preview": async function () {
        const result = await CORE.cancelPreview();
        toastResult(result, "预览已取消 · 项目不变", "取消被拒绝");
        refresh(true);
      },
      "save": async function () {
        /* W169：保存窗口在真实 ROI 上可到 10 秒以上（W168-D02：期间无任何反馈，
           操作者以为失败）。这里只加"进行中"状态：按钮文案与禁用，不改保存语义。 */
        if (view.saving) return;
        view.saving = true;
        refresh(true);
        try {
          const result = await CORE.saveProject();
          if (result.ok) {
            UI.toast("保存成功 · v" + result.receipt.project_version + " · verified=" + result.receipt.verified + " · " + result.receipt.bundle_id, "ok");
          } else {
            toastResult(result, "", "保存失败");
          }
        } finally {
          view.saving = false;
          refresh(true);
        }
      },
      "retry-save": async function () {
        if (view.saving) return;
        view.saving = true;
        refresh(true);
        try {
          const result = await CORE.saveProject();
          if (result.ok) UI.toast("重试保存成功 · v" + result.receipt.project_version, "ok");
          else toastResult(result, "", "仍失败");
        } finally {
          view.saving = false;
          refresh(true);
        }
      },
      "export": async function () {
        if (!busyStart("export", undefined, "导出已发起 · 新进程读回核对中…")) return;
        try {
          const result = await CORE.exportProject();
          if (result.ok) {
            UI.toast("导出读回 PASS · " + result.receipt.enumeration_count + " 项 · 版本一致=" + result.receipt.current, "ok");
          } else {
            UI.toast("导出读回未通过 · " + (result.code || "READBACK_FAILED"), "err");
          }
          refresh(true);
        } finally { busyEnd("export"); }
      },
      "download-project": function () { downloadExportFile("project.json"); },
      "download-manifest": function () { downloadExportFile("manifest.json"); },
      "download-readback": function () { downloadExportFile("readback.json"); },
      "solve": async function () {
        const result = await CORE.startSolve({ scope: "local" });
        toastResult(result, "求解已提交 · 独立进程", "求解未启动");
        refresh(true);
      },
      "solve-full": async function () {
        const result = await CORE.startSolve({ scope: "full" });
        toastResult(result, "全域求解已提交", "求解未启动");
        refresh(true);
      },
      "learned": async function () {
        const result = await CORE.startSolve({ learned: true, scope: "local" });
        toastResult(result, "学习模式已提交", "学习模式未启动");
        refresh(true);
      },
      "cancel-job": async function () {
        const result = await CORE.cancelJob();
        toastResult(result, "任务已取消", "取消被拒绝");
        refresh(true);
      },
      "undo": async function () {
        /* W170 W169C-03：撤销在服务层要重算证据（真实 ROI 约 10 秒）。点击即给忙态+提示，
           期间禁用撤销/重做防重复点击触发版本冲突；完成后由 adoptState 的早期 emit 即时刷新控件。 */
        if (view.undoRedoBusy) return;
        view.undoRedoBusy = true; refresh(true);
        UI.toast("撤销中… · 真实资料集证据重算约需 10 秒，完成后版本号 +1、重做按钮启用", "info");
        try {
          const result = await CORE.undoRedo("undo");
          toastResult(result, "已撤销 · v" + S.project_version + " · 撤销发布新版本（不回退版本号），可重做", "撤销被拒绝");
        } finally { view.undoRedoBusy = false; refresh(true); }
      },
      "redo": async function () {
        if (view.undoRedoBusy) return;
        view.undoRedoBusy = true; refresh(true);
        UI.toast("重做中… · 真实资料集证据重算约需 10 秒，完成后版本号 +1", "info");
        try {
          const result = await CORE.undoRedo("redo");
          toastResult(result, "已重做 · v" + S.project_version + " · 恢复最近一次撤销", "重做被拒绝");
        } finally { view.undoRedoBusy = false; refresh(true); }
      },
      "review-mark": async function () {
        if (!S.selectedId) return;
        const result = await CORE.preview({ action: "SET_INTERPRETATION", targets: [S.selectedId], payload: { interpretation_status: "human_reviewed" }, base_version: S.project_version });
        toastResult(result, "已生成评审预览", previewFailText(result));
        refresh(true);
      },
      "review-unknown": async function () {
        if (!S.selectedId) return;
        const result = await CORE.preview({ action: "SET_UNKNOWN", targets: [S.selectedId], payload: { unknown_reason: "OCCLUDED" }, base_version: S.project_version });
        toastResult(result, "已生成 unknown 预览", previewFailText(result));
        refresh(true);
      },
      "view-log": function () { ctx2.focusLog(); },
      "log-export": function () { UI.download("operation_log.json", JSON.stringify(S.log, null, 2)); UI.toast("操作日志已导出", "ok"); },
      "fit-view": function () { if (ctx2.scene) ctx2.scene.fit(); },
      "shot-view": function () {
        if (!ctx2.scene) return;
        const url = ctx2.scene.screenshot();
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "scene_evidence.png";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        UI.toast("三维截图已保存", "ok");
      }
    };
  }



  const ACTIONS = [
    ["SET_INTERPRETATION", "修改解释状态"],
    ["SET_RELATION", "修改关系判定"],
    ["SET_UNKNOWN", "标记 unknown"],
    ["EDIT_BOUNDARY", "编辑边界环"],
    ["SELECT_SUPPORT", "选择支持索引"],
    ["SPLIT_SURFACE", "拆分面/片"],
    ["MERGE_SURFACE", "合并面"],
    ["PROTECT_REGION", "新增保护区"],
    ["UNPROTECT_REGION", "解除保护区"],
    ["DELETE_OBJECT", "删除对象"],
  ];
  /* SPLIT_SURFACE 必须带载荷：fragment 目标给 new_indices（当前支持的子集，两侧各 >=3 点），
     surface 目标给 member_ids_for_new。它曾误入 NO_PAYLOAD，导致 UI 不渲染 #ed-payload 且载荷被重置为空对象。 */
  const NO_PAYLOAD = new Set(["MERGE_SURFACE", "UNPROTECT_REGION", "DELETE_OBJECT"]);

  function payloadObj() {
    try { return JSON.parse(view.edit.payloadText || "{}"); } catch { return null; }
  }

  /* 拆分骨架：只为"用户还没动过载荷"时提供一个可编辑起点，绝不改写用户已输入的内容。 */
  function splitSkeleton(obj) {
    if (obj && obj.kind === "fragment" && Array.isArray(obj.support_indices) && obj.support_indices.length >= 6) {
      const half = Math.floor(obj.support_indices.length / 2);
      return { new_indices: obj.support_indices.slice(0, half) };
    }
    if (obj && obj.kind === "surface") return { member_ids_for_new: [] };
    return {};
  }

  function payloadTemplate(action, obj) {
    if (action === "SPLIT_SURFACE") return splitSkeleton(obj);
    return CORE.defaultPayload(action, obj) || {};
  }

  /* ---------------- 顶栏 ---------------- */
  function topbarHTML() {
    const open = S.project_open;
    const p = S.project;
    /* 预览态终局动作是「提交新版本」，顶部保存降为纸面按钮：
       同一时刻全页只允许一个实心按钮，否则两个终点互相抢注意力。 */
    const previewing = !!(S.preview && CORE.previewIsLive());
    const readOnly = !!S.read_only_legacy;
    const saving = !!view.saving;
    const hasBundle = !!lastBundleId();
    const chips = [];
    if (open) {
      chips.push(S.source === "internal" ? badge("warn", "内部数据") : badge("neutral", "合成契约"));
      if (p.unit !== "m") chips.push(badge("warn", COPY.unit_unverified));
      if (p.real_quality === "undefined") chips.push(badge("unk", REAL_QUALITY_UNDEFINED));
    }
    const chip = open
      ? `<span class="project-kicker">当前项目</span><span class="project-line"><span class="pid">${esc(p.project_id)}</span><span class="vtag">v${S.project_version}</span><span class="phash">${esc((S.content_hash || "").slice(0, 8))}</span>${icon("expand")}</span>`
      : `<span class="project-kicker">项目</span><span class="project-line"><span class="pid">未打开项目</span>${icon("expand")}</span>`;
    return `
      <div class="mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M4 7.2 12 3l8 4.2-8 4.2z" fill="currentColor" fill-opacity=".08"/>
          <path d="m4 12 8 4.2 8-4.2M4 16.8 12 21l8-4.2"/>
        </svg>
        <span class="brand-copy">
          <span class="wordmark">STRATA</span>
          <span class="mark-sub">结构面工作台</span>
        </span>
      </div>
      <span class="mark-sep"></span>
      <button class="proj-chip ${open ? "" : "closed"}" data-action="project-menu" data-testid="project-menu" aria-haspopup="menu" title="项目动作与已保存版本">${chip}</button>
      <span class="spacer"></span>
      <span class="chips" id="e-chips" data-testid="boundary-chips">${chips.join("")}</span>
        <button class="btn" data-action="reopen" data-testid="reopen" ${view.reopening ? "disabled" : ""} title="${view.reopening ? "正在重开已保存版本…" : hasBundle ? "重开最近一次保存的版本（POST /api/projects/open-saved）" : "重开已保存版本：若暂未读到，点击会按需向服务重新拉取已保存清单（不再永久禁用）"}">${view.reopening ? "重开中…" : "重开"}</button>
        <button class="btn${readOnly ? " primary" : ""}" data-action="migrate" data-testid="migrate" ${view.busy.migrate ? "disabled" : ""} title="${view.busy.migrate ? "迁移进行中…" : readOnly ? "当前是只读 v1：点此迁移为可编辑副本（POST /api/projects/migrate）" : "显式迁移只读 v1 副本（POST /api/projects/migrate）"}">${view.busy.migrate ? "迁移中…" : readOnly ? "迁移（先点这里）" : "迁移"}</button>
      <span class="top-actions">
        <button class="btn ${previewing || saving ? "" : "primary"}" data-action="save" data-testid="save" ${open && !saving ? "" : "disabled"} title="${saving ? "正在保存：真实资料集需要十几秒，请勿关闭页面" : previewing ? "预览待提交 · 先提交或取消预览再保存" : esc(COPY.save)}">${icon("save")}${saving ? "保存中…" : esc(COPY.save)}</button>
        <button class="btn" data-action="export" data-testid="export" title="${view.busy.export ? "导出进行中 · 新进程读回核对中…" : esc(COPY.export)}" ${open && !view.busy.export ? "" : "disabled"}>${icon("download")}${view.busy.export ? "导出中…" : "导出"}</button>
        <button class="btn" data-action="portable-panel" data-testid="portable-panel-open" title="W154 便携包：导出 / 打开 / 重定位 / 状态">${icon("folder")}便携</button>
      </span>`;
  }

  /* ---------------- 主状态行（唯一权威；动作只在命令栏/横幅出现一次） ---------------- */
  function statusrowHTML() {
    if (S.saveError) {
      const e = S.saveError;
      return `<div class="sl err" data-testid="primary-status" role="alert">
        <i class="sl-dot"></i>
        <div class="sl-main" data-testid="save-error-banner">
          <div class="sl-line1"><span class="sl-text">${esc(COPY.save_failed_title)}</span><span class="code">${esc(e.code)}</span></div>
          <div class="sl-reason">${esc(e.message)} ${esc(COPY.save_failed_rollback)}</div>
        </div>
        <span class="sl-actions">
          <button class="btn sm danger" data-action="retry-save" data-testid="retry-save-btn">${icon("refresh")}${esc(COPY.retry_save)}</button>
          <button class="btn sm" data-action="view-log" data-testid="view-log-btn">${icon("list")}${esc(COPY.view_log)}</button>
        </span>
      </div>`;
    }
    const st = UI.statusOf(S);
    /* W170 W169C-01：项目打开且有证据计数时，状态行下常驻一行中文当前性图例。 */
    const legend = (S.project_open && S.evidence_counts) ? evidenceLegendHTML() : "";
    return `<div class="sl ${st.tone}" data-testid="primary-status" role="status">
      <i class="sl-dot"></i>
      <span class="sl-text">${esc(st.text)}</span>
      <span class="sl-next">${esc(st.next)}</span>
    </div>` + legend;
  }

  /* ---------------- 左栏 ---------------- */
  function ledgerHTML() {
    const open = S.project_open;
    const objs = open ? S.project.objects : [];
    const tabs = [["all", "全部"], ["surface", "面"], ["fragment", "片"], ["relation", "关系"], ["protected", "保护"]]
      .map(([id, label]) => `<button role="tab" aria-selected="${view.filter === id}" data-action="filter" data-filter="${id}">${label}</button>`).join("");
    let list = "";
    if (!open) {
      list = `<div class="ledger-empty">项目已关闭。<br>重开或新建后，对象清单在此列示。</div>`;
    } else {
      const q = view.query.trim().toLowerCase();
      const groups = [["surface", "面"], ["fragment", "片"], ["relation", "关系"], ["protected", "保护区"]];
      let shown = 0;
      for (const [kind, label] of groups) {
        if (view.filter !== "all" && view.filter !== kind) continue;
        const rows = objs.filter((o) => o.kind === kind && (!q || o.id.toLowerCase().includes(q)));
        if (!rows.length) continue;
        shown += rows.length;
        list += `<div class="group-head"><span class="micro">${label}</span><span class="n">${rows.length}</span></div>`;
        list += rows.map((o) => {
          const rv = reviewOf(o);
          const tone = UI.reviewTone[rv] || "unk";
          /* 版本号只在 >1 时显示：全为 v1 时每行多一条副行，纯噪声，也把行高撑到必须滚动 */
          const ver = o.version > 1 ? `<span class="row-ver">v${o.version}</span>` : "";
          return `<button class="row" role="option" aria-selected="${S.selectedId === o.id}" data-action="select-object" data-id="${esc(o.id)}">
            <span class="glyph">${GLYPHS[o.kind] || ""}</span>
            <span class="row-id">${esc(o.id)}</span>
            <span class="row-state"><i class="dot" style="background:${TONE_VAR[tone]}"></i>${esc(UI.reviewLabel[rv] || rv)}${ver}</span>
          </button>`;
        }).join("");
      }
      if (!shown) list = `<div class="ledger-empty">无匹配对象。<br>调整检索词或筛选。</div>`;
    }
    return `
      <div class="ledger-head">
        <div class="ledger-title"><span class="micro">对象清单</span><span class="spacer"></span><span class="n">${objs.length}</span><button class="icon-btn ledger-close" data-action="ledger-toggle" data-testid="ledger-close" title="收起清单" aria-label="收起清单">${icon("close")}</button></div>
        <div class="search">${icon("search")}<input id="ledger-search" type="text" value="${esc(view.query)}" placeholder="按稳定 ID 检索（/）" autocomplete="off"></div>
      </div>
      <div class="tabs" role="tablist" aria-label="对象类型">${tabs}</div>
      <div class="ledger-list" id="ledger-list" role="listbox">${list}</div>
      <div class="ledger-foot">${esc(COPY.boundary_note).split("；").map((s) => `<b>${s}${s.endsWith("。") ? "" : "；"}</b>`).join("")}</div>`;
  }

  /* ---------------- 视口覆盖层 ---------------- */
  function captionHTML() {
    if (!S.project_open) return `<span class="micro">图版</span><div class="l2">图纸待载入</div>`;
    const c = S.project.counts || {};
    const src = S.source === "internal" ? "内部域 · 单位未核" : "合成契约域";
    return `<span class="micro">图版</span><div class="l2">${src}　${c.surfaces ?? 0} 面 · ${c.relations ?? 0} 关系 · ${c.display_points ?? 0} 显示点</div>`;
  }
  function toolsHTML() {
    return `
      <button class="icon-btn" data-action="fit-view" data-testid="fit-view" title="${esc(COPY.fit)}" aria-label="${esc(COPY.fit)}">${icon("fit")}</button>
      <button class="icon-btn" data-action="shot-view" data-testid="shot-view" title="${esc(COPY.shot)}" aria-label="${esc(COPY.shot)}">${icon("camera")}</button>
      <span class="sep"></span>
      <button class="icon-btn" data-action="layers-btn" data-testid="layers-btn" title="图层" aria-label="图层">${icon("layers")}</button>
      <button class="icon-btn inspector-toggle" data-action="inspector-toggle" data-testid="inspector-toggle" title="对象与证据" aria-label="对象与证据">${icon("list")}</button>`;
  }
  function legendHTML() {
    /* 图例只列"当前可见"的图层：关掉的图层仍留在图例里，会让人以为画面漏画。
       首帧 scene 可能还没挂上，此时取默认值（与 plate_c.js 的 layers 初始值一致），
       否则保护区会"清单里有、图例和图面都没有"。 */
    const L = (scene && scene.getLayers()) || { surfaces: true, fragments: true, points: true, relations: true, protected: false, grid: false };
    /* 状态族与图层族分开：前者读的是"这个面是什么状态"，后者读的是"画面上有什么"，
       混成一排时读者要在两种语义之间来回切。 */
    let out = `
      <span class="lg"><i class="sw"></i>面 · 已复核</span>
      <span class="lg"><i class="sw warn"></i>待复核</span>
      <span class="lg"><i class="sw err"></i>冲突</span>
      <span class="lg lg-secondary"><i class="sw yes"></i>贯通</span>
      <span class="lg"><i class="sw dash"></i>unknown</span>`;
    const layers = [];
    if (L.fragments !== false) layers.push(`<span class="lg lg-layer"><i class="sw thin"></i>碎片</span>`);
    if (L.points !== false) layers.push(`<span class="lg lg-layer"><i class="pt"></i>支持点</span>`);
    if (L.protected) layers.push(`<span class="lg lg-layer"><i class="sw wash"></i>保护区</span>`);
    if (layers.length) out += `<i class="lg-sep"></i>` + layers.join("");
    return out;
  }
  /* 比例尺由渲染器报"多少像素 = 多少米"，在 DOM 图例栏里描线。
     左半实心右半留白是制图惯例，两端数字与线端严格对齐。 */
  function paintScale(si) {
    const el = $("#vp-scale");
    if (!el || !si || !isFinite(si.px)) return;
    const inner = Math.max(24, si.px - 2);
    const half = Math.round(inner / 2);
    el.innerHTML = `<span class="sc-lab">0</span>`
      + `<span class="sc-bar" style="width:${si.px}px"><i class="dark" style="width:${half}px"></i><i class="open" style="width:${inner - half}px"></i></span>`
      + `<span class="sc-lab">${esc(si.label)}</span>`;
  }
  /* W172 OBS-D：已保存版本四态分流。state_loaded 只在 applyState 成功落地时置 true；
     ready 在 state 三次重试全失败后仍置 true，故 !state_loaded && ready = 读取失败（非真空）。 */
  function libState() {
    if (!S.state_loaded) return S.ready ? "failed" : "loading";
    return S.library.length ? "list" : "empty";
  }
  /* W172 忙态统一：open 家族（open-catalog/open-synthetic/load-internal）与 import 飞行中互禁。 */
  function openFamilyBusy() { return !!(view.busy.open || view.busy.import); }

  /* 空态按扉页排：刊头 → 图版徽记 → 题字 → 细线 → 动作 → 版本目录 */
  function emptyHTML() {
    const libs = S.library.slice().reverse();
    const famBusy = openFamilyBusy();
    const libSt = libState();
    const libBody = libSt === "list"
      ? libs.map((b, i) => `<button class="bundle" data-action="reopen-project" data-bundle="${esc(b.bundle_id)}" ${i === 0 ? 'data-testid="reopen-project"' : `data-testid="reopen-project-${i}"`}>
            <span class="bid">${esc(b.bundle_id)}</span><span class="vtag">v${b.project_version}</span>
            <span class="when">${esc(fmtTime(b.saved_at))}</span>${icon("reopen")}
          </button>`).join("")
      : `<div class="empty-hint" style="text-align:left;font-size:11.5px">${libSt === "loading" ? "正在读取已保存版本…" : libSt === "failed" ? "已保存版本读取失败 · 点顶栏「重开」可按需重新拉取" : "暂无已保存版本。"}</div>`;
    return `<div class="empty-card" data-testid="empty-state">
      <div class="empty-kicker">STRATA · WORKBENCH</div>
      <svg class="plate-mark" width="52" height="52" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" aria-hidden="true">
        <rect x="4" y="4" width="32" height="32"/>
        <path d="M9 13h8.5M9 19h8.5M9 25h8.5M22.5 15.5h8.5M22.5 21.5h8.5M22.5 27.5h8.5"/>
        <path d="M20.3 8l-1.3 24" opacity=".5"/>
      </svg>
      <div class="empty-title">${esc(COPY.empty_title)}</div>
      <div class="empty-rule"></div>
      <div class="empty-hint">${esc(COPY.empty_hint)}</div>
      <div class="empty-actions">
        <button class="btn primary" data-action="open-synthetic" data-testid="open-synthetic" ${famBusy ? "disabled" : ""}>${icon("plus")}${view.busy.open === "synthetic_contract" ? "打开中…" : esc(COPY.new_synthetic)}</button>
        <button class="btn ghost" data-action="load-internal" data-testid="load-internal" ${famBusy ? "disabled" : ""}>${icon("folder")}${view.busy.open === "internal_pf02" ? "打开中…" : esc(COPY.load_internal)}</button>
        <button class="btn ghost" data-action="import-json" data-testid="import-json" ${famBusy ? "disabled" : ""}>${icon("import")}${esc(COPY.import_json)}</button>
      </div>
      <div class="empty-lib" data-testid="empty-lib-state" data-state="${libSt}">
        <span class="micro">已保存版本</span>
        ${libBody}
      </div>
    </div>`;
  }

  /* ---------------- 命令栏（事务台 + 底栏合并） ---------------- */
  function cmdbarHTML() {
    /* ≤720px 时左栏收成抽屉，这里给它唯一的入口；宽屏下 display:none，不占位 */
    const lead = `<button class="icon-btn ledger-toggle" data-action="ledger-toggle" data-testid="ledger-toggle" title="对象清单" aria-label="对象清单">${icon("list")}</button>`;
    const tail = `
      <span class="cmd-actions">
        <button class="btn ghost" data-action="log-toggle" data-testid="log-toggle">${icon("list")}操作日志 · ${S.log.length}</button>
      </span>`;
    if (!S.project_open) {
      return `${lead}<span class="cmd-label micro">事务</span><span class="cmd-text">项目未打开 · 写动作停用</span>${tail}`;
    }
    const job = S.jobs[S.jobs.length - 1];
    if (S.activeJobId && job && job.status === "running") {
      const pct = Math.round(job.progress * 100);
      return `${lead}<span class="cmd-label micro">求解任务</span>
        <span class="cmd-text">${esc(job.phase)} · 范围 ${esc(job.scope)} · 完成前不发布候选</span>
        <span class="prog"><span class="prog-fill" style="width:${pct}%"></span></span>
        <span class="pct">${pct}%</span>
        <span class="cmd-actions"><button class="btn sm" data-action="cancel-job" data-testid="cancel-job-btn">${esc(COPY.cancel_job)}</button></span>
        ${tail}`;
    }
    if (S.preview && CORE.previewIsLive()) {
      const d = S.preview.diff || {};
      const n = (a) => (a || []).length;
      return `${lead}<span class="cmd-label micro">编辑事务</span>
        <span class="diffchips">
          <span class="diffchip">改 ${n(d.changed)}</span>
          <span class="diffchip add">增 ${n(d.added)}</span>
          <span class="diffchip del">删 ${n(d.removed)}</span>
        </span>
        <span class="cmd-text">基版本 v${S.preview.base_version} · ${esc(COPY.preview_unchanged)}</span>
        <span class="cmd-actions">
          <button class="btn sm primary" data-action="commit" data-testid="commit" ${view.busy.commit ? "disabled" : ""}>${view.busy.commit ? "提交中…" : esc(COPY.commit)}</button>
          <button class="btn sm ghost" data-action="cancel-preview" data-testid="cancel-preview-btn">${esc(COPY.cancel_preview)}</button>
        </span>
        ${tail}`;
    }
    const failed = job && ["failed", "cancelled", "timeout"].includes(job.status);
    const text = failed
      ? `上一任务已回退 · 候选未发布`
      : S.dirty ? `v${S.project_version} 尚未落盘 · 保存后生成可重开版本` : "选择动作后生成预览 · 项目不变";
    return `${lead}<span class="cmd-label micro">编辑事务</span>
      <span class="cmd-text">${esc(text)}</span>
      <span class="cmd-actions">
        <button class="icon-btn" data-action="undo" data-testid="undo-btn" title="${esc(view.undoRedoBusy ? "撤销/重做进行中…（真实资料集证据重算约需 10 秒，请勿重复点击）" : CORE.canUndo() ? "撤销：回退最近一次已提交的编辑事务，发布新版本（不回退版本号）" : "撤销：当前无可撤销的已提交编辑")}" aria-label="${esc(COPY.undo)}" ${CORE.canUndo() && !view.undoRedoBusy ? "" : "disabled"}>${icon("undo")}</button>
        <button class="icon-btn" data-action="redo" data-testid="redo-btn" title="${esc(view.undoRedoBusy ? "撤销/重做进行中…请等待完成" : CORE.canRedo() ? "重做：恢复最近一次撤销（需上下文未变）" : "重做：仅在执行撤销后可用；新的编辑/提交会清空重做栈")}" aria-label="${esc(COPY.redo)}" ${CORE.canRedo() && !view.undoRedoBusy ? "" : "disabled"}>${icon("redo")}</button>
        <span class="sep"></span>
        <button class="btn sm" data-action="solve" data-testid="solve-btn">${icon("play")}${esc(COPY.solve)}</button>
        <button class="icon-btn" data-action="solve-menu" data-testid="solve-menu" title="求解范围与演练" aria-label="求解范围与演练">${icon("expand")}</button>
      </span>
      <span class="cmd-hint" data-testid="undo-redo-hint">撤销/重做作用于已提交的编辑事务：每次发布新版本、不回退版本号；新编辑或提交会清空重做栈，故提交后「重做」暂禁用，先「撤销」一次即可重做。</span>
      ${tail}`;
  }

  /* ---------------- 右栏 ---------------- */
  function editFieldsHTML(o) {
    const a = view.edit.action;
    const po = payloadObj() || {};
    let payload = "";
    if (a === "SET_INTERPRETATION") {
      payload = `<div class="field"><label class="micro" for="ed-interp">解释状态</label>
        <select id="ed-interp" data-edit="interpretation">
          ${["human_reviewed", "synthetic_contract_case", "needs_field_check"].map((v) => `<option value="${v}" ${po.interpretation_status === v ? "selected" : ""}>${v}</option>`).join("")}
        </select></div>`;
    } else if (a === "SET_RELATION" || a === "SET_UNKNOWN") {
      if (a === "SET_RELATION") {
        payload += `<div class="field"><label class="micro" for="ed-relstate">判定</label>
          <select id="ed-relstate" data-edit="relstate">
            ${["yes", "no", "unknown"].map((v) => `<option value="${v}" ${po.state === v ? "selected" : ""}>${v === "yes" ? "贯通 yes" : v === "no" ? "不贯通 no" : "不可判 unknown"}</option>`).join("")}
          </select></div>`;
      }
      if (a === "SET_UNKNOWN" || po.state === "unknown") {
        payload += `<div class="field"><label class="micro" for="ed-reason">unknown 原因</label>
          <select id="ed-reason" data-edit="relreason">
            ${["OCCLUDED", "INSUFFICIENT_SUPPORT"].map((v) => `<option value="${v}" ${po.unknown_reason === v ? "selected" : ""}>${v}</option>`).join("")}
          </select></div>`;
      }
    } else if (a === "SELECT_SUPPORT") {
      /* W169 F1：默认载荷由 core_api.defaultPayload 从"当前支持索引"派生合法子集，
         不再给空数组（空数组必被后端 BAD_SUPPORT 拒绝）。这里只负责解释字段含义。
         W171 W170C-02：提示语与行为一致——骨架取前一半（非全集）、手改后切对象不覆盖、可显式重填。 */
      const sup = (o && Array.isArray(o.support_indices)) ? o.support_indices : [];
      const sample = sup.length
        ? "本片现有 " + sup.length + " 个支持点，例如 " + esc(sup.slice(0, 6).join(", ")) + (sup.length > 6 ? " …" : "") + "。"
        : "当前没有读到这一片的支持索引；请先在左栏选中一个「片」对象。";
      payload = `<div class="field"><label class="micro" for="ed-payload">载荷 JSON</label>
        <button class="btn sm ghost" data-action="refill-skeleton" data-testid="refill-skeleton" title="按当前选中对象重新生成载荷骨架（会覆盖框内现有内容）">重填骨架</button>
        <textarea id="ed-payload" data-edit="payload" spellcheck="false" placeholder='{"indices": [ 0, 1, 2 ]}'>${esc(view.edit.payloadText)}</textarea>
        <div class="hint">indices = <b>这一片当前支持点的索引列表</b>（从 0 起的整数）：非空、不重复、<b>至少 3 个（必要但不充分）</b>，建议取自该片已有的支持点。选点还须围出<b>非空的有限支持域</b>（凹包 alpha-shape，禁止凸包填充）：过于稀疏/分散的小子集会被拒 FINITE_DOMAIN_EMPTY，建议从骨架起步逐步删减。骨架取<b>当前支持索引的前一半</b>作可编辑起点（不是全集）；手动编辑后切换对象不会覆盖你的输入，点「重填骨架」可按当前选中对象重建。${sample}</div></div>`;
    } else if (a === "SPLIT_SURFACE") {
      /* W171 W170C-05：预览前就给出事前中文警告（单成员面 / 支持点不足），不改后端拆分校验语义。 */
      const memberCount = (o && Array.isArray(o.member_fragment_ids)) ? o.member_fragment_ids.length : 0;
      const supCount = (o && Array.isArray(o.support_indices)) ? o.support_indices.length : 0;
      let preWarn = "";
      if (o && o.kind === "surface" && memberCount < 2) {
        preWarn = `<div class="callout warn" data-testid="split-prewarn"><div class="c-title">${icon("alert")}拆分前提醒</div><div class="c-body">该面只有 ${memberCount} 个成员片：面拆分要求拆后原面至少留 1 片，单成员面必然被后端拒绝（BAD_SPLIT）。请改选一个「片」对象，用 new_indices 拆分。</div></div>`;
      } else if (o && o.kind === "fragment" && supCount < 6) {
        preWarn = `<div class="callout warn" data-testid="split-prewarn"><div class="c-title">${icon("alert")}拆分前提醒</div><div class="c-body">该片只有 ${supCount} 个支持点：拆不出两侧各 ≥3 点（需 ≥6）。请改选支持点更多的片。</div></div>`;
      }
      payload = `${preWarn}<div class="field"><label class="micro" for="ed-payload">载荷 JSON</label>
        <button class="btn sm ghost" data-action="refill-skeleton" data-testid="refill-skeleton" title="按当前选中对象重新生成载荷骨架（会覆盖框内现有内容）">重填骨架</button>
        <textarea id="ed-payload" data-edit="payload" spellcheck="false" placeholder='{"new_indices": [ ... ]}'>${esc(view.edit.payloadText)}</textarea>
        <div class="hint">fragment（片）目标：new_indices（当前支持的子集，拆分后两侧各 >=3 点）；surface（面）目标：member_ids_for_new（拆后原面须留至少 1 片）。骨架仅在未手动编辑时自动给出，「重填骨架」可按当前对象重建。载荷原样发送，页面不改写。</div></div>`;
    } else if (!NO_PAYLOAD.has(a)) {
      payload = `<div class="field"><label class="micro" for="ed-payload">载荷 JSON</label>
        <textarea id="ed-payload" data-edit="payload" spellcheck="false">${esc(view.edit.payloadText)}</textarea></div>`;
    } else {
      payload = `<div class="field"><div class="hint">该动作无需载荷；目标以首个 ID 为主对象。</div></div>`;
    }
    return `
      <div class="field"><label class="micro" for="ed-action">动作</label>
        <select id="ed-action" data-edit="action">${ACTIONS.map(([v, l]) => `<option value="${v}" ${a === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      <div class="field"><label class="micro" for="ed-targets">目标稳定 ID</label>
        <input type="text" id="ed-targets" data-edit="targets" value="${esc(view.edit.targets)}" placeholder="${esc(S.selectedId || "")}" autocomplete="off" spellcheck="false">
        <div class="hint">留空取当前选中对象；逗号分隔多选。</div></div>
      ${payload}`;
  }

  /* 右栏三段：题头常驻、证据可滚、主操作常驻。scroll 里只放正文。 */
  function insHeadHTML() {
    return `<div class="ins-headrow"><span class="micro">对象与证据</span><button class="icon-btn inspector-close" data-action="inspector-toggle" data-testid="inspector-close" title="收起证据栏" aria-label="收起证据栏">${icon("close")}</button></div>`;
  }
  function insDockHTML() {
    if (!S.project_open || !S.selectedId) return "";
    const o = CORE.getObject(S.selectedId);
    if (!o) return "";
    return `<div class="ins-actions"><button class="btn block" data-action="preview" data-testid="preview" ${view.busy.preview ? "disabled" : ""} title="${view.busy.preview ? "预览进行中 · 真实资料集约需 10 秒，请勿重复点击" : ""}">${icon("eye")}${view.busy.preview ? "预览中…" : esc(COPY.preview)}</button></div>`;
  }
  function inspectorHTML() {
    /* 空态分两种：没有项目时让用户"去清单里拾取"是空指令 —— 清单里也没有东西 */
    if (!S.project_open) {
      return `<div class="ins-empty">项目未打开。<br>重开已保存版本或新建项目后，对象的证据在此列示。</div>`;
    }
    if (!S.selectedId) {
      return `<div class="ins-empty">未选择对象。<br>从清单或图版中拾取一个稳定 ID。</div>`;
    }
    const o = CORE.getObject(S.selectedId);
    const ev = CORE.getEvidence(S.selectedId);
    if (!o || !ev) return `<div class="ins-empty">对象不存在于当前版本。</div>`;
    const p = S.project;
    const rv = reviewOf(o);
    /* F1：关系对象的 /api/evidence 不带 source_files（证据由两端对象派生）。
       这里必须容错，否则解引用 undefined 会抛 TypeError，整次右栏渲染中断。 */
    const srcFiles = Array.isArray(ev.source_files) ? ev.source_files : [];
    const src = srcFiles[0] || null;
    const stateTxt = o.kind === "relation" ? `${o.state}${o.state === "unknown" ? " · " + (o.unknown_reason || "") : ""}` : (o.interpretation_status || "—");
    const saveR = S.saveReceipt;
    const expR = S.exportReceipt;
    /* 回执置顶：事务结果是最新事件，紧随对象身份；对象头永不被裁 */
    const saveReceiptHTML = saveR ? `<div class="receipt-block" data-testid="save-receipt">
        <div class="receipt-head">${icon("okc")}保存回执<span class="spacer"></span><span class="vtag">v${saveR.project_version}</span></div>
        <dl class="dl">
          <dt>文件数</dt><dd class="mono">${saveR.file_count}</dd>
          <dt>哈希</dt><dd>${hashline(saveR.content_hash, "save-receipt-hash")}</dd>
          <dt>时刻</dt><dd class="mono ellipsis" style="font-size:11px" title="${esc(saveR.at)}">${esc(saveR.at)}</dd>
        </dl>
      </div>` : "";
    const expReceiptHTML = expR ? `<div class="receipt-block" data-testid="export-receipt">
        <div class="receipt-head">${icon(expR.verified && expR.current ? "okc" : "alert")}导出读回 · ${expR.verified && expR.current ? "PASS" : "FAIL"}<span class="spacer"></span>${badge(expR.verified && expR.current ? "ok" : "err", expR.enumeration_count + " 项")}</div>
        <dl class="dl">
          <dt>版本一致</dt><dd class="mono">${expR.current}</dd>
          <dt>export_id</dt><dd>${hashline(expR.export_id, "export-id")}</dd>
          <dt>导出目录（在盘路径）</dt><dd data-testid="export-dir-cell">${expR.export_dir ? hashline(expR.export_dir, "export-dir") : '<span class="mono">—</span>'}</dd>
        </dl>
        <div class="btn-row">
          <button class="btn sm" data-action="download-project" data-testid="download-project">${icon("download")}project.json</button>
          <button class="btn sm" data-action="download-manifest" data-testid="download-manifest">${icon("download")}manifest.json</button>
          <button class="btn sm" data-action="download-readback" data-testid="download-readback">${icon("download")}readback.json</button>
        </div>
        <div class="btn-row">
          <button class="btn sm ghost" data-action="export-receipt-open" data-testid="export-receipt-open">${icon("list")}查看 ${expR.enumeration_count} 项核对清单</button>
        </div>
      </div>` : "";
    return `
      <div class="ins-idrow" data-value="${esc(o.id)}">
        <span class="ins-id" data-testid="object-id">${esc(o.id)}</span>
        <button class="icon-btn" data-action="copy" title="复制稳定 ID" aria-label="复制稳定 ID">${icon("copy")}</button>
      </div>
      <div class="ins-sub">${badge("neutral", UI.kindLabel[o.kind] || o.kind)}<span class="vtag">v${o.version}</span>${badge(UI.reviewTone[rv] || "unk", UI.reviewLabel[rv] || rv)}</div>
      ${saveReceiptHTML}${expReceiptHTML}${(saveR || expR) ? '<hr class="hr">' : ""}
      <dl class="dl">
        <dt>解释 / 判定</dt><dd class="mono wrap-any">${esc(stateTxt)}</dd>
        ${o.member_fragment_ids ? `<dt>成员片</dt><dd class="mono wrap-any">${esc(o.member_fragment_ids.join(", ") || "—")}</dd>` : ""}
        ${o.from ? `<dt>两端</dt><dd class="mono">${esc(o.from)} → ${esc(o.to)}</dd>` : ""}
        <dt>支持索引</dt><dd>${UI.supportIndicesHTML(o.support_indices, "support-indices")}</dd>
      </dl>
      <hr class="hr">
      <div class="block-title"><span class="micro">支持证据</span></div>
      <dl class="dl">
        <dt>单位</dt><dd>${p.unit !== "m" ? `${badge("warn", COPY.unit_unverified)}<span class="mono ellipsis" style="display:block;margin-top:3px" title="${esc(p.unit)}">${esc(p.unit)}</span>` : `<span class="unit">${esc(p.unit)}</span>`}</dd>
        <dt>坐标</dt><dd class="mono ellipsis" title="${esc(p.coordinate_frame)}">${esc(p.coordinate_frame)}</dd>
        <dt>许可</dt><dd class="mono ellipsis" title="${esc(p.license)}">${esc(p.license)}</dd>
        <dt>证据政策</dt><dd class="mono policy" title="${esc(p.evidence_policy)}">${esc(p.evidence_policy).replace(/_/g, "_<wbr>")}</dd>
        ${src
          ? `<dt>来源</dt><dd class="mono ellipsis" title="${esc(src.path)}">${esc(src.path)}</dd>
        <dt>来源 SHA</dt><dd class="sha-row">${hashline(src.sha256, "source-sha")}<span class="dim mono">${UI.fmtBytes(src.byte_count)}</span></dd>`
          : `<dt>来源</dt><dd class="mono" data-testid="source-missing">${o.kind === "relation" ? "无来源文件（由两端对象派生）" : "无来源文件（服务未返回 source_files）"}</dd>`}
        <dt>真实质量</dt><dd>${p.real_quality === "undefined" ? badge("unk", REAL_QUALITY_UNDEFINED) : `<span class="mono wrap-any">${esc(p.real_quality)}</span>`}</dd>
      </dl>
      ${o.conflict_note ? `<div class="callout err"><div class="c-title">${icon("alert")}边界解释冲突</div><div class="c-body">${esc(o.conflict_note)}</div></div>` : ""}
      ${o.kind === "relation" && o.state === "unknown" ? `<div class="callout unk"><div class="c-title">${icon("question")}unknown · ${esc(o.unknown_reason || "")}</div><div class="c-body">${esc(o.note || "")}</div></div>` : ""}
      <hr class="hr">
      <div class="block-title"><span class="micro">编辑事务</span></div>
      ${editFieldsHTML(o)}`;
  }

  /* ---------------- 日志 ---------------- */
  function logHTML() {
    const rows = S.log.map((l) => `<div class="log-row ${l.level}">
      <span class="t">${esc(String(l.measured_at).slice(11, 19))}</span>
      <i class="dot"></i>
      <span class="log-op" title="${esc(l.operation)}">${esc(l.operation)}</span>
      <span class="detail" title="${esc(l.expected)} → ${esc(l.actual)}">${esc(l.expected)} → ${esc(l.actual)}</span>
    </div>`).join("");
    return `
      <div class="log-head">
        <span class="micro">操作日志</span><span class="n">${S.log.length}</span>
        <span class="spacer"></span>
        <span class="kbd-hints" title="Ctrl S 保存 · Ctrl Enter 预览 · / 检索 · Esc 取消"><span class="kbd">Ctrl S</span><span class="kbd">Ctrl ↵</span><span class="kbd">/</span><span class="kbd">Esc</span></span>
        <button class="btn ghost sm" data-action="log-export" data-testid="log-export">${icon("download")}导出 JSON</button>
        <button class="icon-btn" data-action="log-toggle" title="收起日志" aria-label="收起日志">${icon("close")}</button>
      </div>
      <div class="log-body" id="log-body">${rows || `<div class="log-empty">暂无操作记录。</div>`}</div>`;
  }

  /* ---------------- 菜单浮层 ---------------- */
  function menuHTML(kind) {
    if (kind === "project") {
      const libs = S.library.slice().reverse();
      const famBusy = openFamilyBusy();
      const libSt = libState();
      const libBody = libSt === "list"
        ? libs.map((b) => `<button class="bundle" data-action="reopen-project" data-bundle="${esc(b.bundle_id)}">
            <span class="bid">${esc(b.bundle_id)}</span><span class="vtag">v${b.project_version}</span><span class="when">${esc(fmtTime(b.saved_at))}</span>
          </button>`).join("")
        : `<div class="pop-label dim" style="font-size:11px">${libSt === "loading" ? "读取中…" : libSt === "failed" ? "读取失败 · 点「重开」重试" : "暂无"}</div>`;
      return `
        <button class="pop-item" data-action="open-synthetic" data-testid="menu-open-synthetic" ${famBusy ? "disabled" : ""}>${icon("plus")}${view.busy.open === "synthetic_contract" ? "打开中…" : esc(COPY.new_synthetic)}</button>
        <button class="pop-item" data-action="load-internal" data-testid="menu-load-internal" ${famBusy ? "disabled" : ""}>${icon("folder")}${view.busy.open === "internal_pf02" ? "打开中…" : esc(COPY.load_internal)}</button>
        <button class="pop-item" data-action="import-json" data-testid="menu-import-json" ${famBusy ? "disabled" : ""}>${icon("import")}${esc(COPY.import_json)}</button>
        <button class="pop-item" data-action="portable-panel" data-testid="menu-portable-panel">${icon("folder")}W154 便携包 · 导出 / 打开 / 重定位</button>
        <div class="pop-sep"></div>
        <button class="pop-item danger" data-action="close-project" data-testid="menu-close-project" ${S.project_open ? "" : "disabled"}>${icon("close")}${esc(COPY.close)}</button>
        <div class="pop-sep"></div>
        <div data-testid="menu-lib-state" data-state="${libSt}">
        <div class="pop-label micro">已保存版本</div>
        ${libBody}
        </div>`;
    }
    if (kind === "solve") {
      const dis = S.project_open ? "" : "disabled";
      return `
        <button class="pop-item" data-action="solve" data-testid="menu-solve-local" ${dis}>${icon("play")}规则求解 · 局部</button>
        <button class="pop-item" data-action="solve-full" data-testid="menu-solve-full" ${dis}>${icon("play")}规则求解 · 全域</button>
        <button class="pop-item" data-action="learned" data-testid="menu-learned" ${dis}>${icon("spark")}${esc(COPY.learned)}</button>
        <div class="pop-sep"></div>
        <div class="pop-label micro">可靠性演练</div>
        <button class="pop-item" data-action="drill-save-interrupt" data-testid="menu-drill-save" ${dis}>${icon("alert")}保存中断（负向）</button>
        <button class="pop-item" data-action="drill-resource" data-testid="menu-drill-resource" ${dis}>${icon("alert")}资源限制（负向）</button>
        <button class="pop-item" data-action="drill-hang" data-testid="menu-drill-hang" ${dis}>${icon("clock")}挂起超时（负向）</button>`;
    }
    const layers = scene ? scene.getLayers() : {};
    const names = { surfaces: "结构面", fragments: "碎片", points: "支持点", relations: "关系", protected: "保护区", grid: "底面网格" };
    return Object.keys(names).map((k) => `<label class="pop-check"><input type="checkbox" data-layer="${k}" ${layers[k] ? "checked" : ""}>${names[k]}</label>`).join("");
  }
  function openMenu(kind, anchor) {
    view.menu = kind;
    const root = $("#popover-root");
    root.innerHTML = `<div class="pop" role="menu">${menuHTML(kind)}</div>`;
    const pop = root.firstElementChild;
    const r = anchor.getBoundingClientRect();
    const pw = pop.offsetWidth, ph = pop.offsetHeight;
    let left = Math.min(Math.max(8, r.left), window.innerWidth - pw - 8);
    let top = r.bottom + 6;
    if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - ph - 6);
    pop.style.left = left + "px";
    pop.style.top = top + "px";
  }
  function closeMenu() {
    view.menu = null;
    $("#popover-root").innerHTML = "";
  }

  /* ---------------- W163 便携包面板 ----------------
     只显示服务端 open_package / portable_export 的读数：包状态、资产逐件状态、
     证据绑定与当前性、警告原文。页面不推断、不补造“可用”状态。 */
  function portableTone(status) {
    if (status === "FULL" || status === "RELOCATED") return "ok";
    if (status === "FULL_WITH_WARNINGS" || status === "RESTRICTED" || status === "NEEDS_RECOMPUTE") return "warn";
    if (status === "REPLACED" || status === "CORRUPT") return "err";
    return "unk";
  }
  function portablePanelHTML() {
    const st = S.portableStatus || {};
    const w154 = st.w154 || {};
    const detail = S.portableDetail || (st.current || null);
    const receipt = S.portableReceipt;
    const status = detail ? detail.status : null;
    const tone = portableTone(status);
    const project = detail ? (detail.project || {}) : {};
    const evidence = detail ? (detail.evidence || {}) : {};
    const assets = detail ? (detail.assets || {}) : {};
    const items = assets.items || [];
    const warnings = (detail && detail.warnings) || [];
    const byStatus = assets.by_status || {};
    const missing = items.filter(function (a) { return a.status === "MISSING" || a.status === "REPLACED"; })[0];
    const matrix = w154.matrix || {};
    const w154Block = `<div class="pt-sec" data-testid="portable-w154">
      <div class="receipt-head">W154 便携格式判定 · 读盘</div>
      <dl class="dl">
        <dt>verdict</dt><dd class="mono" data-testid="portable-w154-verdict">${esc(w154.present === false ? "缺失" : (w154.verdict || "—"))}</dd>
        <dt>second_machine</dt><dd class="mono" data-testid="portable-w154-second-machine">${esc(w154.second_machine_status || "—")}</dd>
        <dt>format_version</dt><dd class="mono">${esc(w154.format_version || "—")}</dd>
        <dt>matrix</dt><dd class="mono">${matrix.passed === undefined ? "—" : (matrix.passed + " / " + matrix.cases + " 例")}</dd>
      </dl>
    </div>`;
    const currentBlock = detail ? `<div class="pt-sec" data-testid="portable-current">
      <div class="receipt-head">当前便携包 · open_package 读数<span class="spacer"></span>${badge(tone, status || "—")}</div>
      <div class="pt-status-line"><span class="micro">状态</span><b class="mono" data-testid="portable-status">${esc(status || "")}</b>
        <span class="micro">ok</span><span class="mono" data-testid="portable-ok">${String(!!detail.ok)}</span>
        <span class="micro">clean</span><span class="mono">${String(detail.clean)}</span></div>
      <dl class="dl">
        <dt>package_id</dt><dd class="mono ellipsis" data-testid="portable-package-id" title="${esc((detail.manifest || {}).package_id || "")}">${esc((detail.manifest || {}).package_id || "—")}</dd>
        <dt>包路径</dt><dd class="mono ellipsis" data-testid="portable-package-path" title="${esc(detail.package || "")}">${esc(detail.package || "")}</dd>
        <dt>project</dt><dd class="mono ellipsis" data-testid="portable-project-id">${esc(project.project_id || "—")} · v${project.project_version === undefined ? "—" : project.project_version}</dd>
        <dt>content_hash</dt><dd class="sha-row" data-testid="portable-project-hash">${hashline(project.content_hash, "portable-hash")}</dd>
        <dt>几何与清单一致</dt><dd class="mono" data-testid="portable-geometry-match">${String(!!project.geometry_matches_manifest)}</dd>
        <dt>证据 binding</dt><dd class="mono" data-testid="portable-evidence-binding">${esc(evidence.binding || "—")}</dd>
        <dt>证据当前</dt><dd class="mono" data-testid="portable-evidence-current">${String(!!evidence.current)}</dd>
        <dt>资产</dt><dd class="mono" data-testid="portable-assets"${Object.keys(byStatus).map(function (k) {
          return ' data-' + k.toLowerCase() + '="' + esc(byStatus[k]) + '"';
        }).join("")}>${["AVAILABLE", "RELOCATED", "DRIFT_ACCEPTED", "MISSING", "REPLACED"].map(function (k) {
          return k + "=" + (byStatus[k] === undefined ? 0 : byStatus[k]);
        }).join(" ")}</dd>
        <dt>可执行动作</dt><dd class="mono" data-testid="portable-options">${esc(((detail.state_machine || {}).options || []).join(", ") || "—")}</dd>
      </dl>
      ${items.length ? `<div class="pt-assets" data-testid="portable-asset-list">${items.map(function (a, i) {
        const at = portableTone(a.status);
        const need = a.status === "MISSING" || a.status === "REPLACED";
        return `<div class="pt-asset" data-testid="portable-asset-${i}" data-asset="${esc(a.asset_id)}" data-status="${esc(a.status)}">`
          + `<i class="dot" style="background:${TONE_VAR[at]}"></i>`
          + `<span class="mono ellipsis" title="${esc(a.path || a.asset_id)}">${esc(a.asset_id)}</span>`
          + `<span class="badge ${at}">${esc(a.status)}</span>`
          + `<span class="micro">${esc(a.hash_verification || "")}</span>`
          + (need ? `<button class="btn sm" data-action="portable-relocate-fill" data-asset="${esc(a.asset_id)}" data-testid="portable-relocate-fill-${i}"${view.busy["portable-relocate"] ? " disabled" : ""}>填入重定位</button>` : "")
          + `</div>`;
      }).join("")}</div>` : ""}
      ${warnings.length ? `<div class="callout warn" data-testid="portable-warnings" data-count="${warnings.length}"><div class="c-title">${icon("alert")}open_package 警告 · ${warnings.length}</div><div class="c-body">${warnings.map(function (w) { return `<div class="pt-warn">${esc(w)}</div>`; }).join("")}</div>` : `<div class="pt-warn-none" data-testid="portable-warnings" data-count="0">open_package 无警告。</div>`}
    </div>` : `<div class="pt-sec" data-testid="portable-empty"><div class="receipt-head">当前便携包</div><div class="ins-empty">尚未打开便携包。<br>从下方清单打开一个 .ffportable，或先导出当前项目。</div></div>`;
    const relocateBlock = `<div class="pt-sec" data-testid="portable-relocate">
      <div class="receipt-head">重定位（只接受全文件哈希相同的文件）</div>
      <div class="field"><label class="micro" for="portable-asset-id">资产 ID</label>
        <input id="portable-asset-id" type="text" value="${esc(missing ? missing.asset_id : "")}" data-testid="portable-asset-id" autocomplete="off" spellcheck="false"></div>
      <div class="field"><label class="micro" for="portable-new-path">新文件路径（绝对路径或相对仓库根）</label>
        <input id="portable-new-path" type="text" placeholder="C:\\...\\points.ply" data-testid="portable-new-path" autocomplete="off" spellcheck="false"></div>
      <div class="btn-row"><button class="btn sm" data-action="portable-relocate" data-testid="portable-relocate-btn" ${detail && !view.busy["portable-relocate"] ? "" : "disabled"}${view.busy["portable-relocate"] ? ' title="全文件哈希校验约需十几秒，请勿重复点击"' : ""}>${view.busy["portable-relocate"] ? "重定位中…" : "重定位并重开"}</button></div>
    </div>`;
    const receiptBlock = receipt ? `<div class="pt-sec" data-testid="portable-export-receipt">
      <div class="receipt-head">最近一次便携导出 · 服务端回执</div>
      <dl class="dl">
        <dt>exported</dt><dd class="mono" data-testid="portable-export-exported">${String(!!receipt.exported)}</dd>
        <dt>package_id</dt><dd class="mono ellipsis" data-testid="portable-export-package-id">${esc(receipt.package_id || "—")}</dd>
        <dt>模式</dt><dd class="mono" data-testid="portable-export-mode">${esc(receipt.distribution_mode || "—")} · 复制资产字节=${String(!!receipt.copied_asset_bytes)}</dd>
        <dt>自校验</dt><dd class="mono" data-testid="portable-export-verify">${esc(((receipt.verification || {}).status) || "—")} · 文件 ${esc(((receipt.verification || {}).checked_files) || 0)}</dd>
        <dt>资产哈希</dt><dd class="mono" data-testid="portable-export-hash">核过 ${esc(((receipt.assets || {}).hash_verified_count) || 0)} / 共 ${esc(((receipt.assets || {}).asset_count) || 0)}</dd>
        <dt>证据</dt><dd class="mono" data-testid="portable-export-evidence">${esc(((receipt.evidence || {}).binding) || "—")}</dd>
        <dt>重开状态</dt><dd class="mono" data-testid="portable-export-open-status">${esc(receipt.open_status || "—")}</dd>
        <dt>警告</dt><dd class="mono" data-testid="portable-export-warnings">${(receipt.warnings || []).length}</dd>
      </dl>
    </div>` : "";
    const candidates = st.candidates || [];
    return `
      <div class="pt-panel" data-testid="portable-panel">
        ${w154Block}
        ${currentBlock}
        ${receiptBlock}
        ${relocateBlock}
        <div class="pt-sec" data-testid="portable-export">
          <div class="receipt-head">便携导出</div>
          <div class="hint">只从已保存的 bundle 构建；REFERENCE_ONLY 不复制真实测量字节，导出后立即独立复核并打开。</div>
          <div class="btn-row"><button class="btn sm primary" data-action="portable-export" data-testid="portable-export-btn" ${S.project_open && !view.busy["portable-export"] ? "" : "disabled"}>${view.busy["portable-export"] ? "导出中…" : "导出当前项目为 .ffportable"}</button></div>
          ${S.project_open && S.dirty ? `<div class="hint">当前有未保存修改：先保存，否则服务端会拒绝导出（PORTABLE_DIRTY）。</div>` : ""}
        </div>
        <div class="pt-sec" data-testid="portable-candidates">
          <div class="receipt-head">可打开的便携包 · ${candidates.length}</div>
          <div class="pt-cands">${candidates.length ? candidates.map(function (c, i) {
            const opening = view.busy["portable-open"] === c.path;
            return `<button class="pt-cand" data-action="portable-open" data-package="${esc(c.path)}" data-testid="portable-candidate-${i}"${view.busy["portable-open"] ? " disabled" : ""}>`
              + `<span class="mono ellipsis" title="${esc(c.path)}">${opening ? "打开中…" : esc(c.name)}</span>`
              + `<span class="badge ${c.readable ? "ok" : "err"}">${c.readable ? esc(c.distribution_mode || "?") : "不可读"}</span>`
              + `</button>`;
          }).join("") : `<div class="ins-empty">允许根内没有 .ffportable。<br>${esc((st.roots || []).join(" · "))}</div>`}</div>
        </div>
        <div class="modal-foot pt-foot">
          <button class="btn sm ghost" data-action="portable-refresh" data-testid="portable-refresh-btn">刷新</button>
          <button class="btn sm ghost" data-action="portable-detach" data-testid="portable-detach-btn" ${detail ? "" : "disabled"}>关闭包视图</button>
          <span class="spacer"></span>
          <button class="btn sm primary" data-action="modal-close" data-testid="portable-done-btn">完成</button>
        </div>
      </div>`;
  }

  /* ---------------- 模态 ---------------- */
  function openModal(kind) {
    view.modal = kind;
    if (kind === "portable") {
      $("#modal-root").innerHTML = `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal modal-wide" role="dialog" aria-modal="true" aria-label="W154 便携包">
            <div class="modal-head"><span class="modal-title">W154 便携项目 · 导出 / 打开 / 重定位</span><span class="spacer"></span>
              <button class="icon-btn" data-action="modal-close" aria-label="关闭">${icon("close")}</button></div>
            <div class="modal-body" id="portable-body">${portablePanelHTML()}</div>
          </div>
        </div>`;
      return;
    }
    const r = S.exportReceipt;
    if (kind === "export" && r) {
      $("#modal-root").innerHTML = `
        <div class="modal-backdrop" data-modal-backdrop>
          <div class="modal" role="dialog" aria-modal="true" aria-label="导出回执">
            <div class="modal-head"><span class="modal-title">导出回执 · 读回核对</span><span class="spacer"></span>
              <button class="icon-btn" data-action="modal-close" aria-label="关闭">${icon("close")}</button></div>
            <div class="modal-body">
              <div class="meta-row">
                <span class="meta"><span class="micro">export_id</span>${hashline(r.export_id, "modal-export-id")}</span>
                <span class="meta"><span class="micro">版本</span><span class="v">v${r.project_version}</span></span>
                <span class="meta"><span class="micro">枚举</span><span class="v">${r.enumeration_count} 项</span></span>
                <span class="meta"><span class="micro">时刻</span><span class="v">${esc(fmtTime(r.at))}</span></span>
              </div>
              <div class="export-dir-line" data-testid="modal-export-dir"><span class="micro">导出目录（在盘路径）</span>${r.export_dir ? hashline(r.export_dir, "modal-export-dir-path") : '<span class="mono">—</span>'}</div>
              <div class="checks">${r.checks.map((c) => `<span class="check ${c.pass ? "" : "bad"}">${icon(c.pass ? "check" : "close")}<span title="${esc(c.name)}">${esc(c.name)}</span></span>`).join("")}</div>
            </div>
            <div class="modal-foot">
              <button class="btn sm" data-action="download-project" data-testid="modal-download-project">${icon("download")}project.json</button>
              <button class="btn sm" data-action="download-manifest" data-testid="modal-download-manifest">${icon("download")}manifest.json</button>
              <button class="btn sm" data-action="download-readback" data-testid="modal-download-readback">${icon("download")}readback.json</button>
              <button class="btn sm primary" data-action="modal-close" data-testid="modal-done">完成</button>
            </div>
          </div>
        </div>`;
    }
  }
  function closeModal() {
    view.modal = null;
    $("#modal-root").innerHTML = "";
  }

  /* ---------------- 渲染调度 ---------------- */
  function sig() {
    const job = S.jobs[S.jobs.length - 1];
    /* W169 D09：登记资料/便携入口/就绪态必须进 sig。否则 boot 后异步 catalog 到达不触发 full 重渲染，
       数据集按钮在 /api/state 慢（负载）时长时间缺失（受控延迟已确定性复现）。
       catalog 签名含条目 id 与 build_present；view.saving/reopening 进 sig 使长操作反馈即时渲染。 */
    const catSig = ((S.catalog && S.catalog.entries) || []).map(function (e) { return e.id + ":" + (e.build_present === false ? 0 : 1); }).join(",");
    const w154 = (S.portableStatus && S.portableStatus.w154) || null;
    return JSON.stringify([
      S.project_open, S.source, S.project_version, S.dirty, S.selectedId,
      S.preview && S.preview.token, S.saveError && S.saveError.at,
      S.saveReceipt && S.saveReceipt.at, S.exportReceipt && S.exportReceipt.export_id,
      S.library.length, Object.keys(S.review).length, S.log.length,
      S.activeJobId, job && job.status,
      view.filter, view.query, view.logOpen, view.inspectorOpen,
      catSig, S.ready, S.state_loaded, w154 && w154.present, w154 && w154.verdict, view.saving, view.reopening,
      view.undoRedoBusy, JSON.stringify(view.busy), S.can_undo, S.can_redo,
    ]);
  }
  function captureFocus() {
    const el = document.activeElement;
    if (!el || el === document.body) return null;
    const key = el.dataset && el.dataset.edit ? "edit:" + el.dataset.edit : el.id ? "id:" + el.id : null;
    if (!key) return null;
    let sel = null;
    try { if (typeof el.selectionStart === "number") sel = [el.selectionStart, el.selectionEnd]; } catch { /* select 元素 */ }
    return { key, sel };
  }
  function restoreFocus(f) {
    if (!f) return;
    const el = f.key.startsWith("edit:") ? document.querySelector(`[data-edit="${f.key.slice(5)}"]`) : document.getElementById(f.key.slice(3));
    if (!el) return;
    el.focus();
    if (f.sel && typeof el.setSelectionRange === "function") { try { el.setSelectionRange(f.sel[0], f.sel[1]); } catch { /* ignore */ } }
  }
  function syncScene() {
    if (!scene) return;
    if (S.project !== lastProjectRef) {
      const st = scene._state;
      const cam = { yaw: st.yaw, pitch: st.pitch, dist: st.dist, pan: st.pan.slice() };
      const had = !!lastProjectRef;
      const prevRadius = st.radius;
      scene.setProject(S.project);
      if (had && S.project) {
        /* 同项目换版本保持机位；换项目且尺度显著变化时重新适配，否则图版被裁 */
        const ratio = st.radius / (prevRadius || 1);
        if (ratio > 0.8 && ratio < 1.25) { st.yaw = cam.yaw; st.pitch = cam.pitch; st.dist = cam.dist; st.pan = cam.pan; }
        else scene.fit();
      }
      lastProjectRef = S.project;
    }
    scene.setSelected(S.selectedId);
    scene.setReview(S.review);
    const ids = S.preview && CORE.previewIsLive()
      ? [...(S.preview.diff.changed || []), ...(S.preview.diff.added || [])].map((d) => (typeof d === "string" ? d : d.id))
      : [];
    scene.setPreviewIds(ids);
    scene.requestRender();
  }
  function refresh(force) {
    const s = sig();
    const full = force || s !== lastSig;
    lastSig = s;
    const focus = full ? captureFocus() : null;
    const ledgerScroll = $("#ledger-list") ? $("#ledger-list").scrollTop : 0;
    const insScroll = $("#ins-scroll").scrollTop;
    if (full) {
      $("#topbar").innerHTML = topbarHTML();
      $("#ledger").innerHTML = ledgerHTML();
      Array.prototype.forEach.call(document.querySelectorAll("#ledger-list .row"), function (row) {
        if (row.dataset.id) row.dataset.testid = "object-row-" + row.dataset.id;
      });
      $("#ins-head").innerHTML = insHeadHTML();
      $("#ins-scroll").innerHTML = inspectorHTML();
      const receiptBlock = receiptsHTML();
      if (receiptBlock) $("#ins-scroll").insertAdjacentHTML("beforeend", receiptBlock);
      const dockHTML = insDockHTML();
      $("#ins-dock").innerHTML = dockHTML;
      $("#ins-dock").hidden = !dockHTML;
      syncInsFade();
      $("#vp-caption").innerHTML = captionHTML();
      $("#vp-tools").innerHTML = toolsHTML();
      $("#vp-legend").innerHTML = legendHTML();
      const empty = $("#vp-empty");
      empty.hidden = S.project_open;
      if (!S.project_open) empty.innerHTML = emptyHTML();
      if (!S.project_open) {
        const registryHost = empty.querySelector(".empty-actions");
        if (registryHost) registryHost.insertAdjacentHTML("afterend", catalogButtonsHTML());
      }
      /* 空态是一张扉页，不是图纸：撤掉工具、图例与比例尺，纸面才完整 */
      $("#vp-tools").style.display = S.project_open ? "" : "none";
      $("#vp-foot").hidden = !S.project_open;
      $("#logdock").innerHTML = logHTML();
      $("#ledger-list").scrollTop = ledgerScroll;
      $("#ins-scroll").scrollTop = insScroll;
      const rid = (S.saveReceipt ? S.saveReceipt.at : "") + "|" + (S.exportReceipt ? S.exportReceipt.export_id : "");
      if (rid !== lastReceiptSig && (S.saveReceipt || S.exportReceipt)) $("#ins-scroll").scrollTop = 0;
      lastReceiptSig = rid;
      restoreFocus(focus);
    }
    $("#statusrow").innerHTML = statusrowHTML();
      const statusLine = $("#statusrow .sl");
      if (statusLine) {
        const statusText = statusLine.querySelector(".sl-text");
        if (statusText) statusText.dataset.testid = "status-text";
        statusLine.insertAdjacentHTML("beforeend", statusMetaHTML());
      }
    /* W169：只读引导条每次刷新都同步（迁移后立即消失），不依赖 full 重绘 */
    const banner = $("#readonly-banner");
    if (banner) {
      const bannerHTML = readonlyBannerHTML();
      banner.hidden = !bannerHTML;
      banner.innerHTML = bannerHTML;
    }
    $("#cmdbar").innerHTML = cmdbarHTML();
    $("#logdock").dataset.open = String(view.logOpen);
    if (view.logOpen) { const b = $("#log-body"); if (b) b.scrollTop = b.scrollHeight; }
    document.body.classList.toggle("inspector-open", view.inspectorOpen);
    document.body.classList.toggle("ledger-open", view.ledgerOpen);
    syncScene();
    if (view.menu) {
      const anchor = document.querySelector(`[data-action="${view.menu === "project" ? "project-menu" : view.menu === "solve" ? "solve-menu" : "layers-btn"}"]`);
      if (anchor) openMenu(view.menu, anchor); else closeMenu();
    }
  }
  /* 证据栏滚动截断的渐隐开关：内容装得下、或已滚到底，就不虚化 */
  function syncInsFade() {
    const sc = $("#ins-scroll"), dk = $("#ins-dock");
    if (!sc || !dk) return;
    const cut = sc.scrollHeight - sc.clientHeight > 1;
    const atEnd = sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 2;
    dk.classList.toggle("is-clipped", cut && !atEnd);
  }
  function scheduleRefresh() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; refresh(false); });
  }

  /* ---------------- 动作 ---------------- */
  const ctx = {
    refresh: () => refresh(true),
    focusLog: () => { view.logOpen = true; refresh(true); },
    getPreviewSpec: () => ({
      action: view.edit.action,
      targets: (view.edit.targets || S.selectedId || "").split(",").map((t) => t.trim()).filter(Boolean),
      payload: payloadObj() || {},
    }),
    selectObject: (id) => CORE.selectObject(id),
    solveScope: () => "local",
    scene: null,
  };
const base = createApiActions(ctx);
  /* 窄屏：用户主动选中（清单或图纸拾取）即滑出证据栏；状态注入不触发 */
  async function userSelect(id) {
    if (id && window.innerWidth <= 1180) view.inspectorOpen = true;
    if (id && window.innerWidth <= 720) view.ledgerOpen = false;
    refresh(true);
    await CORE.selectObject(id);
    /* W169 F1：骨架类动作换对象后按新对象的支持索引重算；用户手改过的载荷一律不动。
       W171 W170C-02：重生成时清掉当前动作的旧草稿（已按新对象重建）；dirty 时仍不覆盖用户输入。 */
    if (!view.edit.payloadDirty
      && (view.edit.action === "SELECT_SUPPORT" || view.edit.action === "SPLIT_SURFACE")) {
      view.edit.payloadText = JSON.stringify(payloadTemplate(view.edit.action, CORE.getObject(id)), null, 2);
      delete view.edit.drafts[view.edit.action];
    }
    refresh(true);
  }
  const handlers = Object.assign({}, base, {
    "select-object": (t) => userSelect(t.dataset.id),
    "filter": (t) => { view.filter = t.dataset.filter; refresh(true); },
    "project-menu": (t) => { view.menu === "project" ? closeMenu() : openMenu("project", t); },
    "solve-menu": (t) => { view.menu === "solve" ? closeMenu() : openMenu("solve", t); },
    "layers-btn": (t) => { view.menu === "layers" ? closeMenu() : openMenu("layers", t); },
    /* W172 solve-full 覆盖修复：原 fire-and-forget 遮蔽 base 的成功/失败 toast。
       保留 iterations/timeout 参数——这是覆盖 handler 的既有语义，不是重复代码。 */
    "solve-full": async () => {
      const result = await CORE.startSolve({ scope: "full", iterations: 2, timeout: 30 });
      toastResult(result, "全域求解已提交", "求解未启动");
      refresh(true);
    },
    /* W169 D03：删除原 "reopen"/"reopen-project" 覆盖 handler。
       旧覆盖未 await 异步结果（r 为 Promise，r.ok 恒 undefined → toast 恒「没有可重开的保存版本」），
       且用 S.bundle_id（关闭后为 null）而非 lastBundleId/按需恢复。现回落到 base 的稳健 async handler。 */
    "log-toggle": () => { view.logOpen = !view.logOpen; refresh(true); },
    "inspector-toggle": () => { view.inspectorOpen = !view.inspectorOpen; if (view.inspectorOpen) view.ledgerOpen = false; refresh(true); },
    "ledger-toggle": () => { view.ledgerOpen = !view.ledgerOpen; if (view.ledgerOpen) view.inspectorOpen = false; refresh(true); },
    "modal-close": () => closeModal(),
    "export-receipt-open": () => openModal("export"),
    "preview": () => {
      if (!NO_PAYLOAD.has(view.edit.action) && payloadObj() === null) { UI.toast("载荷 JSON 解析失败 · 预览未发起", "warn"); return; }
      base["preview"]();
    },
    /* W171 W170C-02：显式重填骨架——仅对 SELECT_SUPPORT/SPLIT_SURFACE 生效，按当前选中对象重建。 */
    "refill-skeleton": () => {
      const a = view.edit.action;
      if (a !== "SELECT_SUPPORT" && a !== "SPLIT_SURFACE") return;
      const o = CORE.getObject(S.selectedId);
      if (!o) { UI.toast("未选中对象：请先在左栏选一个片", "warn"); return; }
      view.edit.payloadText = JSON.stringify(payloadTemplate(a, o), null, 2);
      view.edit.payloadDirty = false;
      delete view.edit.drafts[a];
      refresh(true);
      UI.toast("已按当前选中对象重填骨架", "ok");
    },
  });
  const wrapped = {};
  for (const key of Object.keys(handlers)) {
    wrapped[key] = (t, e) => {
      handlers[key](t, e);
      if (view.menu && !["project-menu", "solve-menu", "layers-btn"].includes(key)) closeMenu();
    };
  }
  UI.bindActions(document.body, wrapped);
  $("#ins-scroll").addEventListener("scroll", syncInsFade, { passive: true });

  document.addEventListener("click", (e) => {
    if (view.modal && e.target.hasAttribute && e.target.hasAttribute("data-modal-backdrop")) { closeModal(); return; }
    if (view.menu && !e.target.closest(".pop") && !e.target.closest('[data-action="project-menu"],[data-action="solve-menu"],[data-action="layers-btn"]')) closeMenu();
    /* 清单抽屉：点纸面或任何栏外区域就收起，不必非要找那个叉 */
    if (view.ledgerOpen && !e.target.closest(".ledger") && !e.target.closest('[data-action="ledger-toggle"]')) {
      view.ledgerOpen = false; refresh(true);
    }
  });
  document.addEventListener("change", (e) => {
    const layer = e.target.dataset ? e.target.dataset.layer : null;
    if (layer && scene) { scene.setLayers({ [layer]: e.target.checked }); return; }
    const ed = e.target.dataset ? e.target.dataset.edit : null;
    if (!ed) return;
    const o = CORE.getObject(S.selectedId);
    if (ed === "action") {
      const next = e.target.value;
      /* W171 W170C-02：切换动作＝换载荷模板。编辑过的载荷按动作存草稿，切回恢复；
         无草稿一律按当前对象重生成（不再保留旧动作的编辑文本——它已存进旧动作草稿）。
         结构化动作（解释/关系/unknown）无 textarea，payloadDirty 恒 false，草稿自然为空，行为不变。 */
      const prev = view.edit.action;
      if (view.edit.payloadDirty) view.edit.drafts[prev] = view.edit.payloadText;
      else delete view.edit.drafts[prev];
      view.edit.action = next;
      if (Object.prototype.hasOwnProperty.call(view.edit.drafts, next)) {
        view.edit.payloadText = view.edit.drafts[next];
        view.edit.payloadDirty = true;
      } else {
        view.edit.payloadText = JSON.stringify(payloadTemplate(next, o), null, 2);
        view.edit.payloadDirty = false;
      }
      refresh(true);
    } else if (ed === "interpretation") {
      view.edit.payloadText = JSON.stringify({ interpretation_status: e.target.value });
    } else if (ed === "relstate") {
      const po = payloadObj() || {};
      po.state = e.target.value;
      po.unknown_reason = e.target.value === "unknown" ? (po.unknown_reason || "OCCLUDED") : null;
      view.edit.payloadText = JSON.stringify(po);
      refresh(true);
    } else if (ed === "relreason") {
      const po = payloadObj() || {};
      po.unknown_reason = e.target.value;
      view.edit.payloadText = JSON.stringify(po);
    }
  });
  document.addEventListener("input", (e) => {
    const ed = e.target.dataset ? e.target.dataset.edit : null;
    if (e.target.id === "ledger-search") { view.query = e.target.value; refresh(true); return; }
    if (ed === "targets") view.edit.targets = e.target.value;
    else if (ed === "payload") { view.edit.payloadText = e.target.value; view.edit.payloadDirty = true; }
  });
  document.addEventListener("keydown", (e) => {
    const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Escape") {
      if (view.modal) closeModal();
      else if (view.menu) closeMenu();
      else if (S.preview && CORE.previewIsLive()) { CORE.cancelPreview(); refresh(true); }
      else if (view.logOpen) { view.logOpen = false; refresh(true); }
      else if (view.ledgerOpen) { view.ledgerOpen = false; refresh(true); }
      else if (view.inspectorOpen) { view.inspectorOpen = false; refresh(true); }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      if (S.project_open) wrapped["save"]();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); if (S.project_open) wrapped["preview"](); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !inField) { e.preventDefault(); wrapped[e.shiftKey ? "redo" : "undo"](); return; }
    if (e.key === "/" && !inField) { e.preventDefault(); const el = $("#ledger-search"); if (el) el.focus(); }
  });


  /* ---------------- 启动 ---------------- */
  scene = window.PlateC.create($("#plate"), {
    onPick: (id) => { if (id) userSelect(id); },
    onFurniture: (si) => paintScale(si),
  });
  ctx.scene = scene;
  view.edit.payloadText = JSON.stringify(payloadTemplate(view.edit.action, null), null, 2);
  CORE.subscribe(scheduleRefresh);
  refresh(true);
  CORE.init().then(function () {
    if (!S.selectedId && S.project && S.project.objects.length) {
      CORE.selectObject(S.project.objects[0].id).then(function () { refresh(true); });
    }
    refresh(true);
  });
  window.__W162_UI__ = { refresh: function () { refresh(true); }, view: view };

})();