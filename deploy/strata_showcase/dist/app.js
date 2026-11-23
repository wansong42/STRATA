/* =========================================================================
   STRATA · 岩体非连续裂隙网络贯通可辨识性三维展示台 (极简画廊 + AI 自助助手版)
   -------------------------------------------------------------------------
   1. 严格相对路径 (./data/...)，100% 兼容根路径与 Caddy /strata/ 子路径托管。
   2. 零 Emoji、零外部 CDN、零写盘接口；100% 忠于 M1/M2/M3 真包与三项 null 边界。
   3. 极简两栏画廊结构（宽阔 3D 舞台 + 右侧三页签极简面板）+ 悬浮 AI 智能自助小助手
      （离线零幻觉真包驱动，可直接用大白话讲解并自动帮你切换案例、分屏、聚焦断口与坍缩）。
   ========================================================================= */
"use strict";

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (m) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[m]));
  }

  /* ---------------- 定制岩石力学与精密测绘 SVG 图标系统 (零 Emoji) ---------------- */
  const ICONS = {
    /* STRATA 专属徽记：三维岩体块段 + 两条非正交交叉裂隙面 + 端点 A→B 贯通路径 */
    strataBrand: `<svg viewBox="0 0 32 32" width="26" height="26" fill="none" aria-hidden="true">
      <polygon points="16,2 28.5,9.2 28.5,22.8 16,30 3.5,22.8 3.5,9.2" fill="currentColor" fill-opacity="0.05" stroke="currentColor" stroke-width="1.35" stroke-linejoin="round"/>
      <path d="M16 2v9.2M3.5 9.2l12.5 7.2 12.5-7.2M16 16.4V30" stroke="currentColor" stroke-opacity="0.18" stroke-width="0.9"/>
      <polygon points="7.8,9.8 20.6,12.8 18.2,24.4 6.2,20.2" fill="#0d9488" fill-opacity="0.22" stroke="#0f766e" stroke-width="1.1" stroke-linejoin="round"/>
      <polygon points="14.2,7 25.4,14.6 23.4,23.8 12,16.8" fill="#d97706" fill-opacity="0.20" stroke="#b45309" stroke-width="1.1" stroke-linejoin="round"/>
      <line x1="12.6" y1="16.9" x2="19.8" y2="12.9" stroke="#059669" stroke-width="1.2" stroke-dasharray="1.8 1.2"/>
      <path d="M9 12.4 L16 16.4 L22.8 20.6" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="9" cy="12.4" r="2.2" fill="#0f766e" stroke="#ffffff" stroke-width="0.9"/>
      <circle cx="22.8" cy="20.6" r="2.2" fill="#1d4ed8" stroke="#ffffff" stroke-width="0.9"/>
    </svg>`,
    copilot: `<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M10 2.6 11.7 7.6l5 1.7-5 1.7L10 16l-1.7-5-5-1.7 5-1.7L10 2.6Z" fill="currentColor" fill-opacity="0.14"/>
      <circle cx="15.5" cy="4.5" r="1.2" fill="currentColor" stroke="none"/>
    </svg>`,
    send: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M15.5 2.5 8.5 9.5"/>
      <polygon points="15.5,2.5 11,15.5 8.5,9.5 2.5,7" fill="currentColor" fill-opacity="0.14"/>
    </svg>`,
    tour: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polygon points="6,4 14,9 6,14" fill="currentColor" fill-opacity="0.16"/>
    </svg>`,
    shieldOk: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M9 2 14.8 4.2v4.3c0 3.8-2.4 6.6-5.8 7.6-3.4-1-5.8-3.8-5.8-7.6V4.2L9 2Z" fill="currentColor" fill-opacity="0.10"/>
      <path d="m6.5 9.1 1.8 1.9 3.4-3.6"/>
    </svg>`,
    split: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="2" y="3" width="14" height="12" rx="2"/>
      <line x1="9" y1="3" x2="9" y2="15"/>
    </svg>`,
    focusGap: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="9" cy="9" r="5.2"/>
      <line x1="9" y1="1.8" x2="9" y2="4.2"/>
      <line x1="9" y1="13.8" x2="9" y2="16.2"/>
      <line x1="1.8" y1="9" x2="4.2" y2="9"/>
      <line x1="13.8" y1="9" x2="16.2" y2="9"/>
    </svg>`,
    fit: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M2.8 6.2V2.8h3.4M11.8 2.8h3.4v3.4M15.2 11.8v3.4h-3.4M6.2 15.2H2.8v-3.4"/>
      <polygon points="9,5.4 12.6,7.4 12.6,11.2 9,13.2 5.4,11.2 5.4,7.4" fill="currentColor" fill-opacity="0.08"/>
    </svg>`,
    topView: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polygon points="9,2.8 15.2,6.3 9,9.8 2.8,6.3" fill="currentColor" fill-opacity="0.18"/>
      <path d="m2.8 10.2 6.2 3.5 6.2-3.5" stroke-opacity="0.45"/>
    </svg>`,
    frontView: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="4" width="12" height="10" rx="1.5" fill="currentColor" fill-opacity="0.10"/>
      <line x1="3" y1="9" x2="15" y2="9" stroke-dasharray="2 1.8"/>
    </svg>`,
    rotate: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M14.6 7.4A5.8 5.8 0 1 0 13.5 12.8"/>
      <polyline points="15,4.2 14.8,7.6 11.4,7.4"/>
    </svg>`,
    camera: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M2.5 6h2.2l1.2-1.8h6.2L13.3 6h2.2a1.3 1.3 0 0 1 1.3 1.3v6.6a1.3 1.3 0 0 1-1.3 1.3h-13a1.3 1.3 0 0 1-1.3-1.3V7.3A1.3 1.3 0 0 1 2.5 6Z"/>
      <circle cx="9" cy="10.4" r="2.4"/>
    </svg>`,
    hole: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
      <circle cx="8" cy="8" r="5.8"/>
      <circle cx="8" cy="8" r="2.3" stroke-dasharray="1.8 1.2"/>
    </svg>`,
    search: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
      <circle cx="7.8" cy="7.8" r="4.8"/>
      <line x1="11.4" y1="11.4" x2="15.2" y2="15.2"/>
    </svg>`,
    reset: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M3.8 6.8A5.8 5.8 0 1 1 3.4 10.8"/>
      <polyline points="3.1,3.5 3.5,7 7,6.6"/>
    </svg>`,
    copy: `<svg viewBox="0 0 18 18" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="6.5" y="6.5" width="8.5" height="8.5" rx="1.5"/>
      <path d="M11.5 6.5V4.5A1.5 1.5 0 0 0 10 3H4.5A1.5 1.5 0 0 0 3 4.5V10a1.5 1.5 0 0 0 1.5 1.5h2"/>
    </svg>`,
    close: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
      <line x1="4.5" y1="4.5" x2="13.5" y2="13.5"/>
      <line x1="13.5" y1="4.5" x2="4.5" y2="13.5"/>
    </svg>`,
    sun: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true">
      <circle cx="9" cy="9" r="3.2"/>
      <path d="M9 2v1.8M9 14.2V16M2 9h1.8M14.2 9H16M4 4l1.3 1.3M12.7 12.7 14 14M14 4l-1.3 1.3M5.3 12.7 4 14"/>
    </svg>`,
    moon: `<svg viewBox="0 0 18 18" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M14.2 10.6A5.8 5.8 0 0 1 7.4 3.8a5.8 5.8 0 1 0 6.8 6.8Z"/>
    </svg>`,
    chevronDown: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polyline points="4,6 8,10 12,6"/>
    </svg>`,
    arrowRight: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <line x1="3" y1="8" x2="13" y2="8"/>
      <polyline points="9,4 13,8 9,12"/>
    </svg>`,
    list: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true">
      <line x1="3" y1="4.5" x2="15" y2="4.5"/>
      <line x1="3" y1="9" x2="15" y2="9"/>
      <line x1="3" y1="13.5" x2="11" y2="13.5"/>
    </svg>`,
    sliders: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true">
      <line x1="3" y1="5" x2="15" y2="5"/>
      <circle cx="7" cy="5" r="1.6" fill="var(--bg-surface)"/>
      <line x1="3" y1="13" x2="15" y2="13"/>
      <circle cx="11.5" cy="13" r="1.6" fill="var(--bg-surface)"/>
    </svg>`,
    layers: `<svg viewBox="0 0 18 18" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polygon points="9,2.8 15.2,6.2 9,9.6 2.8,6.2"/>
      <polyline points="2.8,9.8 9,13.2 15.2,9.8"/>
      <polyline points="2.8,13 9,16.2 15.2,13"/>
    </svg>`,
    chevronUp: `<svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polyline points="4,10 8,6 12,10"/>
    </svg>`,
  };

  /* ---------------- 语义中文映射 ---------------- */
  function statusMeta(st) {
    if (st === "CERTIFIED_CONNECTED") return { cls: "connected", zh: "确定贯通", short: "确定贯通", code: "CERTIFIED_CONNECTED" };
    if (st === "CERTIFIED_DISCONNECTED") return { cls: "disconnected", zh: "确定断开", short: "确定断开", code: "CERTIFIED_DISCONNECTED" };
    if (st === "AMBIGUOUS_WITH_VERIFIED_WITNESSES") return { cls: "ambiguous", zh: "存在双见证歧义", short: "双见证歧义", code: "AMBIGUOUS_WITH_VERIFIED_WITNESSES" };
    if (st === "BUDGET_EXCEEDED" || st === "REFUSED" || st === "INFEASIBLE") return { cls: "refused", zh: "超预算诚实拒答", short: "超预算拒答", code: st || "BUDGET_EXCEEDED" };
    return { cls: "pending", zh: "加载核验中", short: "待核", code: st || "IDLE" };
  }

  function friendlyRole(role) {
    return {
      ANCHOR_A: "A 端锚定面",
      ANCHOR_B: "B 端锚定面",
      INTERIOR_F1: "核心裂隙 F1",
      INTERIOR_F2: "核心裂隙 F2",
      INTERIOR: "内部传导面",
    }[role] || role || "内部裂隙面";
  }

  function friendlyStratum(stratum) {
    if (stratum === "MUTUAL_EXCL_ATTRIB_MULTISEG") return "互斥归属 · 多段残差域";
    if (stratum === "STANDARD_EXT_TILT") return "标准延展 · 倾角微扰域";
    return stratum || "条件几何域";
  }

  function friendlySite(siteId) {
    if (siteId === "jardim_garcia_uh") return "Jardim Garcia 采石场";
    if (siteId === "perozzo_finale_ligure") return "Perozzo 意大利 Finale Ligure";
    return siteId || "真实露头场地";
  }

  function friendlyActionType(act) {
    if (act === "RESOLVE_MODE_FEATURE") return "复核裂隙产状 / 边界归属";
    if (act === "REFINE_PARAMETER_BAND") return "精测法向位移临界区间 t*";
    return "复核关键裂隙特征";
  }

  function friendlyBranchLabel(rawLabel, _rawCond, branchId) {
    const s = String(rawLabel || branchId || "");
    if (s.includes("TILT_0DEG")) return "确认该裂隙倾角为 0°（基准产状）";
    if (s.includes("TILT_NEG_3.95DEG")) return "确认该裂隙倾角负向偏转 -3.95°";
    if (s.includes("TILT_POS_3.95DEG")) return "确认该裂隙倾角正向偏转 +3.95°";
    if (s.includes("EXT_1.00")) return "确认延展系数为 1.00×（基准边界）";
    if (s.includes("EXT_1.15")) return "确认延展系数为 1.15×（外扩 15%）";
    if (s.includes("ATTRIB_NONE")) return "确认交界带不作偏向归属（基准）";
    if (s.includes("ATTRIB_TO_F1")) return "确认交界支持点归属主裂隙 F1";
    if (s.includes("ATTRIB_TO_F2")) return "确认交界支持点归属次裂隙 F2";
    return s;
  }

  function friendlyConditionSummary(rawCond, branchId) {
    const c = String(rawCond || "");
    if (c === "Confirm branch TILT_0DEG") return "复测倾角无偏转 (Δθ = 0°)，排除 ±3.95° 微扰模态";
    if (c === "Confirm branch TILT_NEG_3.95DEG") return "复测倾角负向偏转 (Δθ = -3.95°)";
    if (c === "Confirm branch TILT_POS_3.95DEG") return "复测倾角正向偏转 (Δθ = +3.95°)";
    if (c === "Confirm branch EXT_1.00") return "确认迹线延展维持 1.00× 基准观测边界";
    if (c === "Confirm branch EXT_1.15") return "确认迹线延展外扩至 1.15×";
    if (c === "Confirm branch ATTRIB_NONE") return "确认交界区维持基准分割";
    if (c === "Confirm branch ATTRIB_TO_F1") return "确认交界区划归 F1 裂隙面";
    if (c === "Confirm branch ATTRIB_TO_F2") return "确认交界区划归 F2 裂隙面";
    if (c.startsWith("Confirm t ")) return `精测法向位移满足 ${c.replace("Confirm ", "")}`;
    return c || branchId || "";
  }

  function friendlyAttribution(rawText) {
    return String(rawText || "")
      .replace("TILT_0DEG/TILT_NEG_3.95DEG/TILT_POS_3.95DEG", "0° / -3.95° / +3.95° 倾角分支")
      .replace("EXT_1.00/EXT_1.15", "1.00× / 1.15× 延展分支")
      .replace("ATTRIB_NONE/ATTRIB_TO_F1/ATTRIB_TO_F2", "基准 / F1 / F2 归属分支");
  }

  function friendlyBaselineFailure(code) {
    return {
      MASKS_DISCONNECTED_WITNESS: "掩盖断割反例（假阳性）",
      FALSE_UNION_PATH: "伪造虚假并集路径",
      SILENT_FALSE_SAFE_FLIP: "名义假安全翻转",
      SILENT_MISSED_CONNECTION_FLIP: "名义漏报贯通翻转",
      OBSERVED_CONTRAST: "预算外单列对照",
      AGREES_WITH_CERTIFIED: "与精确认证一致",
    }[code] || code || "基线对照";
  }

  /* 用一句大白话总结当前案例的核心意义（零基础也能秒懂） */
  function plainInsightForCase(cur, eff) {
    if (!cur || !eff) return "";
    if (eff.collapsed && eff.collapsed_to_certified) {
      return eff.status === "CERTIFIED_CONNECTED"
        ? `仅凭 ${eff.step_count} 步针对性复测，原先的几何歧义已完全消除：端点 A 到 B 的绿色裂隙连通路径现已获得 100% 精确有理证明。`
        : `仅凭 ${eff.step_count} 步针对性复测，已彻底排除所有连通可能：端点 A 与 B 被物理阻断，系统已给出确定断开证明。`;
    }
    if (eff.collapsed && !eff.collapsed_to_certified) {
      return "第 1 步复测已大幅收窄不确定范围。请继续确认下方推荐的第 2 步裂隙，即可彻底锁定唯一结论。";
    }
    if (cur.status === "BUDGET_EXCEEDED") {
      return "该簇含 2 处真实孔洞且拓扑极其复杂，超出精确求解预算。系统诚实单列拒答，绝不用近似算法糊弄结论。";
    }
    if (cur.status === "CERTIFIED_CONNECTED") {
      return "无论观测误差如何允许变形，端点 A 与 B 之间始终存在连续裂隙通路，无需再做额外现场复测。";
    }
    if (cur.status === "CERTIFIED_DISCONNECTED") {
      const gapM = cur.disconnected_witness && cur.disconnected_witness.cut_gap
        ? Number(cur.disconnected_witness.cut_gap.distance_m)
        : null;
      return `在所有合法误差范围内，端点 A 与 B 始终被完整岩桥阻断${gapM !== null ? `（最小物理间隔 ${gapM < 0.05 ? `${(gapM * 1000).toFixed(1)} 毫米` : `${gapM.toFixed(2)} 米`}）` : ""}，无需额外复测。`;
    }
    const dw = eff.disconnected_witness;
    const gapM = dw && dw.cut_gap ? Number(dw.cut_gap.distance_m) : null;
    const gapStr = gapM !== null
      ? (gapM < 0.05 ? `仅 ${(gapM * 1000).toFixed(2)} 毫米` : `${gapM.toFixed(3)} 米`)
      : "极小间隙";
    const holeCnt = (cur.fractures || []).reduce((s, f) => s + (Number(f.preserved_hole_count) || 0), 0);
    const holeHint = holeCnt > 0 ? `（含 ${holeCnt} 处真实镂空孔洞）` : "";
    return `同一批观测数据下${holeHint}，裂隙在误差范围内微移 ${gapStr} 就会在「贯通」与「断开」间翻转。点击下方 #1 优先裂隙即可验证如何快速定音。`;
  }

  /* ---------------- 1 分钟 3 步答辩导览脚本 ---------------- */
  const GUIDED_TOUR_STEPS = [
    {
      step: 1,
      clusterId: "jardim_c04",
      siteFilter: "all",
      shortTitle: "保洞真拓扑 · 1 步坍缩为确定贯通",
      desc: "观察含 1 处真实镂空孔洞的 jardim_c04：琥珀金片为 #1 优先复核临界裂隙，点击右侧首条确认分支或「演示本步」，1 步锁定确定贯通。",
      ctaLabel: "演示本步：1 步确定贯通",
    },
    {
      step: 2,
      clusterId: "jardim_c02",
      siteFilter: "all",
      shortTitle: "2.97 mm 断口分屏对比 · 2 步坍缩",
      desc: "已开启左右 3D 同步分屏：左为翠绿连通见证 (m+)，右为红橙断割见证 (m-)，最小割断口仅 2.97 mm。执行两步复核即可锁定确定断开。",
      ctaLabel: "演示本步：2 步确定断开",
    },
    {
      step: 3,
      clusterId: "perozzo_c02",
      siteFilter: "perozzo_finale_ligure",
      shortTitle: "意大利 Finale Ligure 跨场地验证",
      desc: "已切换至意大利 Perozzo 2024 真实露头（ρ = 0.5796，断口 3.97 mm），验证方法在不同地质场地的一致性。",
      ctaLabel: "聚焦 3.97 mm 断口微距",
    },
  ];

  /* ---------------- 全局应用状态 ---------------- */
  const S = {
    theme: "light",
    loadingCases: true,
    loadingCase: false,
    casesError: null,
    caseError: null,
    casesMeta: null,
    cases: [],
    caseCache: new Map(),

    // 右栏三页签: "explore" (默认极简探索与复核) | "clusters" (22 簇案例) | "details" (基线与几何)
    railTab: "explore",
    showMoreCandidates: false,

    siteFilter: "all",
    statusFilter: "all",
    searchQuery: "",

    selectedClusterId: "jardim_c02",
    currentCase: null,
    selectedFractureId: null,

    witnessView: "nominal",
    splitMode: false,
    autoRotate: false,
    interpAlpha: 50,
    interpActive: false,
    ampFactor: 1,

    collapseState: null,
    applyingCheck: false,

    tourActive: false,
    tourStepIndex: 0,

    provenanceOpen: false,

    // 移动端 (<900px) 底部抽屉三态 ("collapsed" 沉浸3D | "half" 均衡分屏 | "expanded" 展开阅读) 与微扰折叠
    mobileSheetState: "half",
    mobileInterpOpen: false,

    // AI 智能导览与讲解（桌面端整合为右栏第 4 页签「AI 讲解」零遮挡；移动端复用底部抽屉）
    copilotOpen: false,
    copilotBusy: false,
    copilotMode: "auto", // "auto" | "llm" | "offline"
    copilotModel: "",
    copilotBackendAvailable: true,
    copilotMessages: [
      {
        role: "assistant",
        text: "您好，我是 STRATA 智能科研讲解助手。所有回答严格锚定 M1–M3 真实露头数据包与三项 null 边界，并可直接联动左侧 3D 视口。\n\n您可以直接点击下方「新手秒懂」或「现场演示」，或输入任意问题。",
        actionLog: null,
        mode: null,
        streaming: false,
      },
    ],
  };

  let viewer = null;

  function isMobileViewport() {
    return window.innerWidth < 900;
  }

  function setMobileSheetState(nextState) {
    const valid = nextState === "collapsed" || nextState === "expanded" ? nextState : "half";
    S.mobileSheetState = valid;
    const ws = $("#workspace");
    if (ws) ws.dataset.mobileSheet = valid;
    renderMobileSheetHandle();
    if (viewer) {
      window.setTimeout(() => viewer && viewer.resize(), 60);
      window.setTimeout(() => viewer && viewer.resize(), 280);
    }
  }

  /* ---------------- 右下角极简浮动 Toast ---------------- */
  function toast(msg, tone = "info") {
    const root = $("#toast-root");
    if (!root) return;
    const maxToasts = isMobileViewport() ? 1 : 2;
    while (root.children.length >= maxToasts) root.firstElementChild.remove();
    const item = document.createElement("div");
    item.className = `toast-card tone-${tone}`;
    item.innerHTML = `<i class="toast-dot"></i><span>${esc(msg)}</span>`;
    root.appendChild(item);
    window.setTimeout(() => {
      item.classList.add("is-leaving");
      window.setTimeout(() => item.remove(), 220);
    }, 2800);
  }

  function hashlineHTML(hash, testid) {
    const h = String(hash || "");
    if (!h) return `<span class="mono dim">—</span>`;
    const short = h.length > 20 ? `${h.slice(0, 10)}…${h.slice(-8)}` : h;
    return `<span class="sha-pill" title="${esc(h)}" ${testid ? `data-testid="${esc(testid)}"` : ""}>
      <code class="mono">${esc(short)}</code>
      <button type="button" class="icon-mini" data-action="copy-text" data-copy="${esc(h)}" title="复制完整 SHA256">${ICONS.copy}</button>
    </span>`;
  }

  function getEffectiveCaseState(cur) {
    if (!cur) return null;
    const col = S.collapseState;
    if (col && col.cluster_id === cur.cluster_id) {
      return {
        status: col.resulting_status,
        collapsed: true,
        collapsed_to_certified: !!col.collapsed_to_certified,
        root_status: col.root_status || cur.status,
        previous_status: col.previous_status,
        provenance_mode: col.provenance_mode || cur.provenance_mode,
        rho: col.rho || cur.rho,
        connected_witness: col.connected_witness,
        disconnected_witness: col.disconnected_witness,
        cell_count: col.cell_count ?? cur.cell_count,
        check_receipt_sha256: col.check_receipt_sha256,
        branch_label: col.branch_label,
        fracture_id: col.fracture_id,
        branch_id: col.branch_id,
        step_count: col.step_count || 1,
        step1_fracture_id: col.step1_fracture_id || col.fracture_id,
        step1_branch_id: col.step1_branch_id || col.branch_id,
        step1_branch_label: col.step1_branch_label || col.branch_label,
        step2_fracture_id: col.step2_fracture_id || null,
        step2_branch_id: col.step2_branch_id || null,
        next_step: col.next_step || null,
      };
    }
    return {
      status: cur.status,
      collapsed: false,
      collapsed_to_certified: false,
      root_status: cur.status,
      previous_status: cur.status,
      provenance_mode: cur.provenance_mode,
      rho: cur.rho,
      connected_witness: cur.connected_witness,
      disconnected_witness: cur.disconnected_witness,
      cell_count: cur.cell_count,
      check_receipt_sha256: null,
      branch_label: null,
      fracture_id: null,
      branch_id: null,
      step_count: 0,
      step1_fracture_id: null,
      step1_branch_id: null,
      step1_branch_label: null,
      step2_fracture_id: null,
      step2_branch_id: null,
      next_step: null,
    };
  }

  function buildSceneSpec(cur, eff, targetView) {
    if (!cur || !cur.modes_geometry) return null;
    const ref = cur.nominal_reference || { mode: "m0", t: "0/1" };
    let modeId = ref.mode || "m0";
    let tVal = 0.0;
    let wv = targetView;

    const cw = eff.connected_witness;
    const dw = eff.disconnected_witness;

    if (wv === "connected" && cw) {
      modeId = cw.mode || modeId;
      tVal = Number(cw.t_float || 0);
    } else if (wv === "disconnected" && dw) {
      modeId = dw.mode || modeId;
      tVal = Number(dw.t_float || 0);
    } else if (wv === "interpolated" && cw && dw) {
      const alpha = S.interpAlpha / 100;
      modeId = alpha < 0.5 ? (cw.mode || modeId) : (dw.mode || modeId);
      tVal = Number(cw.t_float || 0) + (Number(dw.t_float || 0) - Number(cw.t_float || 0)) * alpha;
      wv = alpha <= 0.35 ? "connected" : alpha >= 0.65 ? "disconnected" : "nominal";
    }

    const modeGeom = cur.modes_geometry[modeId] || Object.values(cur.modes_geometry)[0];
    if (!modeGeom) return null;

    const fractureMeta = {};
    for (const f of cur.fractures || []) fractureMeta[f.id] = f;
    const ranked = (cur.active_check && cur.active_check.ranked_fractures) || [];
    const top1Id = !eff.collapsed && ranked.length ? ranked[0].fracture_id : null;

    const connGeom = cw && cur.modes_geometry[cw.mode] ? cur.modes_geometry[cw.mode] : modeGeom;
    const discGeom = dw && cur.modes_geometry[dw.mode] ? cur.modes_geometry[dw.mode] : modeGeom;

    return {
      cluster_id: cur.cluster_id,
      mode_id: modeId,
      t_value: tVal,
      witness_view: wv,
      status: eff.status,
      endpoints: cur.endpoints,
      triangles: modeGeom.triangles || [],
      connTriangles: connGeom ? connGeom.triangles : null,
      discTriangles: discGeom ? discGeom.triangles : null,
      connT: cw ? Number(cw.t_float || 0) : 0,
      discT: dw ? Number(dw.t_float || 0) : 0,
      interpAlpha: targetView === "interpolated" && cw && dw ? S.interpAlpha / 100 : null,
      ampFactor: S.ampFactor || 1,
      fractureMeta,
      fractureContours: cur.fracture_contours || null,
      pathFractures: cw ? cw.path_fractures : [],
      pathTriangles: cw ? cw.path_triangles : [],
      pathPolyline3d: cw ? cw.path_polyline_3d : [],
      contactPoints: cw ? cw.contact_points : [],
      cutSFractures: dw ? dw.cut_S_fractures : [],
      cutTFractures: dw ? dw.cut_T_fractures : [],
      cutGap: dw ? dw.cut_gap : null,
      top1FractureId: top1Id,
      holeProbes: cur.hole_probes || [],
    };
  }

  function syncViewerScene(keepCamera = false) {
    if (!viewer) return;
    const cur = S.currentCase;
    if (!cur) {
      viewer.setScenes(null, null, { splitMode: false });
      return;
    }
    const eff = getEffectiveCaseState(cur);
    const canSplit = !!(eff.connected_witness && eff.disconnected_witness);
    const activeSplit = S.splitMode && canSplit;

    const vpCard = $("#viewport");
    const secPane = $("#vp-pane-secondary");
    const divider = $("#vp-split-divider");
    const badgePri = $("#vp-badge-primary");
    const badgeSec = $("#vp-badge-secondary");

    if (vpCard) vpCard.dataset.split = activeSplit ? "true" : "false";
    if (secPane) secPane.hidden = !activeSplit;
    if (divider) divider.hidden = !activeSplit;

    if (activeSplit) {
      const leftSpec = buildSceneSpec(cur, eff, "connected");
      const rightSpec = buildSceneSpec(cur, eff, "disconnected");
      if (badgePri) {
        badgePri.hidden = false;
        const cw = eff.connected_witness;
        badgePri.className = "vp-pane-badge is-connected";
        badgePri.innerHTML = `<i class="pdot"></i><span>连通见证 (${esc(cw.mode)}, t=${Number(cw.t_float).toFixed(4)})</span>`;
      }
      if (badgeSec) {
        badgeSec.hidden = false;
        const dw = eff.disconnected_witness;
        const gapM = dw.cut_gap ? Number(dw.cut_gap.distance_m) : null;
        badgeSec.className = "vp-pane-badge is-disconnected";
        badgeSec.innerHTML = `<i class="pdot"></i><span>断割见证 (${esc(dw.mode)}, t=${Number(dw.t_float).toFixed(4)})${gapM !== null ? ` · ${(gapM * 1000).toFixed(2)}mm` : ""}</span>`;
      }
      viewer.setScenes(leftSpec, rightSpec, { splitMode: true, keepCamera });
    } else {
      if (badgePri) badgePri.hidden = true;
      if (badgeSec) badgeSec.hidden = true;
      const effectiveView = S.interpActive ? "interpolated" : S.witnessView;
      viewer.setScenes(buildSceneSpec(cur, eff, effectiveView), null, { splitMode: false, keepCamera });
    }
    viewer.setSelectedFracture(S.selectedFractureId);
  }

  /* ---------------- 严格相对路径数据加载 (100% 适配 Caddy /strata/ 子路径) ---------------- */
  async function fetchJsonWithFallback(urls) {
    let lastErr = null;
    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: "no-cache" });
        if (!res.ok) {
          let errBody = null;
          try { errBody = await res.json(); } catch (_e) { /* ignore */ }
          lastErr = (errBody && errBody.error) || { code: `HTTP_${res.status}`, message: `请求 ${url} 失败 (${res.status})` };
          continue;
        }
        return await res.json();
      } catch (err) {
        lastErr = { code: "NETWORK_ERROR", message: String(err && err.message ? err.message : err) };
      }
    }
    throw lastErr || { code: "LOAD_FAILED", message: "无法加载只读数据包" };
  }

  async function loadCasesList() {
    S.loadingCases = true;
    S.casesError = null;
    renderAll(false);
    try {
      const data = await fetchJsonWithFallback([
        "./data/cases.json",
        "./api/identifiability/cases",
      ]);
      const payload = data.result || data;
      S.casesMeta = payload;
      S.cases = payload.cases || [];
      S.loadingCases = false;
      renderAll(false);
      await selectCluster(S.selectedClusterId || (S.cases[0] && S.cases[0].cluster_id) || "jardim_c02");
    } catch (err) {
      S.loadingCases = false;
      S.casesError = err;
      renderAll(false);
    }
  }

  async function selectCluster(clusterId, opts = {}) {
    if (!clusterId) return;
    S.selectedClusterId = clusterId;
    S.caseError = null;
    S.collapseState = null;
    S.interpActive = false;
    S.interpAlpha = 50;
    S.showMoreCandidates = false;
    if (!opts.preserveSplit) S.splitMode = !!opts.enableSplit;
    if (opts.switchToExplore) S.railTab = "explore";

    if (S.caseCache.has(clusterId)) {
      adoptCurrentCase(S.caseCache.get(clusterId), opts);
      return;
    }

    S.loadingCase = true;
    renderAll(false);
    try {
      const raw = await fetchJsonWithFallback([
        `./data/cases/${encodeURIComponent(clusterId)}.json`,
        `./api/identifiability/case?id=${encodeURIComponent(clusterId)}`,
      ]);
      const caseObj = raw.result || raw;
      S.caseCache.set(clusterId, caseObj);
      S.loadingCase = false;
      adoptCurrentCase(caseObj, opts);
    } catch (err) {
      S.loadingCase = false;
      S.caseError = err;
      renderAll(false);
    }
  }

  function adoptCurrentCase(caseObj, opts = {}) {
    S.currentCase = caseObj;
    const ranked = (caseObj.active_check && caseObj.active_check.ranked_fractures) || [];
    const firstFrag = (caseObj.fractures && caseObj.fractures[0] && caseObj.fractures[0].id) || null;
    S.selectedFractureId = ranked.length ? ranked[0].fracture_id : firstFrag;

    if (opts.witnessView) {
      S.witnessView = opts.witnessView;
    } else if (caseObj.status === "CERTIFIED_CONNECTED" && caseObj.connected_witness) {
      S.witnessView = "connected";
    } else if (caseObj.status === "CERTIFIED_DISCONNECTED" && caseObj.disconnected_witness) {
      S.witnessView = "disconnected";
    } else {
      S.witnessView = "nominal";
    }

    if (caseObj.connected_witness && caseObj.disconnected_witness) {
      const tC = Number(caseObj.connected_witness.t_float || 0);
      const tD = Number(caseObj.disconnected_witness.t_float || 0);
      S.interpAlpha = Math.abs(tD - tC) > 1e-6 ? Math.round(clamp(((0 - tC) / (tD - tC)) * 100, 0, 100)) : 50;
    }

    renderAll(true);
    if (opts.autoFocusGap && viewer) {
      window.setTimeout(() => viewer.focusGap(null, true), 140);
    }
  }

  async function applyActiveCheck(params) {
    const cur = S.currentCase;
    if (!cur || S.applyingCheck) return null;
    const { cluster_id, fracture_id, branch_id, step2_fracture_id, step2_branch_id } = params;

    S.applyingCheck = true;
    renderAll(false);
    try {
      let result = null;
      const preMap = cur.collapse_responses || {};
      const key = step2_branch_id
        ? `${fracture_id}::${branch_id}::${step2_fracture_id}::${step2_branch_id}`
        : `${fracture_id}::${branch_id}`;

      if (preMap[key]) {
        result = preMap[key];
      } else {
        const res = await fetch("./api/identifiability/apply_check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            cluster_id,
            fracture_id,
            branch_id,
            step2_fracture_id: step2_fracture_id || undefined,
            step2_branch_id: step2_branch_id || undefined,
          }),
        });
        const data = await res.json();
        if (!res.ok || data.ok === false) {
          throw (data && data.error) || { code: "APPLY_CHECK_FAILED", message: "复核坍缩被拒收" };
        }
        result = data.result || data;
      }

      S.collapseState = result;
      S.interpActive = false;
      if (result.resulting_status === "CERTIFIED_CONNECTED") {
        S.splitMode = false;
        S.witnessView = "connected";
        toast(`已坍缩为「确定贯通」 · 回执 ${String(result.check_receipt_sha256 || "").slice(0, 8)}`, "ok");
      } else if (result.resulting_status === "CERTIFIED_DISCONNECTED") {
        S.splitMode = false;
        S.witnessView = "disconnected";
        toast(`已坍缩为「确定断开」 · 回执 ${String(result.check_receipt_sha256 || "").slice(0, 8)}`, "ok");
      } else {
        toast("第 1 步缩域完成，请继续确认第 2 步", "warn");
      }
      return result;
    } catch (err) {
      toast(`复核拒收：${(err && err.code) || "ERROR"}`, "err");
      return null;
    } finally {
      S.applyingCheck = false;
      renderAll(true);
    }
  }

  function resetCollapse() {
    if (!S.currentCase) return;
    S.collapseState = null;
    S.witnessView = "nominal";
    S.interpActive = false;
    toast(`已重置回未复核初始态 (${S.currentCase.cluster_id})`, "info");
    renderAll(true);
  }

  /* =========================================================================
     STRATA 智能讲解助手 (真 LLM SSE 流式 + 零幻觉真包注入 + 3D 动作联动 + 离线增强兜底)
     ========================================================================= */
  function waitMs(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  function openCopilotTab(focusInput = false) {
    S.railTab = "copilot";
    S.copilotOpen = true;
    if (isMobileViewport() && S.mobileSheetState === "collapsed") {
      setMobileSheetState("half");
    }
    renderTopbar();
    renderCopilotWidget();
    renderRightRail();
    renderMobileSheetHandle();
    renderMobileSheetNav();
    requestAnimationFrame(() => {
      const list = $("#ai-copilot-messages");
      if (list) list.scrollTop = list.scrollHeight;
      if (focusInput && !isMobileViewport()) {
        const inp = $("#ai-copilot-input");
        if (inp) inp.focus();
      }
    });
  }

  function formatCopilotBubbleHTML(text, streaming) {
    const safe = esc(text || "").replace(/\n/g, "<br>");
    return streaming ? `${safe}<span class="cp-cursor" aria-hidden="true"></span>` : safe;
  }

  function scrollCopilotMessagesToBottom() {
    const list = $("#ai-copilot-messages");
    if (list) list.scrollTop = list.scrollHeight;
  }

  function syncStreamingBubbleDOM(msgIdx, msg) {
    const node = $(`[data-cp-msg-idx="${msgIdx}"] .cp-bubble`);
    if (node) {
      node.innerHTML = formatCopilotBubbleHTML(msg.text, msg.streaming);
      scrollCopilotMessagesToBottom();
    } else if (S.railTab === "copilot") {
      renderRightRail();
    }
  }

  function pushCopilotMsg(role, text, actionLog = null, mode = null, streaming = false) {
    const msg = { role, text, actionLog, mode, streaming };
    S.copilotMessages.push(msg);
    if (S.copilotMessages.length > 24) S.copilotMessages.shift();
    if (S.railTab === "copilot") {
      renderRightRail();
      requestAnimationFrame(scrollCopilotMessagesToBottom);
    }
    return S.copilotMessages.length - 1;
  }

  function buildCopilotRequestContext() {
    const cur = S.currentCase;
    const eff = getEffectiveCaseState(cur);
    const dw = eff && (eff.disconnected_witness || (cur && cur.disconnected_witness));
    const gapM = dw && dw.cut_gap && dw.cut_gap.distance_m !== undefined ? Number(dw.cut_gap.distance_m) : null;
    const ranked = (cur && cur.active_check && cur.active_check.ranked_fractures) || [];
    const top1 = ranked[0] || null;
    return {
      cluster_id: cur ? cur.cluster_id : S.selectedClusterId,
      status: eff ? eff.status : null,
      rho: eff && eff.rho && eff.rho.infimum_float !== null ? Number(eff.rho.infimum_float) : null,
      gap_mm: gapM !== null ? Number((gapM * 1000).toFixed(2)) : null,
      split_mode: !!S.splitMode,
      witness_view: S.interpActive ? "interpolated" : S.witnessView,
      collapse_steps: eff && eff.collapsed ? Number(eff.step_count || 1) : 0,
      collapsed_status: eff && eff.collapsed ? eff.status : null,
      top1_fracture_id: top1 ? top1.fracture_id : null,
    };
  }

  function buildCopilotHistory(excludeTailCount = 2) {
    const end = Math.max(0, S.copilotMessages.length - excludeTailCount);
    const slice = S.copilotMessages.slice(1, end); // skip static greeting
    const out = [];
    for (const m of slice) {
      if ((m.role === "user" || m.role === "assistant") && m.text && String(m.text).trim()) {
        out.push({ role: m.role, content: String(m.text).trim().slice(0, 500) });
      }
    }
    return out.slice(-12); // max 6 turns
  }

  /* 受控白名单 3D 视口联动执行器（执行时保持用户停留在 AI 讲解页签，绝不打断阅读） */
  async function executeCopilotAction(action) {
    if (!action || !action.type) return null;
    const kind = String(action.type);
    const arg = String(action.arg || "");

    if (kind === "select" && arg) {
      S.siteFilter = "all";
      await selectCluster(arg, { enableSplit: false, switchToExplore: false });
      return `已联动：切换至 ${arg}`;
    }

    if (kind === "witness" && (arg === "nominal" || arg === "connected" || arg === "disconnected")) {
      S.splitMode = false;
      S.interpActive = false;
      S.witnessView = arg;
      S.interpAlpha = arg === "connected" ? 0 : arg === "disconnected" ? 100 : 50;
      syncViewerScene(true);
      renderViewportChrome();
      const zh = arg === "connected" ? "连通见证 m+" : arg === "disconnected" ? "断割见证 m-" : "名义态";
      return `已联动：切换至 ${zh}`;
    }

    if (kind === "split_gap") {
      const cur = S.currentCase;
      const eff = getEffectiveCaseState(cur);
      const canSplitHere = !!(cur && eff && eff.connected_witness && eff.disconnected_witness);
      const targetId = canSplitHere ? cur.cluster_id : "jardim_c02";
      if (!canSplitHere) {
        await selectCluster(targetId, { enableSplit: true, switchToExplore: false });
      } else {
        S.splitMode = true;
        S.interpActive = false;
        syncViewerScene(true);
      }
      await waitMs(220);
      if (viewer) viewer.focusGap(null, true);
      renderViewportChrome();
      const effAfter = getEffectiveCaseState(S.currentCase);
      const dw = effAfter && effAfter.disconnected_witness;
      const gm = dw && dw.cut_gap ? Number(dw.cut_gap.distance_m) : null;
      const gStr = gm !== null ? `并聚焦 ${(gm * 1000).toFixed(2)} mm 断口` : "";
      return `已联动：打开 ${targetId} 左右 3D 同步分屏${gStr}`;
    }

    if (kind === "focus_gap") {
      const cur = S.currentCase;
      const eff = getEffectiveCaseState(cur);
      if (!eff || !eff.disconnected_witness || !eff.disconnected_witness.cut_gap) {
        await selectCluster("jardim_c02", { enableSplit: false, witnessView: "disconnected", autoFocusGap: true });
        return "已联动：切换 jardim_c02 并聚焦 2.97 mm 断口";
      }
      if (!S.splitMode) S.witnessView = "disconnected";
      S.interpActive = false;
      syncViewerScene(true);
      if (viewer) viewer.focusGap(eff.disconnected_witness.cut_gap, true);
      renderViewportChrome();
      const dMm = (Number(eff.disconnected_witness.cut_gap.distance_m) * 1000).toFixed(2);
      return `已联动：微距聚焦 ${cur.cluster_id} 断口 (${dMm} mm)`;
    }

    if (kind === "collapse_step") {
      const cur = S.currentCase;
      const eff = getEffectiveCaseState(cur);
      if (cur && eff && eff.status === "AMBIGUOUS_WITH_VERIFIED_WITNESSES") {
        if (!eff.collapsed) {
          const r1 = ((cur.active_check || {}).ranked_fractures || [])[0];
          if (r1 && r1.branches && r1.branches[0]) {
            const res1 = await applyActiveCheck({
              cluster_id: cur.cluster_id,
              fracture_id: r1.fracture_id,
              branch_id: r1.branches[0].branch_id,
            });
            const sm1 = statusMeta(res1 && res1.resulting_status);
            return `已联动：${cur.cluster_id} 执行第 1 步复核 → ${sm1.zh}`;
          }
        } else if (eff.next_step && eff.next_step.branches) {
          const b2 = Object.values(eff.next_step.branches).find((b) => b.collapsed_to_certified) || Object.values(eff.next_step.branches)[0];
          if (b2) {
            const res2 = await applyActiveCheck({
              cluster_id: cur.cluster_id,
              fracture_id: eff.step1_fracture_id,
              branch_id: eff.step1_branch_id,
              step2_fracture_id: eff.next_step.fracture_id,
              step2_branch_id: b2.branch_id,
            });
            const sm2 = statusMeta(res2 && res2.resulting_status);
            return `已联动：${cur.cluster_id} 完成第 2 步复核 → ${sm2.zh}`;
          }
        }
      }
      return await executeCopilotAction({ type: "demo_1step" });
    }

    if (kind === "demo_1step") {
      await selectCluster("jardim_c04", { enableSplit: false, witnessView: "nominal", switchToExplore: false });
      await waitMs(260);
      const r1 = ((S.currentCase && S.currentCase.active_check && S.currentCase.active_check.ranked_fractures) || [])[0];
      if (r1 && r1.branches && r1.branches[0]) {
        await applyActiveCheck({
          cluster_id: "jardim_c04",
          fracture_id: r1.fracture_id,
          branch_id: r1.branches[0].branch_id,
        });
      }
      return "已联动：jardim_c04 确认 #1 裂隙 → 1 步坍缩为确定贯通";
    }

    if (kind === "demo_2step") {
      await selectCluster("jardim_c02", { enableSplit: false, witnessView: "nominal", switchToExplore: false });
      await waitMs(220);
      const r1 = ((S.currentCase && S.currentCase.active_check && S.currentCase.active_check.ranked_fractures) || [])[0];
      if (r1 && r1.branches && r1.branches[0]) {
        const res1 = await applyActiveCheck({
          cluster_id: "jardim_c02",
          fracture_id: r1.fracture_id,
          branch_id: r1.branches[0].branch_id,
        });
        await waitMs(280);
        const ns = res1 && res1.next_step;
        if (ns && ns.branches) {
          const b2 = Object.values(ns.branches).find((b) => b.collapsed_to_certified) || Object.values(ns.branches)[1];
          if (b2) {
            await applyActiveCheck({
              cluster_id: "jardim_c02",
              fracture_id: r1.fracture_id,
              branch_id: r1.branches[0].branch_id,
              step2_fracture_id: ns.fracture_id,
              step2_branch_id: b2.branch_id,
            });
          }
        }
      }
      return "已联动：jardim_c02 两步复核 → 坍缩为确定断开";
    }

    if (kind === "open_verify") {
      S.provenanceOpen = true;
      renderProvenanceDrawer();
      return "已联动：打开 SHA256 证据链与三项 null 边界抽屉";
    }

    if (kind === "reset") {
      if (S.collapseState) resetCollapse();
      S.splitMode = false;
      syncViewerScene(true);
      if (viewer) viewer.fit(true);
      renderViewportChrome();
      return "已联动：复位 3D 全景视角与未复核初始态";
    }

    if (kind === "theme" && (arg === "light" || arg === "dark")) {
      S.theme = arg;
      document.documentElement.dataset.theme = arg;
      if (viewer) viewer.setTheme(arg);
      renderTopbar();
      return `已联动：切换至${arg === "dark" ? "深地岩芯夜览" : "浅色科研画廊"}主题`;
    }

    return null;
  }

  /* 离线增强知识库（无 API Key / 纯静态部署 / 上游超时自动零错降级） */
  function buildOfflineEnhancedAnswer(rawText) {
    const q = String(rawText || "").trim();
    const low = q.toLowerCase();

    // 1) 6,034 万点数据清洗 / 1,034 对象 / 44→46 面 / 分簇原理
    if (low.includes("6000") || low.includes("6,034") || low.includes("点云") || low.includes("怎么变") || low.includes("分簇") || low.includes("清洗") || low.includes("1034") || low.includes("1,034") || low.includes("北山")) {
      return {
        text: "以巴西 Jardim Garcia 玄武岩露头为例，6,034 万点并不是主观删片挑出来的，而是逐级可溯源加工：\n1) 全云 60,343,162 点 → 按预注册固定 6 m×6 m 窗口无抽稀裁出 1,279,478 点（约 2.1%）；\n2) 几何核自动拟合 44 个有限面与 946 条两两关系（工作台界面合记 1,034 个对象，绝非「从 1,034 片筛出 46 片」）；\n3) 经 3 步溯源编辑（增补 1、重拟合 1、拆分 1）形成 46 个有限面（其中 4 面含真实孔洞），再按空间邻近与非正交规则聚出 6 簇（其中 3 簇保洞）。\n\n加上意大利 Finale Ligure（8,321 万点 + 1,405 专家面，按 2×2 象限建 16 簇）合计 22 簇；中国北山 22 井 880 条编录因无三维真值，仅作字段格式参照，不进 22 簇。",
        action: null,
      };
    }

    // 2) 残差公式 σ = mean(|r|) + std(|r|) 与 k=2 设计系数
    if (low.includes("残差") || low.includes("公式") || low.includes("乘2") || low.includes("乘 2") || low.includes("2倍") || low.includes("标准差") || low.includes("方差") || low.includes("k=2") || low.includes("模式")) {
      return {
        text: "合法参数域由实测点面残差定标，再显式声明设计系数：\n• 残差尺度公式：σ = mean(|rᵢ|) + std(|rᵢ|)，明确取绝对残差的均值加「标准差 std」（不是方差 var！），与意大利专家表 MeanAbsError + StDevAbsError 口径一致；\n• 法向平移速度：vᵢ = sᵢ · k · σᵢ · nᵢ，其中 k = 2 是预注册冻结的「设计选择」，用来构建受控扰动包络，并不是物理定律，也不宣称覆盖地下全部可能；\n• 每簇同步纳入 6 个离散模式（延展 1.00×/1.15×、边界互斥归属及 ±3.95° 精确有理倾角旋转，cosθ=420/421, sinθ=29/421）。",
        action: null,
      };
    }

    // 3) 为何用有理数（分数）算 2.97 mm / 量化误差与有限面近似
    if (low.includes("分数") || low.includes("有理") || low.includes("浮点") || low.includes("3mm") || low.includes("3 mm") || low.includes("量化") || low.includes("hausdorff")) {
      return {
        text: "因为在真实露头 jardim_c02 中，最小标称片间间隙仅 0.00297 m（2.97 毫米）。普通浮点容差一旦设偏，通断结论会无提示静默翻转。\n• 精确方案：先将局部坐标吸附到 Tier-2 有理网格（步长 0.1 mm，巴西最大欧氏量化界 0.000248 m，全 22 簇最大 0.000640 m 见于 perozzo_c06，共面精确保持），再用精确有理分数求接触窗口并切分为 10,551 个 POINT/OPEN 原子格（21 完备簇枚举 132,000 对三角形，纯 CPU 89.42 秒）；\n• 两本账分清：精确指量化后的运算，前置量化敏感性（如 perozzo_c02 标 quantization_sensitive=true）与有限面近似（jardim_c02 最大 Hausdorff 偏差 0.722811 m）均单独记账。",
        action: { type: "focus_gap" },
      };
    }

    // 4) 红绿金颜色与画面标记解释
    if (low.includes("红绿") || low.includes("颜色") || low.includes("标记") || low.includes("图例") || low.includes("怎么看")) {
      const cur = S.currentCase;
      const eff = getEffectiveCaseState(cur);
      const sm = eff ? statusMeta(eff.status) : statusMeta("");
      const rhoStr = eff && eff.rho && eff.rho.infimum_float !== null ? `稳定裕度 ρ = ${Number(eff.rho.infimum_float).toFixed(4)}` : "已确定态（ρ = null）";
      const dw = eff && eff.disconnected_witness;
      const gapStr = dw && dw.cut_gap ? `最小割断口 ${(Number(dw.cut_gap.distance_m) * 1000).toFixed(2)} mm` : "无断割间隙";
      return {
        text: `当前画布为 ${cur ? cur.cluster_id : "—"}（${sm.zh} · ${rhoStr} · ${gapStr}），颜色语义一目了然：\n• 青绿标桩 A / 深蓝标桩 B：待判贯通性的起点与终点；\n• 翠绿光链与绿面：连通见证 (m+, t+) 下的真实逐点接触通路；\n• 红蓝分侧与红色断口卡尺：断割见证 (m-, t-) 下的 S/T 零跨边断割与最小间隙；\n• 琥珀金色裂隙面：算法推荐的「#1 优先复核临界片」，测它最省力；\n• 青玉色立体圆环：严格保留的真实岩面镂空孔洞（禁凸包填洞）。`,
        action: null,
      };
    }

    // 5) 与 Dips / FracMan / CloudCompare 对比 & 前期转向
    if (low.includes("dips") || low.includes("fracman") || low.includes("cloudcompare") || low.includes("软件") || low.includes("对比") || low.includes("创新") || low.includes("神经网络") || low.includes("深度学习") || low.includes("转向")) {
      return {
        text: "STRATA 不替代也不贬低现成商业软件，而是补上「结论可信度认证与复核排序」中间环：\n• Dips 偏产状统计，FracMan 偏 DFN 建模与常规连通分析，CloudCompare/DSE 偏点云提取（注：同一意大利露头自动 DSE 202,371 碎片 vs 专家 1,405 面，尺度差约 144 倍，不能直接拿碎片算通断）；\n• 前期研究显示神经网络未稳定超过几何基线，且产状统计难以直接回答两点通断；因此转向有限合法域精确认证、通断双见证与零泄漏主动复核。",
        action: null,
      };
    }

    // 6) 22 簇定量对照 / 三方对照 / 0/21 / 2.05 vs 4.65 步
    if (low.includes("22") || low.includes("21") || low.includes("三方") || low.includes("静默翻转") || low.includes("并集") || low.includes("名义") || low.includes("2.05") || low.includes("4.65") || low.includes("步数")) {
      return {
        text: "全部 22 簇中，21 簇完成精确认证（1 确定贯通 perozzo_c04 / 14 确定断开 / 6 有据歧义），另 1 簇 jardim_c01 超 4,096 格上限诚实拒答：\n• 名义单态：6/21（28.57%）发生静默翻转（3 断→通、3 通→断）；\n• 朴素并集图：在 14 个真断开簇上 0/14 误报（如实肯定），但在 6 个歧义簇上 6/6 全单边报通，掩盖断开反例；\n• 本文精确认证：在声明合法域内 0/21 静默误判（绝不说成地下真值 100% 准确率）；\n• 主动复核仿真（6 歧义簇、6,045 根原子格、零泄漏）：测度分裂平均 2.05 步（首步消歧 57.02%）vs 随机 4.65 步，注意这是无量纲仿真步数，不是现场工时（human_time = null）。",
        action: null,
      };
    }

    // 7) 超预算拒答 (jardim_c01 · 4,096 格上限)
    if (low.includes("超预算") || low.includes("拒答") || low.includes("4096") || low.includes("4,096") || low.includes("5024") || low.includes("c01")) {
      return {
        text: "已为您定位超预算诚实拒答案例 jardim_c01：该簇含 2 处真实镂空孔洞，精确分区推进至模态 m4 时累计 3,358 + 835 = 4,193 格，突破预注册 4,096 格上限（全 6 模式总需 5,024 格）。按学术红线，系统拒绝凸包填洞或截断近似，单列登记 BUDGET_EXCEEDED（不塞进 21 的分母，也不解释为断开），并由 M3 核验拒答状态一致。",
        action: { type: "select", arg: "jardim_c01" },
      };
    }

    // 8) 唯一贯通案例 perozzo_c04
    if (low.includes("唯一贯通") || low.includes("确定贯通是哪") || low.includes("perozzo_c04")) {
      return {
        text: "全部 22 簇中，初始态即为「确定贯通」的是意大利 Finale Ligure 场地的 perozzo_c04：在全部 1,002 个精确有理单元下端点 A 到 B 始终存在连续接触路径，无需额外复测（但几何贯通不等于水力导通，hydraulic = null）。",
        action: { type: "select", arg: "perozzo_c04" },
      };
    }

    // 9) 1 步贯通 / 2 步断开演示
    if (low.includes("1步") || low.includes("一步") || low.includes("c04")) {
      return {
        text: "即将为您演示 jardim_c04（含 1 处真实孔洞，1,755 格）的 1 步坍缩：确认 #1 优先裂隙的参数带左支后，状态由「有据歧义」直接坍缩为「确定贯通」并生成 SHA256 回执。（严谨边界：top_guarantees_1step = false，不承诺所有分支一步完备。）",
        action: { type: "demo_1step" },
      };
    }
    if (low.includes("2步") || low.includes("两步")) {
      return {
        text: "即将为您演示 jardim_c02（751 格）的两步接续复核：第 1 步确认 #1 裂隙 0° 倾角（6 模式缩至 2 模式仍歧义），第 2 步确认边界归属后锁定「确定断开」。",
        action: { type: "demo_2step" },
      };
    }

    // 10) 分屏对比 / 断口聚焦
    if (low.includes("分屏") || low.includes("双见证对比") || low.includes("左右对比")) {
      return {
        text: "已为您开启左右 3D 同步分屏并聚焦 S/T 最小割断口！左侧绿色为连通见证 (m+)，右侧红蓝为断割见证 (m-)，两侧相机实时同步旋转，且 AI 讲解收在右侧页签中，绝不遮挡画面。",
        action: { type: "split_gap" },
      };
    }
    if (low.includes("断口") || low.includes("放大") || low.includes("聚焦") || low.includes("2.97") || low.includes("间隙")) {
      return {
        text: "已将 3D 镜头微距推进至 S/T 最小割断口（如 jardim_c02 仅 2.97 mm = 0.00297 m）。正因误差量级仅数毫米，才需要用精确有理分数消除浮点容差静默翻转。",
        action: { type: "focus_gap" },
      };
    }

    // 11) 主动复核一步
    if (low.includes("复核") || low.includes("坍缩") || low.includes("下一步测")) {
      return {
        text: "已按零泄漏测度分裂排序为您执行推荐的下一步现场复核（只读公开合法域与静态几何，不偷看隐藏世界标签），请在左侧 3D 视口观察状态迁移与 SHA256 回执。",
        action: { type: "collapse_step" },
      };
    }

    // 12) 稳定裕度 ρ
    if (low.includes("稳定裕度") || low.includes("rho") || low.includes("ρ")) {
      const cur = S.currentCase;
      const eff = getEffectiveCaseState(cur);
      const rVal = eff && eff.rho && eff.rho.infimum_float !== null ? Number(eff.rho.infimum_float).toFixed(4) : null;
      return {
        text: `稳定裕度 ρ = min max(|Δtᵣ|/σref, eₘ)，衡量从名义结论到相反通断结论的最小参数代价下确界：\n• 6 个歧义簇上 ρ 为 0.0873–1.6448（均值 0.7339），越小越敏感（如 jardim_c02 仅 0.0873）；\n• 严谨边界：ρ 不是概率、不是置信度，也不是米制距离；确定贯通、确定断开与超预算拒答簇无有限翻转边界，记为 null。${rVal ? `\n当前 ${cur.cluster_id} 的稳定裕度为 ρ = ${rVal}。` : ""}`,
        action: null,
      };
    }

    // 13) 真实孔洞 / 凸包
    if (low.includes("孔洞") || low.includes("保洞") || low.includes("凸包")) {
      return {
        text: "真实岩面常含天然镂空孔洞，若用凸包填洞会凭空造出假连通通路。巴西 46 个有限面中 4 面含真实孔洞、6 簇中 3 簇保洞（两个分母不同）；负控还设置洞心探测点严防隐蔽填洞。已为您切至含 1 处真实孔洞的 jardim_c04（青玉圆环即保洞环）。",
        action: { type: "select", arg: "jardim_c04" },
      };
    }

    // 14) 三项 null / 水力 / SHA256 / M3
    if (low.includes("null") || low.includes("水力") || low.includes("涌水") || low.includes("真值") || low.includes("sha") || low.includes("核验") || low.includes("m3") || low.includes("负控")) {
      return {
        text: "本项目以三重防线和三项法定 null 守住学术底线：\n1) M3 独立判卷器零复用求解器代码，22/22 AGREE（21 簇独立几何重算 + 第 22 簇核拒答），配合 36 项有牙负控 + 290 条平台回归 = 37 项核验；\n2) 三项未实测指标严禁填 0：field_error_rate = null（无地下深部开挖真值）、hydraulic = null（几何贯通是水力必要非充分前提，不报涌水量）、human_time = null（未做野外地质师计时对照）。",
        action: { type: "open_verify" },
      };
    }

    // 15) 点名切换任意 22 簇
    const explicitMatch = (S.cases || []).find((c) => {
      const cid = c.cluster_id.toLowerCase();
      const short = cid.replace("jardim_", "").replace("perozzo_", "");
      return low.includes(cid) || (low.includes("意大利") && low.includes(short)) || (low.includes("jardim") && low.includes(short)) || new RegExp(`\\b${short}\\b`, "i").test(low);
    });
    if (explicitMatch) {
      const sm = statusMeta(explicitMatch.status);
      return {
        text: `已为您切换至 ${explicitMatch.cluster_id}（${friendlySite(explicitMatch.site_id)} · ${sm.zh}），可在左侧 3D 视口自由旋转缩放观察。`,
        action: { type: "select", arg: explicitMatch.cluster_id },
      };
    }

    // 16) 主题 / 复位
    if (low.includes("深色") || low.includes("暗色") || low.includes("夜间")) {
      return { text: "已为您切换至「深地岩芯夜览」暗色主题，适合答辩大屏突出翠绿贯通链与红色断口。", action: { type: "theme", arg: "dark" } };
    }
    if (low.includes("浅色") || low.includes("亮色")) {
      return { text: "已为您切换至「浅色科研画廊」主题。", action: { type: "theme", arg: "light" } };
    }
    if (low.includes("复位") || low.includes("重置")) {
      return { text: "已为您复位 3D 全景视角并恢复未复核初始态。", action: { type: "reset" } };
    }

    // 17) 默认综合科普回答（结合当前真实簇）
    const cur = S.currentCase;
    const eff = getEffectiveCaseState(cur);
    return {
      text: `一句话核心：STRATA 不猜唯一地下网络，而是在实测残差构成的合法参数域内，用精确有理分数算清 A 到 B 是否始终连通——一致才定，不一致就给通/断双见证并推荐下一步复核片。\n\n当前画布是 ${cur ? cur.cluster_id : "真实露头簇"}（${eff ? statusMeta(eff.status).zh : ""}）：${plainInsightForCase(cur, eff)}\n\n您可以直接点击下方「新手秒懂」或「现场演示」按钮，或试着问我：「6000万点怎么变6簇」、「残差公式为何乘2」、「为何用分数算3mm」、「和 FracMan 有什么区别」。`,
      action: null,
    };
  }

  /* 统一提问入口：优先走后端真 LLM SSE 流式（含真包注入与动作联动），无 Key/静态部署/故障时无缝降级至离线增强知识库 */
  async function askCopilot(rawText, opts = {}) {
    const q = String(rawText || "").trim();
    if (!q || S.copilotBusy) return;
    if (q.length > 300) {
      toast("单条提问请控制在 300 字以内", "warn");
      return;
    }

    openCopilotTab(false);
    S.copilotBusy = true;
    pushCopilotMsg("user", q);
    const assistantIdx = pushCopilotMsg("assistant", "", null, null, true);

    const finishWithOffline = async () => {
      S.copilotMode = "offline";
      const ans = buildOfflineEnhancedAnswer(q);
      const msg = S.copilotMessages[assistantIdx];
      if (msg) {
        msg.text = ans.text;
        msg.streaming = false;
        msg.mode = "offline";
      }
      if (ans.action) {
        const log = await executeCopilotAction(ans.action);
        if (msg && log) msg.actionLog = log;
      }
      renderRightRail();
      requestAnimationFrame(scrollCopilotMessagesToBottom);
    };

    try {
      if (opts.forceOffline || !S.copilotBackendAvailable) {
        await waitMs(120);
        await finishWithOffline();
        return;
      }

      const payload = {
        message: q,
        history: buildCopilotHistory(2),
        context: buildCopilotRequestContext(),
        stream: true,
      };

      const res = await fetch("./api/copilot/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream, application/json",
        },
        body: JSON.stringify(payload),
      });

      // 公网防刷限流 429 友好提示
      if (res.status === 429) {
        let rateMsg = "提问频率稍快，请稍后再试，或使用快捷按钮体验离线演示。";
        try {
          const j = await res.json();
          if (j && j.error && j.error.message) rateMsg = j.error.message;
        } catch (_e) { /* ignore */ }
        const msg = S.copilotMessages[assistantIdx];
        if (msg) {
          msg.text = rateMsg;
          msg.streaming = false;
          msg.mode = "limit";
        }
        renderRightRail();
        requestAnimationFrame(scrollCopilotMessagesToBottom);
        return;
      }

      // 纯静态托管无 Python 后端（404/405）→ 标记并静默无缝切入离线知识库
      if (res.status === 404 || res.status === 405) {
        S.copilotBackendAvailable = false;
        await finishWithOffline();
        return;
      }

      if (!res.ok) {
        await finishWithOffline();
        return;
      }

      const ctype = (res.headers.get("content-type") || "").toLowerCase();
      if (ctype.includes("text/event-stream") && res.body && typeof res.body.getReader === "function") {
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buf = "";
        let pendingAction = null;
        let fellBack = false;
        const msg = S.copilotMessages[assistantIdx];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const frames = buf.split("\n\n");
          buf = frames.pop() || "";

          for (const frame of frames) {
            const lines = frame.split("\n");
            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed.startsWith("data:")) continue;
              const rawJson = trimmed.slice(5).trim();
              if (!rawJson || rawJson === "[DONE]") continue;
              let ev = null;
              try { ev = JSON.parse(rawJson); } catch (_e) { continue; }
              if (!ev) continue;

              if (ev.type === "meta") {
                S.copilotMode = "llm";
                S.copilotModel = ev.model || "";
                if (msg) msg.mode = "llm";
              } else if (ev.type === "delta" && ev.text) {
                if (msg) {
                  msg.text += String(ev.text);
                  syncStreamingBubbleDOM(assistantIdx, msg);
                }
              } else if (ev.type === "action" && ev.action) {
                pendingAction = ev.action;
              } else if (ev.type === "fallback" || ev.fallback === true) {
                fellBack = true;
              } else if (ev.type === "done") {
                if (msg) {
                  if (typeof ev.reply === "string" && ev.reply.trim()) {
                    msg.text = ev.reply;
                  }
                  msg.streaming = false;
                  msg.mode = "llm";
                }
                if (ev.action) pendingAction = ev.action;
              }
            }
          }
        }

        if (fellBack || (msg && !msg.text.trim() && !pendingAction)) {
          await finishWithOffline();
          return;
        }

        if (msg) {
          msg.streaming = false;
          if (!msg.mode) msg.mode = "llm";
        }
        if (pendingAction) {
          const log = await executeCopilotAction(pendingAction);
          if (msg && log) msg.actionLog = log;
        }
        renderRightRail();
        requestAnimationFrame(scrollCopilotMessagesToBottom);
        return;
      }

      // JSON 响应（含 fallback: true 自动降级标记）
      const data = await res.json();
      if (data && data.fallback) {
        await finishWithOffline();
        return;
      }
      const msg = S.copilotMessages[assistantIdx];
      if (msg && data && data.reply) {
        msg.text = String(data.reply);
        msg.streaming = false;
        msg.mode = "llm";
        if (data.action) {
          const log = await executeCopilotAction(data.action);
          if (log) msg.actionLog = log;
        }
        renderRightRail();
        requestAnimationFrame(scrollCopilotMessagesToBottom);
      } else {
        await finishWithOffline();
      }
    } catch (_err) {
      await finishWithOffline();
    } finally {
      S.copilotBusy = false;
      if (S.railTab === "copilot") {
        renderRightRail();
        requestAnimationFrame(scrollCopilotMessagesToBottom);
      }
    }
  }

  async function runCopilotPreset(presetId) {
    const promptMap = {
      q_data_60m: { text: "6000万点怎么变成6簇的？1034个对象是什么？", forceOffline: false },
      q_residual_k2: { text: "残差公式为什么乘2？用的是标准差还是方差？", forceOffline: false },
      q_rational_3mm: { text: "为什么要用分数（有理数）算不到3毫米的断口？", forceOffline: false },
      q_colors: { text: "红绿金和圆环分别代表什么，怎么看通断？", forceOffline: false },
      demo_1step_connect: { text: "演示 jardim_c04 的 1 步复核确定贯通", action: { type: "demo_1step" }, offlineReply: "演示完成！已打开含 1 处真实镂空孔洞的 jardim_c04（1,755 格），并对 #1 优先裂隙确认参数带左支：画面瞬间亮起翠绿贯通光链，状态 1 步坍缩为「确定贯通」并生成 SHA256 回执。（严谨边界：top_guarantees_1step = false，不保证所有分支一步完备。）" },
      demo_split_gap: { text: "演示 jardim_c02 左右分屏对比并放大 2.97mm 断口", action: { type: "split_gap" }, offlineReply: "已为您打开 jardim_c02 的左右 3D 同步分屏，并将镜头微距推进至 2.97 mm（0.00297 m）的 S/T 最小割断口！\n左侧绿色为连通见证 (m+)，右侧红蓝为断割见证 (m-)，两侧相机实时同步旋转，且 AI 讲解停靠在右侧页签，零遮挡 3D 画面。" },
      demo_2step_disconnect: { text: "演示 jardim_c02 两步复核坍缩为确定断开", action: { type: "demo_2step" }, offlineReply: "两步接续复核演示完成！在 jardim_c02（751 格）中，第 1 步先确认 #1 裂隙 0° 倾角（6 模式缩至 2 模式仍歧义），第 2 步再确认边界归属，状态立即 100% 坍缩为「确定断开」。" },
      open_verify: { text: "核验 SHA256 证据链与三项法定 null 边界", action: { type: "open_verify" }, offlineReply: "已为您打开「可信溯源与 SHA256 核验抽屉」：M3 独立判卷 22/22 AGREE（21 簇几何重算 + 第 22 簇 jardim_c01 核拒答状态），且地下无条件真值 (field_error_rate)、水力连通/涌水量 (hydraulic) 与野外人工耗时 (human_time) 三项严格保持 null，绝不填 0。" },
    };

    const spec = promptMap[presetId];
    if (!spec) {
      await askCopilot("用大白话讲懂 STRATA 在做什么");
      return;
    }

    // 「现场演示」类快捷按钮直接稳定执行 3D 联动并输出精准解说；「新手秒懂」类直接走真 LLM 流式（无 Key 自动降级）
    if (spec.action) {
      if (S.copilotBusy) return;
      openCopilotTab(false);
      S.copilotBusy = true;
      pushCopilotMsg("user", spec.text);
      try {
        const log = await executeCopilotAction(spec.action);
        pushCopilotMsg("assistant", spec.offlineReply, log, S.copilotMode === "llm" ? "llm" : "offline", false);
      } finally {
        S.copilotBusy = false;
        renderRightRail();
        requestAnimationFrame(scrollCopilotMessagesToBottom);
      }
      return;
    }

    await askCopilot(spec.text, { forceOffline: !!spec.forceOffline });
  }

  async function handleCopilotTextQuery(rawText) {
    await askCopilot(rawText);
  }

  /* ---------------- 界面各模块渲染 ---------------- */
  function renderTopbar() {
    const el = $("#topbar");
    if (!el) return;
    const meta = S.casesMeta || {};
    const total = meta.total_cases || 22;
    const agreeCount = (meta.m3_counts && meta.m3_counts.AGREE) || 22;
    const cur = S.currentCase;
    const eff = getEffectiveCaseState(cur);
    const sm = eff ? statusMeta(eff.status) : statusMeta("");

    el.innerHTML = `
      <div class="tb-left">
        <div class="brand-lockup" title="STRATA 徽记：三维岩体块段 + 两条非正交交叉裂隙面 + A→B 贯通路径">
          <span class="brand-icon">${ICONS.strataBrand}</span>
          <span class="brand-name">STRATA</span>
          <span class="brand-div"></span>
          <span class="brand-sub">岩体裂隙贯通可辨识性三维展示台</span>
        </div>

        <button type="button" class="cluster-picker-pill" data-action="switch-rail-tab" data-tab="clusters" title="点击浏览与切换全部 22 簇真实露头">
          <i class="sdot is-${sm.cls}"></i>
          <span class="mono">${esc(cur ? cur.cluster_id : S.selectedClusterId)}</span>
          <span class="cp-count">22 簇</span>
          ${ICONS.chevronDown}
        </button>
      </div>

      <div class="tb-right">
        ${eff && eff.collapsed ? `
          <button type="button" class="tb-btn is-accent" data-action="ident-reset-collapse" data-testid="topbar-reset-collapse">
            ${ICONS.reset}<span>重置状态</span>
          </button>` : ""}

        <button type="button" class="tb-btn ${S.railTab === "copilot" ? "is-accent" : ""}" data-action="open-copilot-tab" data-testid="topbar-ai-btn" title="切换至右侧 AI 讲解页签（不遮挡 3D 画面）">
          ${ICONS.copilot}
          <span class="lbl-full">AI 讲解</span>
          <span class="lbl-short">AI</span>
        </button>

        <button type="button" class="tb-btn ${S.tourActive ? "is-accent" : ""}" data-action="start-tour" data-testid="btn-guided-tour">
          ${ICONS.tour}
          <span class="lbl-full">1 分钟导览</span>
          <span class="lbl-short">导览</span>
        </button>

        <button type="button" class="tb-trust" data-action="open-provenance" data-testid="id-m3-badge" title="点击打开可信溯源与 SHA256 核验抽屉">
          ${ICONS.shieldOk}
          <span class="lbl-full">独立复算 ${agreeCount}/${total} 一致</span>
          <span class="lbl-short">${agreeCount}/${total} 核验</span>
        </button>

        <button type="button" class="tb-icon-btn" data-action="toggle-theme" data-testid="theme-toggle" title="${S.theme === "dark" ? "切换浅色画廊主题" : "切换深地岩芯夜览"}" aria-label="切换主题">
          ${S.theme === "dark" ? ICONS.sun : ICONS.moon}
        </button>
      </div>
    `;
  }

  function renderHeroHud() {
    const el = $("#statusrow");
    if (!el) return;

    if (S.casesError || S.caseError) {
      const err = S.caseError || S.casesError;
      el.innerHTML = `
        <div class="hero-hud-inner" data-testid="id-status-error">
          <span class="hud-verdict is-refused"><i class="sdot"></i>校验拦截 · ${esc(err.code)}</span>
          <div class="hud-meta">${esc(err.message)}</div>
        </div>`;
      return;
    }

    const cur = S.currentCase;
    if (!cur) {
      el.innerHTML = `
        <div class="hero-hud-inner" data-testid="id-status-badge" data-status="IDLE">
          <div class="hud-kicker">STRATA 3D</div>
          <div class="hud-title">正在核验 22 簇真实露头 SHA256…</div>
        </div>`;
      return;
    }

    const eff = getEffectiveCaseState(cur);
    const sm = statusMeta(eff.status);
    const rootSm = statusMeta(eff.root_status);
    const rhoInf = eff.rho && eff.rho.infimum_float !== null && eff.rho.infimum_float !== undefined
      ? Number(eff.rho.infimum_float).toFixed(4)
      : "—";
    const dw = eff.disconnected_witness || cur.disconnected_witness;
    const gapM = dw && dw.cut_gap && dw.cut_gap.distance_m !== undefined ? Number(dw.cut_gap.distance_m) : null;
    const gapText = gapM !== null
      ? (gapM < 0.05 ? `${(gapM * 1000).toFixed(2)} mm` : `${gapM.toFixed(3)} m`)
      : null;
    const holeCount = (cur.fractures || []).reduce((acc, f) => acc + (Number(f.preserved_hole_count) || 0), 0);

    el.innerHTML = `
      <div class="hero-hud-inner" data-testid="id-status-bar" data-status="${esc(eff.status)}" data-cluster="${esc(cur.cluster_id)}">
        <div class="hud-kicker">${esc(friendlySite(cur.site_id))} · <span class="mono">${esc(cur.cluster_id)}</span></div>
        <div class="hud-title-row">
          <span class="hud-verdict is-${sm.cls}" data-testid="id-status-badge" data-status="${esc(eff.status)}" data-cluster="${esc(cur.cluster_id)}">
            <i class="sdot"></i><span>${esc(sm.zh)}</span>
          </span>
          ${eff.collapsed ? `
            <span class="hud-collapse-note" data-testid="id-collapse-transition">
              ${eff.collapsed_to_certified ? `${esc(rootSm.short)} → ${esc(sm.short)} (${eff.step_count}步)` : "第 1 步缩域"}
            </span>` : ""}
        </div>
        <div class="hud-metrics">
          <span class="hm-item">稳定裕度 ρ <strong class="mono" data-testid="id-rho-value">${esc(rhoInf)}</strong></span>
          ${gapText ? `<span class="hm-sep">·</span><button type="button" class="hm-link" data-action="focus-gap" title="点击微距聚焦断口">断口 <strong class="mono">${esc(gapText)}</strong></button>` : ""}
          <span class="hm-sep">·</span>
          <span class="hm-item"><strong class="mono">${esc(eff.cell_count ?? "—")}</strong> 精确单元</span>
          ${holeCount > 0 ? `<span class="hm-sep">·</span><span class="hm-item is-hole">${ICONS.hole}保洞 ×${holeCount}</span>` : ""}
        </div>
      </div>
    `;
  }

  function renderTourRibbon() {
    const el = $("#tour-banner");
    if (!el) return;
    if (!S.tourActive) {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    el.hidden = false;
    const stepObj = GUIDED_TOUR_STEPS[S.tourStepIndex] || GUIDED_TOUR_STEPS[0];
    const dots = GUIDED_TOUR_STEPS.map((st, idx) => `
      <button type="button" class="tour-dot-btn ${idx === S.tourStepIndex ? "is-current" : idx < S.tourStepIndex ? "is-done" : ""}"
        data-action="tour-goto" data-step="${idx}" data-testid="tour-step-${idx + 1}">
        0${idx + 1} · ${esc(st.clusterId)}
      </button>
    `).join("");

    el.innerHTML = `
      <div class="tour-bar">
        <div class="tour-bar-top">
          <div class="tour-dots">${dots}</div>
          <div class="tour-actions">
            <button type="button" class="tour-cta" data-action="tour-run-demo" data-testid="tour-run-demo">${esc(stepObj.ctaLabel)}</button>
            <button type="button" class="btn-ghost-sm" data-action="tour-prev" ${S.tourStepIndex === 0 ? "disabled" : ""}>上一步</button>
            <button type="button" class="btn-ghost-sm" data-action="tour-next" data-testid="tour-next">${S.tourStepIndex < 2 ? "下一步" : "完成"}</button>
            <button type="button" class="icon-mini" data-action="tour-exit" data-testid="tour-exit" title="退出导览">${ICONS.close}</button>
          </div>
        </div>
        <div class="tour-bar-desc">${esc(stepObj.desc)}</div>
      </div>
    `;
  }

  function renderViewportChrome() {
    const toolsEl = $("#vp-tools");
    const deckEl = $("#vp-witness-bar");
    const legEl = $("#vp-legend");
    const cur = S.currentCase;
    const eff = getEffectiveCaseState(cur);

    if (toolsEl) {
      const hasGap = !!(eff && eff.disconnected_witness && eff.disconnected_witness.cut_gap);
      const camSt = viewer ? viewer.getCameraState() : {};
      toolsEl.innerHTML = `
        <button type="button" class="vp-icon-btn" data-action="fit-view" data-testid="fit-view" title="复位全景">${ICONS.fit}</button>
        <button type="button" class="vp-icon-btn" data-action="top-view" data-testid="top-view" title="俯视平面 (XY)">${ICONS.topView}</button>
        <button type="button" class="vp-icon-btn" data-action="front-view" data-testid="front-view" title="正视剖面 (XZ)">${ICONS.frontView}</button>
        ${hasGap ? `<button type="button" class="vp-icon-btn is-gap ${camSt.focusedOnGap ? "is-active" : ""}" data-action="focus-gap" data-testid="focus-gap-btn" title="微距聚焦 S/T 最小割断口">${ICONS.focusGap}</button>` : ""}
        <button type="button" class="vp-icon-btn ${S.autoRotate ? "is-active" : ""}" data-action="toggle-autorotate" data-testid="autorotate-btn" title="自动旋转展示">${ICONS.rotate}</button>
        <button type="button" class="vp-icon-btn" data-action="shot-view" data-testid="shot-view" title="导出高清 PNG">${ICONS.camera}</button>
      `;
    }

    if (deckEl) {
      if (!cur) {
        deckEl.hidden = true;
        deckEl.innerHTML = "";
      } else {
        deckEl.hidden = false;
        const hasConn = !!eff.connected_witness;
        const hasDisc = !!eff.disconnected_witness;
        const isDual = hasConn && hasDisc;
        const wv = S.interpActive ? "interpolated" : S.witnessView;
        const gapM = hasDisc && eff.disconnected_witness.cut_gap ? Number(eff.disconnected_witness.cut_gap.distance_m) : null;

        let interpHTML = "";
        if (isDual) {
          const tC = Number(eff.connected_witness.t_float || 0);
          const tD = Number(eff.disconnected_witness.t_float || 0);
          const tInterp = (tC + (tD - tC) * (S.interpAlpha / 100)).toFixed(4);
          const zoneText = S.interpAlpha <= 35 ? "连通邻域" : S.interpAlpha >= 65 ? "断割邻域" : "临界翻转带";
          interpHTML = `
            <div class="deck-interp ${S.mobileInterpOpen ? "is-mob-open" : ""}" data-testid="witness-interp-bar">
              <span class="di-k">微扰插值</span>
              <button type="button" class="di-end" data-action="interp-snap" data-alpha="0">连通 m+</button>
              <input type="range" min="0" max="100" step="1" value="${S.interpAlpha}"
                class="di-range" id="witness-interp-slider" data-testid="witness-interp-slider"
                aria-label="见证连续微扰插值滑块">
              <button type="button" class="di-end" data-action="interp-snap" data-alpha="100">断割 m-</button>
              <span class="di-val mono" data-testid="interp-readout">t=${tInterp} · ${zoneText}</span>
              <button type="button" class="di-amp ${S.ampFactor > 1 ? "is-active" : ""}" data-action="toggle-amp" data-testid="amp-toggle" title="切换真实物理尺度或 8× 视觉放大">
                ${S.ampFactor > 1 ? "放大 8×" : "尺度 1×"}
              </button>
              <span class="di-hint">仅供几何微移直观演示 · 状态以精确证书为准</span>
            </div>
          `;
        }

        deckEl.innerHTML = `
          <div class="deck-main">
            <div class="deck-seg" role="group" aria-label="见证态切换">
              <button type="button" class="deck-seg-btn" aria-pressed="${!S.splitMode && wv === "nominal"}"
                data-action="ident-witness" data-witness="nominal" data-testid="witness-btn-nominal">
                <i class="wdot is-nom"></i><span>名义态</span>
              </button>
              <button type="button" class="deck-seg-btn" aria-pressed="${!S.splitMode && wv === "connected"}"
                data-action="ident-witness" data-witness="connected" data-testid="witness-btn-connected" ${hasConn ? "" : "disabled"}>
                <i class="wdot is-conn"></i><span class="lbl-full">连通见证 m+</span><span class="lbl-short">连通 m+</span>
              </button>
              <button type="button" class="deck-seg-btn" aria-pressed="${!S.splitMode && wv === "disconnected"}"
                data-action="ident-witness" data-witness="disconnected" data-testid="witness-btn-disconnected" ${hasDisc ? "" : "disabled"}>
                <i class="wdot is-disc"></i><span class="lbl-full">断割见证 m-</span><span class="lbl-short">断割 m-</span>
              </button>
            </div>

            <div class="deck-extras">
              ${isDual ? `
                <button type="button" class="deck-pill ${S.splitMode ? "is-active" : ""}"
                  aria-pressed="${S.splitMode}" data-action="toggle-split-view" data-testid="split-view-toggle">
                  ${ICONS.split}<span class="lbl-full">${S.splitMode ? "退出分屏" : "左右分屏对比"}</span><span class="lbl-short">${S.splitMode ? "单屏" : "分屏"}</span>
                </button>` : ""}
              ${gapM !== null ? `
                <button type="button" class="deck-pill is-gap" data-action="focus-gap" data-testid="quick-focus-gap">
                  ${ICONS.focusGap}<span class="lbl-full">聚焦 ${gapM < 0.05 ? `${(gapM * 1000).toFixed(2)}mm` : `${gapM.toFixed(3)}m`} 断口</span><span class="lbl-short">${gapM < 0.05 ? `${(gapM * 1000).toFixed(2)}mm` : `${gapM.toFixed(2)}m`}</span>
                </button>` : ""}
              ${isDual ? `
                <button type="button" class="deck-pill mob-only-pill ${S.mobileInterpOpen ? "is-active" : ""}"
                  aria-pressed="${S.mobileInterpOpen}" data-action="toggle-mobile-interp" data-testid="mobile-interp-toggle">
                  ${ICONS.sliders}<span>微扰</span>
                </button>` : ""}
            </div>
          </div>
          ${interpHTML}
        `;
      }
    }

    if (legEl) {
      legEl.innerHTML = `
        <span class="lg-item"><i class="lg-dot ep-a"></i>A 起点</span>
        <span class="lg-item"><i class="lg-dot ep-b"></i>B 终点</span>
        <span class="lg-item"><i class="lg-bar emerald"></i>连通光链</span>
        <span class="lg-item"><i class="lg-bar crimson"></i>S/T 断口</span>
        <span class="lg-item"><i class="lg-bar amber"></i>#1 临界裂隙</span>
        <span class="lg-item"><i class="lg-bar teal"></i>真孔洞环</span>
      `;
    }
  }

  /* ---------------- 桌面端左下角轻量 AI 讲解入口（绝不弹窗遮挡 3D 画面） ---------------- */
  function renderCopilotWidget() {
    const el = $("#ai-copilot-anchor");
    if (!el) return;

    // 已在「AI 讲解」页签时自动隐藏入口胶囊，确保 3D 视口与分屏断口 100% 零遮挡
    if (S.railTab === "copilot") {
      el.innerHTML = "";
      return;
    }

    el.innerHTML = `
      <button type="button" class="copilot-fab" data-action="open-copilot-tab" data-testid="ai-copilot-fab" title="切换到右侧「AI 讲解」页签（边问边联动 3D，零遮挡）">
        <span class="fab-icon">${ICONS.copilot}</span>
        <span class="fab-copy">
          <strong>AI 讲解</strong>
          <span>边问边联动 3D · 零遮挡</span>
        </span>
      </button>
    `;
  }

  /* ---------------- 右栏极简四页签工作台（探索与复核 | 22 簇案例 | 基线与几何 | AI 讲解） ---------------- */
  function renderRightRail() {
    const tabsEl = $("#rail-tabs");
    const bodyEl = $("#ins-scroll");
    if (!tabsEl || !bodyEl) return;

    const total = (S.casesMeta && S.casesMeta.total_cases) || 22;
    const tabs = [
      ["explore", "探索与复核"],
      ["clusters", `22 簇 (${total})`],
      ["details", "基线与几何"],
      ["copilot", "AI 讲解"],
    ];
    tabsEl.innerHTML = tabs.map(([id, lbl]) => `
      <button type="button" role="tab" class="rail-tab ${id === "copilot" ? "is-ai-tab" : ""}" aria-selected="${S.railTab === id}"
        data-action="switch-rail-tab" data-tab="${id}" data-testid="tab-${id}">
        ${id === "copilot" ? ICONS.copilot : ""}<span>${esc(lbl)}</span>
      </button>
    `).join("");

    bodyEl.classList.toggle("is-copilot-mode", S.railTab === "copilot");

    if (S.railTab === "clusters") {
      bodyEl.innerHTML = buildClustersTabHTML();
    } else if (S.railTab === "details") {
      bodyEl.innerHTML = buildDetailsTabHTML();
    } else if (S.railTab === "copilot") {
      bodyEl.innerHTML = buildCopilotTabHTML();
      requestAnimationFrame(scrollCopilotMessagesToBottom);
    } else {
      bodyEl.innerHTML = buildExploreTabHTML();
    }
  }

  function buildCopilotTabHTML() {
    const modeLabel = S.copilotMode === "llm"
      ? `真 LLM 流式${S.copilotModel ? ` · ${S.copilotModel}` : ""}`
      : S.copilotMode === "offline"
      ? "离线增强知识库 · 零幻觉"
      : "真包注入 · 边问边联动 3D";

    const msgsHTML = S.copilotMessages.map((m, idx) => `
      <div class="cp-msg is-${m.role}" data-cp-msg-idx="${idx}">
        <div class="cp-bubble">${formatCopilotBubbleHTML(m.text, m.streaming)}</div>
        ${m.actionLog ? `<div class="cp-action-log" data-testid="cp-action-badge">${ICONS.shieldOk}<span>${esc(m.actionLog)}</span></div>` : ""}
      </div>
    `).join("");

    return `
      <div class="ai-rail-pane" data-testid="ai-copilot-panel" role="region" aria-label="STRATA AI 讲解与联动助手">
        <div class="ai-rail-status">
          <span class="ai-mode-pill ${S.copilotMode === "llm" ? "is-llm" : ""}">
            <i class="sdot ${S.copilotMode === "llm" ? "is-connected" : "is-ambiguous"}"></i>
            <span>${esc(modeLabel)}</span>
          </span>
          <span class="mono dim">${esc(S.currentCase ? S.currentCase.cluster_id : S.selectedClusterId)}</span>
        </div>

        <div class="cp-messages" id="ai-copilot-messages" data-testid="ai-copilot-messages">
          ${msgsHTML}
        </div>

        <div class="cp-preset-groups">
          <div class="cp-group">
            <span class="cp-group-lbl">新手秒懂</span>
            <div class="cp-chips">
              <button type="button" class="cp-chip" data-action="copilot-preset" data-preset="q_data_60m" data-testid="cp-chip-data" ${S.copilotBusy ? "disabled" : ""}>6000万点怎么变6簇</button>
              <button type="button" class="cp-chip" data-action="copilot-preset" data-preset="q_residual_k2" data-testid="cp-chip-residual" ${S.copilotBusy ? "disabled" : ""}>残差公式为何乘2</button>
              <button type="button" class="cp-chip" data-action="copilot-preset" data-preset="q_rational_3mm" data-testid="cp-chip-rational" ${S.copilotBusy ? "disabled" : ""}>为何用分数算3mm</button>
              <button type="button" class="cp-chip" data-action="copilot-preset" data-preset="q_colors" data-testid="cp-chip-colors" ${S.copilotBusy ? "disabled" : ""}>红绿金怎么看通断</button>
            </div>
          </div>
          <div class="cp-group">
            <span class="cp-group-lbl">现场演示</span>
            <div class="cp-chips">
              <button type="button" class="cp-chip is-primary" data-action="copilot-preset" data-preset="demo_1step_connect" data-testid="cp-preset-1step" ${S.copilotBusy ? "disabled" : ""}>1步确定贯通</button>
              <button type="button" class="cp-chip is-primary" data-action="copilot-preset" data-preset="demo_split_gap" data-testid="cp-preset-split" ${S.copilotBusy ? "disabled" : ""}>分屏看2.97mm断口</button>
              <button type="button" class="cp-chip" data-action="copilot-preset" data-preset="demo_2step_disconnect" data-testid="cp-preset-2step" ${S.copilotBusy ? "disabled" : ""}>2步确定断开</button>
              <button type="button" class="cp-chip" data-action="copilot-preset" data-preset="open_verify" data-testid="cp-preset-verify" ${S.copilotBusy ? "disabled" : ""}>核验SHA与三null</button>
            </div>
          </div>
        </div>

        <form class="cp-input-row" id="ai-copilot-form">
          <input type="text" id="ai-copilot-input" data-testid="ai-copilot-input" maxlength="300"
            placeholder="提问或下达指令（如：分屏看断口 / 与 FracMan 区别）" autocomplete="off" ${S.copilotBusy ? "disabled" : ""}>
          <button type="submit" class="cp-send-btn" data-testid="ai-copilot-send" title="发送" ${S.copilotBusy ? "disabled" : ""}>${ICONS.send}</button>
        </form>
      </div>
    `;
  }

  function buildClustersTabHTML() {
    const meta = S.casesMeta || {};
    const sc = meta.site_counts || {};
    const stc = meta.status_counts || {};

    const siteBtns = [
      ["all", `全部 ${meta.total_cases || 22}`],
      ["jardim_garcia_uh", `Jardim ${sc.jardim_garcia_uh || 6}`],
      ["perozzo_finale_ligure", `Perozzo ${sc.perozzo_finale_ligure || 16}`],
    ].map(([id, lbl]) => `
      <button type="button" class="site-seg-btn" aria-selected="${S.siteFilter === id}"
        data-action="ident-site" data-site="${id}" data-testid="ident-site-${id}">${esc(lbl)}</button>
    `).join("");

    const stFilters = [
      ["all", "全部"],
      ["AMBIGUOUS_WITH_VERIFIED_WITNESSES", `歧义 ${stc.AMBIGUOUS_WITH_VERIFIED_WITNESSES || 6}`],
      ["CERTIFIED_CONNECTED", `贯通 ${stc.CERTIFIED_CONNECTED || 1}`],
      ["CERTIFIED_DISCONNECTED", `断开 ${stc.CERTIFIED_DISCONNECTED || 14}`],
      ["BUDGET_EXCEEDED", `超预算 ${stc.BUDGET_EXCEEDED || 1}`],
    ].map(([id, lbl]) => `
      <button type="button" class="st-link ${S.statusFilter === id ? "is-active" : ""}"
        data-action="status-filter" data-status-filter="${id}">${esc(lbl)}</button>
    `).join("");

    const q = S.searchQuery.trim().toLowerCase();
    const rows = (S.cases || []).filter((c) => {
      if (S.siteFilter !== "all" && c.site_id !== S.siteFilter) return false;
      if (S.statusFilter !== "all" && c.status !== S.statusFilter) return false;
      if (q && !c.cluster_id.toLowerCase().includes(q) && !friendlyStratum(c.stratum).toLowerCase().includes(q)) return false;
      return true;
    });

    const listHTML = rows.length
      ? rows.map((c) => {
        const sm = statusMeta(c.status);
        const siteShort = c.site_id === "jardim_garcia_uh" ? "Jardim" : "Perozzo";
        const isSel = S.selectedClusterId === c.cluster_id;
        const parts = [siteShort, `${c.fracture_count}片/${c.triangles_per_mode}△`];
        if (c.rho_float !== null && c.rho_float !== undefined) parts.push(`ρ=${Number(c.rho_float).toFixed(3)}`);
        if (c.cut_gap_distance_m !== null && c.cut_gap_distance_m !== undefined && c.status === "AMBIGUOUS_WITH_VERIFIED_WITNESSES") {
          parts.push(`断口 ${(Number(c.cut_gap_distance_m) * 1000).toFixed(2)}mm`);
        }
        if (c.preserved_hole_count > 0) parts.push(`保洞×${c.preserved_hole_count}`);

        return `
          <button type="button" class="cluster-row ${isSel ? "is-selected" : ""}"
            data-action="ident-select-cluster" data-cluster="${esc(c.cluster_id)}"
            data-testid="id-cluster-row-${esc(c.cluster_id)}">
            <div class="cr-top">
              <span class="cr-id mono">${esc(c.cluster_id)}</span>
              <span class="cr-st is-${sm.cls}"><i class="sdot"></i>${esc(sm.short)}</span>
            </div>
            <div class="cr-meta mono">${esc(parts.join(" · "))}</div>
          </button>
        `;
      }).join("")
      : `<div class="empty-state">未找到匹配裂隙簇</div>`;

    return `
      <div class="clusters-pane">
        <div class="search-field">
          ${ICONS.search}
          <input id="ident-search" type="search" value="${esc(S.searchQuery)}" placeholder="搜索簇编号（如 jardim_c04）…" autocomplete="off">
        </div>
        <div class="site-seg">${siteBtns}</div>
        <div class="st-filter-bar">${stFilters}</div>
        <div class="cluster-list" id="ledger-list" role="listbox">${listHTML}</div>
      </div>
    `;
  }

  function buildExploreTabHTML() {
    const cur = S.currentCase;
    if (!cur) return `<div class="empty-state">正在加载真实露头裂隙簇…</div>`;
    const eff = getEffectiveCaseState(cur);
    const sm = statusMeta(eff.status);
    const ac = cur.active_check || {};
    const ranked = ac.ranked_fractures || [];

    // 1) 一句话大白话洞察
    const insightHTML = `
      <div class="insight-box">
        <div class="insight-kicker">通俗解读</div>
        <p class="insight-text">${esc(plainInsightForCase(cur, eff))}</p>
      </div>
    `;

    // 2) 主动复核决策区
    let decisionHTML = "";
    if (eff.collapsed) {
      const ns = !eff.collapsed_to_certified ? eff.next_step : null;
      const nsBranches = ns && ns.branches ? Object.values(ns.branches) : [];
      const step2HTML = ns && nsBranches.length ? `
        <div class="step2-box" data-testid="id-next-step-panel">
          <div class="step2-head">
            <span class="step-label">第 2 步接续复核</span>
            <strong class="mono">${esc(ns.short_label || ns.fracture_id)}</strong>
          </div>
          <div class="step2-sub">${esc(ns.action_description || friendlyActionType(ns.action_type))}</div>
          <div class="branch-stack">
            ${nsBranches.map((sb, sIdx) => {
              const outSm = statusMeta(sb.resulting_status);
              const outTxt = sb.resulting_status === "CERTIFIED_CONNECTED" ? "确定贯通" : sb.resulting_status === "CERTIFIED_DISCONNECTED" ? "确定断开" : "继续细化";
              return `
                <button type="button" class="branch-btn is-${outSm.cls}"
                  data-action="ident-apply-check"
                  data-cluster="${esc(cur.cluster_id)}"
                  data-fracture="${esc(eff.step1_fracture_id)}"
                  data-branch="${esc(eff.step1_branch_id)}"
                  data-step2-fracture="${esc(ns.fracture_id)}"
                  data-step2-branch="${esc(sb.branch_id)}"
                  data-testid="apply-step2-b${sIdx}" ${S.applyingCheck ? "disabled" : ""}>
                  <div class="bb-top">
                    <span class="bb-title">${esc(friendlyBranchLabel(sb.label, sb.condition_summary, sb.branch_id))}</span>
                    <span class="bb-out is-${outSm.cls}">${ICONS.arrowRight}<span>${esc(outTxt)}</span></span>
                  </div>
                  <div class="bb-sub">${esc(friendlyConditionSummary(sb.condition_summary, sb.branch_id))}</div>
                </button>`;
            }).join("")}
          </div>
        </div>` : "";

      decisionHTML = `
        <div class="decision-block" data-testid="id-collapse-result">
          <div class="block-head">
            <span>${eff.collapsed_to_certified ? "坍缩验证回执" : "第 1 步缩域回执"}</span>
            <span class="mono dim">${eff.step_count} 步复核</span>
          </div>
          <dl class="kv-clean">
            ${eff.step_count === 2 ? `<dt>第 1 步</dt><dd>${esc(friendlyBranchLabel(eff.step1_branch_label, "", eff.step1_branch_id))}</dd>` : ""}
            <dt>已确认分支</dt><dd><strong>${esc(friendlyBranchLabel(eff.branch_label, "", eff.branch_id))}</strong></dd>
            <dt>状态变化</dt><dd>${esc(statusMeta(eff.previous_status).zh)} → <strong class="st-ink is-${sm.cls}">${esc(sm.zh)}</strong></dd>
            <dt>回执 SHA</dt><dd>${hashlineHTML(eff.check_receipt_sha256, "id-check-receipt-sha")}</dd>
          </dl>
          ${step2HTML}
          <button type="button" class="btn-solid-full" data-action="ident-reset-collapse" data-testid="id-reset-collapse">
            ${ICONS.reset}<span>一键重置回到未复核态</span>
          </button>
        </div>
      `;
    } else if (cur.status === "BUDGET_EXCEEDED" || cur.status === "REFUSED") {
      decisionHTML = `
        <div class="decision-block" data-testid="id-active-check-panel">
          <div class="block-head"><span>诚实超预算拒答信封</span><span class="mono dim">BUDGET_EXCEEDED</span></div>
          <p class="body-p">本簇（<code>jardim_c01</code>）含 2 处真实孔洞，精确分区推进至模态 <code>m4</code> 时累计达 <code>3,358 + 835 = 4,193 &gt; 4,096</code> 格上限（全 6 模式总需求 5,024 格）。系统拒绝凸包填洞或浮点近似，单列登记并经 M3 独立复核一致。</p>
        </div>
      `;
    } else if (cur.status !== "AMBIGUOUS_WITH_VERIFIED_WITNESSES") {
      decisionHTML = `
        <div class="decision-block" data-testid="id-active-check-panel">
          <div class="block-head"><span>全参数域已确定</span><span class="st-ink is-${sm.cls}">${esc(sm.zh)}</span></div>
          <p class="body-p">在全部 ${esc(cur.cell_count)} 个精确有理单元下结论唯一，无需追加现场复测。</p>
        </div>
      `;
    } else {
      const renderRankCard = (r, isPrimary) => {
        const resPct = Math.round((r.resolved_measure_fraction || 0) * 100);
        const certPct = Math.round((r.certified_branch_fraction || 0) * 100);
        const branchesHTML = (r.branches || []).length ? `
          <div class="branch-stack">
            ${r.branches.map((b, bIdx) => {
              const outSm = statusMeta(b.resulting_status);
              const outTxt = b.resulting_status === "CERTIFIED_CONNECTED"
                ? "1 步确定贯通"
                : b.resulting_status === "CERTIFIED_DISCONNECTED"
                ? "1 步确定断开"
                : b.has_next_step
                ? `进第 2 步 (${b.next_step_fracture_label || "次级片"})`
                : "缩减歧义";
              return `
                <button type="button" class="branch-btn is-${outSm.cls}"
                  data-action="ident-apply-check"
                  data-cluster="${esc(cur.cluster_id)}"
                  data-fracture="${esc(r.fracture_id)}"
                  data-branch="${esc(b.branch_id)}"
                  data-testid="apply-check-r${r.rank}-b${bIdx}" ${S.applyingCheck ? "disabled" : ""}>
                  <div class="bb-top">
                    <span class="bb-title">${esc(friendlyBranchLabel(b.label, b.condition_summary, b.branch_id))}</span>
                    <span class="bb-out is-${outSm.cls}">${ICONS.arrowRight}<span>${esc(outTxt)}</span></span>
                  </div>
                  <div class="bb-sub">${esc(friendlyConditionSummary(b.condition_summary, b.branch_id))}</div>
                </button>`;
            }).join("")}
          </div>` : "";

        return `
          <div class="rank-card ${isPrimary ? "is-primary" : ""}" data-action="ident-select-fracture" data-fracture="${esc(r.fracture_id)}" data-testid="rank-card-${r.rank}">
            <div class="rk-head">
              <span class="rk-idx mono">#${r.rank}</span>
              <strong class="mono">${esc(r.short_label || r.fracture_id)}</strong>
              <span class="rk-role">${esc(friendlyRole(r.role))}</span>
            </div>
            <div class="rk-metrics">
              <span>测度消除 <strong class="mono">${resPct}%</strong></span>
              <span>·</span>
              <span>确定化比 <strong class="mono">${certPct}%</strong></span>
            </div>
            <p class="rk-why">${esc(friendlyAttribution(r.geometric_attribution || r.witness_diff_role))}</p>
            ${branchesHTML}
          </div>
        `;
      };

      const top1 = ranked[0];
      const others = ranked.slice(1, 4);
      decisionHTML = `
        <div class="decision-block" data-testid="id-active-check-panel">
          <div class="block-head">
            <span>推荐下一步现场复核</span>
            <span class="block-hint">点击分支即时验证坍缩</span>
          </div>
          ${top1 ? renderRankCard(top1, true) : ""}
          ${others.length ? `
            <button type="button" class="more-toggle" data-action="toggle-more-candidates">
              <span>${S.showMoreCandidates ? "收起次优候选裂隙" : `展开其余 ${others.length} 条次优候选裂隙`}</span>
              ${ICONS.chevronDown}
            </button>
            ${S.showMoreCandidates ? `<div class="more-list">${others.map((r) => renderRankCard(r, false)).join("")}</div>` : ""}
          ` : ""}
        </div>
      `;
    }

    // 3) 简洁快速跳转（基线误报对照 & AI 助手）
    const quickFooterHTML = `
      <div class="rail-quick-links">
        <button type="button" class="ql-btn" data-action="switch-rail-tab" data-tab="details">
          <span>查看传统方法误报对照与几何参数</span>${ICONS.arrowRight}
        </button>
        <button type="button" class="ql-btn" data-action="open-copilot-tab">
          <span>${ICONS.copilot} 需要大白话讲解或自动操作？切换到 AI 讲解</span>${ICONS.arrowRight}
        </button>
      </div>
    `;

    return `${insightHTML}${decisionHTML}${quickFooterHTML}`;
  }

  function buildDetailsTabHTML() {
    const cur = S.currentCase;
    if (!cur) return `<div class="empty-state">尚未载入案例</div>`;
    const bc = cur.baseline_comparison || {};
    const nu = bc.naive_union || {};
    const nom = bc.nominal_single_point || {};
    const nuDanger = nu.failure_type === "FALSE_UNION_PATH" || nu.failure_type === "MASKS_DISCONNECTED_WITNESS";
    const nomDanger = nom.failure_type === "SILENT_FALSE_SAFE_FLIP" || nom.failure_type === "SILENT_MISSED_CONNECTION_FLIP";
    const selMeta = (cur.fractures || []).find((f) => f.id === S.selectedFractureId) || (cur.fractures || [])[0];

    return `
      <div class="decision-block" data-testid="id-baseline-compare">
        <div class="block-head"><span>传统基线方法误报对照</span></div>
        <div class="bl-item">
          <div class="bl-head">
            <strong>朴素并集图</strong>
            <span class="bl-st ${nuDanger ? "is-danger" : "is-ok"}">${nu.connected ? "报连通" : "报不贯通"} · ${esc(friendlyBaselineFailure(nu.failure_type))}</span>
          </div>
          <p class="body-p">${esc(nu.explanation || "—")}</p>
        </div>
        <div class="bl-item">
          <div class="bl-head">
            <strong>名义单一模型 (${esc(nom.mode || "m0")}, t=${esc(nom.t || "0/1")})</strong>
            <span class="bl-st ${nomDanger ? "is-warn" : "is-ok"}">${nom.connected === null ? "未判定" : nom.connected ? "名义判通" : "名义判断"} · ${esc(friendlyBaselineFailure(nom.failure_type))}</span>
          </div>
          <p class="body-p">${esc(nom.explanation || "—")}</p>
        </div>
      </div>

      ${selMeta ? `
        <div class="decision-block">
          <div class="block-head"><span>选中裂隙三维几何</span><span class="dim">${esc(friendlyRole(selMeta.role))}</span></div>
          <dl class="kv-clean">
            <dt>裂隙标识</dt><dd><strong class="mono">${esc(selMeta.short_label)}</strong> ${hashlineHTML(selMeta.id, "id-selected-fracture")}</dd>
            <dt>产状</dt><dd class="mono">倾角 ${Number(selMeta.dip_deg || 0).toFixed(2)}° / 倾向 ${Number(selMeta.dip_direction_deg || 0).toFixed(2)}°</dd>
            <dt>面积 / 半径</dt><dd class="mono">${Number(selMeta.area_m2 || 0).toFixed(3)} m² · R=${Number(selMeta.char_radius_m || 0).toFixed(3)} m</dd>
            <dt>拟合残差 σ</dt><dd class="mono">${Number(selMeta.residual_sigma_m || 0).toFixed(4)} m</dd>
            <dt>孔洞保留</dt><dd>${selMeta.preserved_hole_count > 0 ? `保留 ${selMeta.preserved_hole_count} 处真实孔洞（禁凸包填洞）` : "0 处内部孔洞"}</dd>
          </dl>
        </div>
      ` : ""}

      <div class="rail-quick-links">
        <button type="button" class="ql-btn" data-action="open-provenance">
          <span>${ICONS.shieldOk} 打开完整 SHA256 证据链与学术边界抽屉</span>${ICONS.arrowRight}
        </button>
      </div>
    `;
  }

  function renderAcademicFooter() {
    const el = $("#cmdbar");
    if (!el) return;
    const cur = S.currentCase;
    const scope = (cur && cur.scope) || "REAL_DERIVED_CONDITIONED_FINITE_UNION";
    const m3 = (cur && cur.m3_judge_verdict) || "AGREE";
    el.innerHTML = `
      <div class="academic-bar" data-testid="id-academic-boundary-bar">
        <span class="ab-desktop-only">域：<code class="mono">${esc(scope)}</code></span>
        <span class="ab-dot ab-desktop-only">·</span>
        <span class="ab-desktop-only">学术边界（未实测严禁填 0）：<code>field_error_rate=null</code> · <code>hydraulic=null</code> · <code>human_time=null</code></span>
        <span class="ab-mobile-text">真包只读 · 3 项未实测为 <code>null</code>（不填 0）</span>
        <span class="spacer"></span>
        <button type="button" class="ab-link" data-action="open-provenance">
          ${ICONS.shieldOk}<span class="lbl-full">M3=${esc(m3)} (22/22) · SHA256 核验</span><span class="lbl-short">SHA256 核验</span>
        </button>
      </div>
    `;
  }

  function renderProvenanceDrawer() {
    const drawer = $("#provenance-drawer");
    const backdrop = $("#provenance-backdrop");
    if (!drawer || !backdrop) return;
    if (!S.provenanceOpen) {
      drawer.hidden = true;
      backdrop.hidden = true;
      return;
    }
    drawer.hidden = false;
    backdrop.hidden = false;

    const cur = S.currentCase;
    const meta = S.casesMeta || {};
    const m3Detail = (cur && cur.m3_judge_detail) || {};
    const m3Checks = m3Detail.checks || {};
    const prov = (cur && cur.real_provenance) || {};
    const areg = (cur && cur.assumption_register) || {};
    const srcFiles = prov.source_files || [];

    drawer.innerHTML = `
      <div class="drawer-head">
        <div>
          <div class="eyebrow">可信溯源与防篡改核验</div>
          <h2 class="drawer-title">SHA256 证据链、M3 独立判卷与三项 null 边界</h2>
        </div>
        <button type="button" class="tb-icon-btn" data-action="close-provenance" data-testid="close-provenance" title="关闭">${ICONS.close}</button>
      </div>
      <div class="drawer-scroll">
        <div class="drawer-sec">
          <div class="drawer-sec-head"><strong>1. M3 独立判卷总账</strong><span class="agree-chip mono">22 / 22 AGREE · 0 DISAGREE</span></div>
          <p class="body-p">独立判卷器 <code>judge_real.py</code> 零复用求解器内部状态，独立重算 22 簇精确区间分割、稳定裕度 ρ 及深度 1/2 坍缩树，全部一致通过。</p>
          <dl class="kv-clean">
            <dt>M1 预注册表</dt><dd>${hashlineHTML(meta.m1_register_sha256 || m3Detail.m1_register_sha256, "prov-m1-reg-sha")}</dd>
            <dt>M2 运行汇总</dt><dd>${hashlineHTML(meta.m2_summary_sha256 || m3Detail.m2_summary_sha256, "prov-m2-sum-sha")}</dd>
            <dt>M3 批量回执</dt><dd>${hashlineHTML(meta.m3_verdict_sha256 || (cur && cur.m3_verdict_sha256), "prov-m3-batch-sha")}</dd>
            <dt>M3 判卷代码</dt><dd>${hashlineHTML(meta.m3_judge_real_sha256 || m3Detail.judge_real_sha256, "prov-m3-judge-sha")}</dd>
          </dl>
        </div>

        ${cur ? `
          <div class="drawer-sec">
            <div class="drawer-sec-head"><strong>2. 当前簇 (${esc(cur.cluster_id)}) 证书对账</strong><span class="agree-chip mono">M3: ${esc(cur.m3_judge_verdict)}</span></div>
            <dl class="kv-clean">
              <dt>model canonical</dt><dd>${hashlineHTML(cur.input_sha256, "prov-cur-input-sha")}</dd>
              <dt>model 磁盘文件</dt><dd>${hashlineHTML(cur.model_file_sha256, "prov-cur-file-sha")}</dd>
              <dt>M2 certification</dt><dd>${hashlineHTML(cur.m2_cert_sha256, "prov-cur-cert-sha")}</dd>
              <dt>M2 active_check</dt><dd>${hashlineHTML(cur.m2_active_check_sha256, "prov-cur-ac-sha")}</dd>
              <dt>单元重算对账</dt><dd class="mono">声明 ${esc(m3Checks.cells_declared ?? cur.cell_count)} / 重算 ${esc(m3Checks.cells_recomputed ?? cur.cell_count)}</dd>
              <dt>精确有理 ρ</dt><dd class="mono wrap-any">${esc(m3Checks.rho_recomputed || (cur.rho && cur.rho.infimum) || "— (确定态)")}</dd>
              <dt>原始坐标系</dt><dd class="mono wrap-any">${esc(prov.coordinate_frame || "—")}</dd>
              ${srcFiles[0] ? `<dt>原始数据源</dt><dd>${hashlineHTML(srcFiles[0].sha256, "prov-cur-source-sha")}</dd>` : ""}
            </dl>
          </div>
        ` : ""}

        <div class="drawer-sec">
          <div class="drawer-sec-head"><strong>3. 三项法定 null 指标（严禁填 0）</strong><span class="mono dim">3 × null</span></div>
          <table class="boundary-table">
            <thead><tr><th>指标字段</th><th>法定值</th><th>未实测边界说明</th></tr></thead>
            <tbody>
              <tr><td><code>field_error_rate</code><br><span class="dim">地下无条件真值</span></td><td><code class="null-tag">null</code></td><td>露头仅约束地表条件几何域，不等同深部未开挖真值。</td></tr>
              <tr><td><code>hydraulic</code><br><span class="dim">水力连通/涌水量</span></td><td><code class="null-tag">null</code></td><td>首期锁定几何贯通可辨识性，非隙宽渗透率结论。</td></tr>
              <tr><td><code>human_time</code><br><span class="dim">人工实测耗时</span></td><td><code class="null-tag">null</code></td><td>未开展野外专家人工计时对照实验，严格留空。</td></tr>
            </tbody>
          </table>
        </div>

        <div class="drawer-sec">
          <div class="drawer-sec-head"><strong>4. 中文界面与底层学术机器码对照</strong></div>
          <dl class="kv-clean">
            <dt>真实露头条件域</dt><dd class="mono">REAL_DERIVED_CONDITIONED_FINITE_UNION</dd>
            <dt>存在双见证歧义</dt><dd class="mono">AMBIGUOUS_WITH_VERIFIED_WITNESSES</dd>
            <dt>确定贯通 / 断开</dt><dd class="mono">CERTIFIED_CONNECTED / CERTIFIED_DISCONNECTED</dd>
            <dt>超预算拒答</dt><dd class="mono">BUDGET_EXCEEDED</dd>
            <dt>非观测假设边</dt><dd class="mono">${esc(areg.assumed_edges_tag || "ASSUMED_NOT_OBSERVED")}</dd>
          </dl>
        </div>
      </div>
    `;
  }

  function renderMobileSheetHandle() {
    const el = $("#mob-sheet-handle");
    if (!el) return;
    const tabTitle = S.railTab === "clusters"
      ? "22 簇案例"
      : S.railTab === "details"
      ? "基线与几何"
      : S.railTab === "copilot"
      ? "AI 讲解"
      : "探索与复核";
    const curId = S.currentCase ? S.currentCase.cluster_id : S.selectedClusterId;
    const isCollapsed = S.mobileSheetState === "collapsed";
    const isExpanded = S.mobileSheetState === "expanded";
    el.innerHTML = `
      <div class="msh-pill" aria-hidden="true"></div>
      <div class="msh-row">
        <button type="button" class="msh-title" data-action="mobile-sheet-cycle" title="点击切换面板高度（也可上下拖拽）">
          <strong>${esc(tabTitle)}</strong>
          <span class="mono dim">${esc(curId)}</span>
        </button>
        <div class="msh-actions">
          <button type="button" class="msh-btn ${isCollapsed ? "is-active" : ""}"
            data-action="mobile-sheet-set" data-sheet="${isCollapsed ? "half" : "collapsed"}"
            data-testid="mobile-sheet-3d-toggle">
            ${isCollapsed ? `${ICONS.chevronUp}<span>恢复面板</span>` : `${ICONS.chevronDown}<span>沉浸 3D</span>`}
          </button>
          <button type="button" class="msh-btn ${isExpanded ? "is-active" : ""}"
            data-action="mobile-sheet-set" data-sheet="${isExpanded ? "half" : "expanded"}"
            data-testid="mobile-sheet-expand-toggle">
            ${isExpanded ? `${ICONS.chevronDown}<span>还原</span>` : `${ICONS.chevronUp}<span>展开</span>`}
          </button>
        </div>
      </div>
    `;
  }

  function renderMobileSheetNav() {
    const el = $("#mobile-sheet-nav");
    if (!el) return;
    el.innerHTML = `
      <button type="button" class="mob-tab ${S.railTab === "explore" ? "is-active" : ""}"
        data-action="switch-rail-tab" data-tab="explore" data-testid="mobile-tab-inspect">
        ${ICONS.sliders}<span>探索复核</span>
      </button>
      <button type="button" class="mob-tab ${S.railTab === "clusters" ? "is-active" : ""}"
        data-action="switch-rail-tab" data-tab="clusters" data-testid="mobile-tab-clusters">
        ${ICONS.list}<span>22 簇</span>
      </button>
      <button type="button" class="mob-tab ${S.railTab === "details" ? "is-active" : ""}"
        data-action="switch-rail-tab" data-tab="details" data-testid="mobile-tab-details">
        ${ICONS.layers}<span>基线几何</span>
      </button>
      <button type="button" class="mob-tab ${S.railTab === "copilot" ? "is-active" : ""}"
        data-action="open-copilot-tab" data-testid="mobile-tab-copilot">
        ${ICONS.copilot}<span>AI 讲解</span>
      </button>
    `;
  }

  function renderAll(update3D = false) {
    renderTopbar();
    renderHeroHud();
    renderTourRibbon();
    renderViewportChrome();
    renderCopilotWidget();
    renderRightRail();
    renderAcademicFooter();
    renderProvenanceDrawer();
    renderMobileSheetHandle();
    renderMobileSheetNav();
    if (update3D) syncViewerScene(false);
  }

  /* ---------------- 导览控制 ---------------- */
  async function gotoTourStep(idx) {
    S.tourActive = true;
    S.tourStepIndex = clamp(idx, 0, GUIDED_TOUR_STEPS.length - 1);
    const step = GUIDED_TOUR_STEPS[S.tourStepIndex];
    S.siteFilter = step.siteFilter || "all";
    S.statusFilter = "all";
    S.railTab = "explore";

    if (step.step === 1) await selectCluster("jardim_c04", { enableSplit: false, witnessView: "nominal" });
    else if (step.step === 2) await selectCluster("jardim_c02", { enableSplit: true, witnessView: "disconnected" });
    else if (step.step === 3) await selectCluster("perozzo_c02", { enableSplit: false, witnessView: "connected" });
    renderAll(false);
  }

  async function runTourStepDemo() {
    const step = GUIDED_TOUR_STEPS[S.tourStepIndex];
    if (!step) return;
    if (step.step === 1) {
      await executeCopilotAction({ type: "demo_1step" });
      S.railTab = "explore";
      renderAll(false);
    } else if (step.step === 2) {
      await executeCopilotAction({ type: "demo_2step" });
      S.railTab = "explore";
      renderAll(false);
    } else if (step.step === 3) {
      S.witnessView = "disconnected";
      S.interpActive = false;
      syncViewerScene(true);
      if (viewer) viewer.focusGap(null, true);
      renderViewportChrome();
      toast("已推进微距镜头聚焦至 perozzo_c02 最小割断口 (3.97 mm)", "info");
    }
  }

  /* ---------------- 事件监听 ---------------- */
  function bindEvents() {
    document.addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      const action = btn.dataset.action;

      if (action === "switch-rail-tab") {
        S.railTab = btn.dataset.tab || "explore";
        S.copilotOpen = S.railTab === "copilot";
        if (isMobileViewport()) {
          if (S.railTab === "clusters" && S.mobileSheetState !== "expanded") {
            setMobileSheetState("expanded");
          } else if (S.mobileSheetState === "collapsed") {
            setMobileSheetState("half");
          } else {
            renderMobileSheetHandle();
          }
        }
        renderTopbar();
        renderCopilotWidget();
        renderRightRail();
        renderMobileSheetNav();
      } else if (action === "open-copilot-tab" || action === "toggle-copilot") {
        if (action === "toggle-copilot" && S.railTab === "copilot") {
          S.railTab = "explore";
          S.copilotOpen = false;
          renderTopbar();
          renderCopilotWidget();
          renderRightRail();
          renderMobileSheetHandle();
          renderMobileSheetNav();
        } else {
          openCopilotTab(true);
        }
      } else if (action === "mobile-sheet-set") {
        setMobileSheetState(btn.dataset.sheet || "half");
      } else if (action === "mobile-sheet-cycle") {
        const next = S.mobileSheetState === "half" ? "expanded" : S.mobileSheetState === "expanded" ? "collapsed" : "half";
        setMobileSheetState(next);
      } else if (action === "toggle-mobile-interp") {
        S.mobileInterpOpen = !S.mobileInterpOpen;
        renderViewportChrome();
        if (viewer) window.setTimeout(() => viewer && viewer.resize(), 40);
      } else if (action === "toggle-more-candidates") {
        S.showMoreCandidates = !S.showMoreCandidates;
        renderRightRail();
      } else if (action === "copilot-preset") {
        await runCopilotPreset(btn.dataset.preset || "q_data_60m");
      } else if (action === "ident-select-cluster") {
        const cid = btn.dataset.cluster;
        if (cid) {
          if (isMobileViewport()) setMobileSheetState("half");
          await selectCluster(cid, { preserveSplit: false, switchToExplore: true });
        }
      } else if (action === "ident-site") {
        S.siteFilter = btn.dataset.site || "all";
        renderRightRail();
      } else if (action === "status-filter") {
        S.statusFilter = btn.dataset.statusFilter || "all";
        renderRightRail();
      } else if (action === "ident-witness") {
        const w = btn.dataset.witness || "nominal";
        S.witnessView = w;
        S.interpActive = false;
        S.interpAlpha = w === "connected" ? 0 : w === "disconnected" ? 100 : 50;
        if (S.splitMode) S.splitMode = false;
        syncViewerScene(true);
        renderViewportChrome();
      } else if (action === "toggle-split-view") {
        S.splitMode = !S.splitMode;
        S.interpActive = false;
        if (isMobileViewport() && S.splitMode) {
          setMobileSheetState("collapsed");
        }
        syncViewerScene(true);
        renderViewportChrome();
        toast(S.splitMode ? "已开启左右双见证 3D 同步分屏" : "已返回单视口模式", "info");
      } else if (action === "toggle-amp") {
        S.ampFactor = S.ampFactor > 1 ? 1 : 8;
        syncViewerScene(true);
        renderViewportChrome();
        toast(S.ampFactor > 1 ? "已开启 8× 微移视觉放大" : "已恢复 1× 真实物理尺度", "info");
      } else if (action === "interp-snap") {
        S.interpAlpha = Number(btn.dataset.alpha || 50);
        S.interpActive = true;
        S.splitMode = false;
        syncViewerScene(true);
        renderViewportChrome();
      } else if (action === "focus-gap") {
        const eff = getEffectiveCaseState(S.currentCase);
        if (!eff || !eff.disconnected_witness || !eff.disconnected_witness.cut_gap) {
          toast("当前状态无 S/T 最小割断口", "warn");
          return;
        }
        if (!S.splitMode && S.witnessView !== "disconnected") {
          S.witnessView = "disconnected";
          S.interpActive = false;
          syncViewerScene(true);
        }
        const zoomedIn = viewer ? viewer.focusGap(eff.disconnected_witness.cut_gap, true) : false;
        renderViewportChrome();
        const distM = Number(eff.disconnected_witness.cut_gap.distance_m || 0);
        toast(zoomedIn ? `已聚焦 S/T 断口：${(distM * 1000).toFixed(2)} mm` : "已复位至全景视角", "info");
      } else if (action === "fit-view") {
        if (viewer) viewer.fit(true);
        renderViewportChrome();
      } else if (action === "top-view") {
        if (viewer) viewer.topView(true);
        renderViewportChrome();
      } else if (action === "front-view") {
        if (viewer) viewer.frontView(true);
        renderViewportChrome();
      } else if (action === "toggle-autorotate") {
        S.autoRotate = !S.autoRotate;
        if (viewer) viewer.setAutoRotate(S.autoRotate);
        renderViewportChrome();
      } else if (action === "shot-view") {
        if (!viewer) return;
        const a = document.createElement("a");
        a.href = viewer.screenshot();
        a.download = `STRATA_${(S.currentCase && S.currentCase.cluster_id) || "3d"}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        toast("高清 PNG 三维图版已导出", "ok");
      } else if (action === "ident-apply-check") {
        e.stopPropagation();
        await applyActiveCheck({
          cluster_id: btn.dataset.cluster,
          fracture_id: btn.dataset.fracture,
          branch_id: btn.dataset.branch,
          step2_fracture_id: btn.dataset.step2Fracture || "",
          step2_branch_id: btn.dataset.step2Branch || "",
        });
      } else if (action === "ident-reset-collapse") {
        resetCollapse();
      } else if (action === "ident-select-fracture" || action === "vp-pick-fracture") {
        const fid = btn.dataset.fracture;
        if (fid) {
          S.selectedFractureId = fid;
          if (viewer) viewer.setSelectedFracture(fid);
          renderRightRail();
        }
      } else if (action === "start-tour") {
        if (S.tourActive) {
          S.tourActive = false;
          renderTourRibbon();
          renderTopbar();
        } else {
          await gotoTourStep(0);
        }
      } else if (action === "tour-goto") {
        await gotoTourStep(Number(btn.dataset.step || 0));
      } else if (action === "tour-prev") {
        if (S.tourStepIndex > 0) await gotoTourStep(S.tourStepIndex - 1);
      } else if (action === "tour-next") {
        if (S.tourStepIndex < GUIDED_TOUR_STEPS.length - 1) await gotoTourStep(S.tourStepIndex + 1);
        else {
          S.tourActive = false;
          renderTourRibbon();
          renderTopbar();
          toast("1 分钟导览已完成，可自由浏览全部 22 簇", "ok");
        }
      } else if (action === "tour-exit") {
        S.tourActive = false;
        renderTourRibbon();
        renderTopbar();
      } else if (action === "tour-run-demo") {
        await runTourStepDemo();
      } else if (action === "open-provenance") {
        S.provenanceOpen = true;
        renderProvenanceDrawer();
      } else if (action === "close-provenance") {
        S.provenanceOpen = false;
        renderProvenanceDrawer();
      } else if (action === "toggle-theme") {
        S.theme = S.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = S.theme;
        if (viewer) viewer.setTheme(S.theme);
        renderTopbar();
      } else if (action === "copy-text") {
        e.stopPropagation();
        const text = btn.dataset.copy || "";
        if (text && navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(() => toast("SHA256 已复制", "ok"));
        }
      }
    });

    document.addEventListener("submit", async (e) => {
      if (e.target && e.target.id === "ai-copilot-form") {
        e.preventDefault();
        const inp = $("#ai-copilot-input");
        if (!inp) return;
        const val = inp.value;
        inp.value = "";
        await handleCopilotTextQuery(val);
      }
    });

    const backdrop = $("#provenance-backdrop");
    if (backdrop) {
      backdrop.addEventListener("click", () => {
        S.provenanceOpen = false;
        renderProvenanceDrawer();
      });
    }

    const copScrim = $("#copilot-scrim");
    if (copScrim) {
      copScrim.addEventListener("click", () => {
        S.copilotOpen = false;
        renderCopilotWidget();
        renderMobileSheetNav();
      });
    }

    // 移动端抽屉手柄上下拖拽手势（上滑展开 / 下滑收起进入沉浸 3D）
    const sheetHandle = $("#mob-sheet-handle");
    if (sheetHandle) {
      let startY = null;
      let dragging = false;
      sheetHandle.addEventListener("pointerdown", (e) => {
        if (!isMobileViewport() || e.target.closest("button[data-action='mobile-sheet-set']")) return;
        startY = e.clientY;
        dragging = true;
      });
      window.addEventListener("pointerup", (e) => {
        if (!dragging || startY === null) return;
        const dy = e.clientY - startY;
        dragging = false;
        startY = null;
        if (Math.abs(dy) < 28) return;
        if (dy < -28) {
          setMobileSheetState(S.mobileSheetState === "collapsed" ? "half" : "expanded");
        } else if (dy > 28) {
          setMobileSheetState(S.mobileSheetState === "expanded" ? "half" : "collapsed");
        }
      });
      window.addEventListener("pointercancel", () => {
        dragging = false;
        startY = null;
      });
    }

    window.addEventListener("resize", () => {
      const scrim = $("#copilot-scrim");
      if (scrim) scrim.hidden = !(S.copilotOpen && isMobileViewport());
    });

    document.addEventListener("input", (e) => {
      const t = e.target;
      if (t && t.id === "ident-search") {
        S.searchQuery = t.value || "";
        renderRightRail();
        const inp = $("#ident-search");
        if (inp) {
          inp.focus();
          inp.setSelectionRange(S.searchQuery.length, S.searchQuery.length);
        }
      } else if (t && t.id === "witness-interp-slider") {
        S.interpAlpha = Number(t.value || 50);
        S.interpActive = true;
        if (S.splitMode) S.splitMode = false;
        syncViewerScene(true);
        const eff = getEffectiveCaseState(S.currentCase);
        const ro = $('[data-testid="interp-readout"]');
        if (ro && eff && eff.connected_witness && eff.disconnected_witness) {
          const tC = Number(eff.connected_witness.t_float || 0);
          const tD = Number(eff.disconnected_witness.t_float || 0);
          const tInterp = (tC + (tD - tC) * (S.interpAlpha / 100)).toFixed(4);
          const zoneText = S.interpAlpha <= 35 ? "连通邻域" : S.interpAlpha >= 65 ? "断割邻域" : "临界翻转带";
          ro.textContent = `t=${tInterp} · ${zoneText}`;
        }
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (S.provenanceOpen) {
          S.provenanceOpen = false;
          renderProvenanceDrawer();
        } else if (S.copilotOpen) {
          S.copilotOpen = false;
          renderCopilotWidget();
          renderMobileSheetNav();
        } else if (S.tourActive) {
          S.tourActive = false;
          renderTourRibbon();
          renderTopbar();
        }
      }
    });
  }

  function init() {
    viewer = window.StrataViewer3D.create({
      primaryCanvas: $("#plate"),
      secondaryCanvas: $("#plate-right"),
      primaryOverlay: $("#vp-overlay-primary"),
      secondaryOverlay: $("#vp-overlay-secondary"),
      compassEl: $("#vp-compass"),
      scaleEl: $("#vp-scale"),
      engineTagEl: $("#vp-engine-tag"),
      onSelectFracture(fid) {
        S.selectedFractureId = fid;
        viewer.setSelectedFracture(fid);
        renderRightRail();
      },
    });
    bindEvents();
    loadCasesList();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
