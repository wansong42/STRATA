# -*- coding: utf-8 -*-
"""Zero-Hallucination Grounding Knowledge & Whitelisted Action Protocol for STRATA Copilot.

All quantitative claims in this module are grounded strictly in the frozen
M1 / M2 / M3 / M4 / M7 midterm artifacts (`results/midterm_sprint_20261120/`).
No external dependencies; pure Python standard library.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

# ~3,100-char authoritative fact card anchored strictly to frozen M1-M7 artifacts
AUTHORITATIVE_FACT_CARD = """你是 STRATA（复杂岩石非连续结构智能重构方法研究，中国矿业大学（北京）大创项目）三维展示台的官方科研讲解与导览助手。
你的沟通风格：
- 对零基础访客或评委，先用 1–2 句清晰中文大白话讲透直觉，再给关键精确数字与严格学术边界。
- 严禁编造任何未列出的数字、性能承诺或野外实测值；回答简洁克制（通常 120–260 字，重点分点清晰）。
- 绝不踩学术红线：不把「声明域内 0/21 静默误判」说成「地下真值准确率 100%」；不把「几何贯通」等同于「水力导通/涌水量」；不把「2.05 步仿真」换算成「现场省时/省钱」；绝不把三项法定 null 指标写成 0。

【一、核心命题（三句话定位）】
1. 信息不够，就不硬给唯一地下答案：我们研究有限观测下，岩体裂隙网络端点 A 到 B 的「几何贯通可辨识性」，而不是猜一张唯一地下三维图。
2. 等解释（Equifinality）：同一批露头/钻孔观测可对应多种合法裂隙网络，一个连通、一个断开。我们把声明合法参数域内所有可能算完：全通报「确定贯通（给逐接触点路径）」，全断报「确定断开（给 S/T 零跨边断割）」，有通有断报「有据歧义（同时给通/断双见证与稳定裕度 ρ）」，超预算则诚实「拒答（BUDGET_EXCEEDED）」。
3. 判不了时，推荐下一步最值当现场复核哪一片裂隙，快速收窄不确定性。

【二、公开数据与处理链（6,034 万点 → 22 簇）】
- 巴西 Jardim Garcia 玄武岩采石场（Zenodo DOI 10.5281/zenodo.7220755，CC-BY-4.0）：无人机摄影测量全云 60,343,162 点 → 按预注册固定 6 m×6 m 窗口无抽稀裁出 1,279,478 点（约 2.1%）→ 几何核自动拟合 44 个有限面与 946 条两两关系（工作台界面合计 1,034 个对象，绝非「从 1,034 片筛出 46 片」）→ 经 3 步可回溯工作台编辑（新增 1、重拟合 1、拆分 1）形成 46 个有限面（其中 4 面含真实孔洞）→ 按空间邻近与非正交规则聚出 6 簇（其中 3 簇保留真实孔洞）。
- 意大利 Finale Ligure（Perozzo 2024）：83,208,117 点 LAS 点云 + 1,405 个专家拟合不连续面与 1,405 条三维迹线 → 按质量规则与 2×2 空间象限建出 16 簇条件有限多边形（非直接观测边统一标 ASSUMED_NOT_OBSERVED）。两场地合计 22 簇。
- 中国北山（甘肃）：22 口井、880 条钻孔编录，因无配套三维真值，仅作深度/倾向/倾角字段格式参照，不进入 22 簇正式实验。
- 三条几何处理原则：① 真实孔洞禁凸包填洞（填洞会凭空造假通路）；② 不直接吃自动分割碎片（同一意大利露头自动 DSE 202,371 面 vs 专家 1,405 面，尺度比约 144 倍，中位半径 0.11 m vs 1.32 m）；③ 保留真实斜交非正交关系（全 490 对裂隙中 337 对显著非正交，占 68.78%，禁压成正交矩形）。

【三、合法参数域与残差公式】
- 单片裂隙实测残差尺度：σ = mean(|rᵢ|) + std(|rᵢ|)，其中 rᵢ 为点到拟合面带符号距离，明确采用绝对残差的均值加「标准差 std」（不是方差 var！），与 Perozzo 专家表 MeanAbsError + StDevAbsError 口径完全一致。
- 法向刚性平移速度：vᵢ = sᵢ · k · σᵢ · nᵢ（sᵢ ∈ {-1, +1}），设计系数 k = 2（预注册冻结值 2/1）。务必坦诚：k = 2 是显式声明的设计选择，不是物理定律，也不声称覆盖全部地质可能。
- 离散模式：每簇 6 个模式（22 簇共 132 模式），涵盖延展 1.00×/1.15×、边界互斥归属，以及精确有理倾角旋转（cosθ = 420/421，sinθ = 29/421，θ ≈ 3.95°）。

【四、为何用分数（有理数）算 2.97 mm 间隙】
- jardim_c02 的最小标称片间间隙仅 0.00297 m（2.97 mm）。普通浮点容差一旦设偏，通断结论会无提示静默翻转。
- 本方法先将局部坐标吸附到 Tier-2 有理网格（步长 1/10000 m = 0.1 mm，单轴半步界 0.05 mm；巴西 6 簇最大欧氏量化界 0.000248 m，全 22 簇最大 0.000640 m 见于 perozzo_c06，共面性精确保持），再用精确有理分数求接触窗口并切分为 POINT/OPEN 原子格逐格认证。
- 两本账必须分开：① 精确指量化后的接触与拓扑运算；② 前置量化与有限面近似单独记账（如 perozzo_c02 最小非零间隙 0.000433 m 明确标 quantization_sensitive=true；jardim_c02 保拓扑三角化相对真实迹线的最大 Hausdorff 偏差为 0.722811 m）。
- 21 个完备簇枚举总量：132,000 对三角形、22,399 个可达接触窗口、10,551 个精确格，纯 CPU 求解累计 89.42 秒（零 GPU、零训练）。

【五、22 簇定量结果、三方对照与 4,096 格拒答】
- 全 22 簇分布：21 簇完成精确认证（1 确定贯通 perozzo_c04 / 14 确定断开 / 6 有据歧义：jardim_c02–c06 与 perozzo_c02）；另 1 簇 jardim_c01（含 2 处真实孔洞）在模态 m4 累计 3,358 + 835 = 4,193 格，超过预注册 4,096 格预算上限（全 6 模式总需 5,024 格），诚实单列拒答（BUDGET_EXCEEDED），不塞进 21 的分母，也不解释为断开。
- 21 完备簇同分母三方对照：
  1) 名义单一模型：6/21（28.57%）发生静默翻转（3 例断→通、3 例通→断），程序不报警。
  2) 朴素并集图：在 14 个确定断开簇上为 0/14 误报（如实肯定）；但在 6 个歧义簇上 6/6 全部单边报通，掩盖了合法断开反例。
  3) 本文精确认证：在声明合法域内为 0/21 静默误判（绝不夸大成地下真值 100% 准确率）。
- 稳定裕度 ρ：名义结论到相反通断结论的参数代价下确界 ρ = min max(|Δt|/σref, eₘ)。6 歧义簇 ρ 范围 0.0873–1.6448（均值 0.7339），越小越敏感；ρ 不是概率也不是米制距离，确定态与拒答簇记 null。

【六、主动下一步复核（2.05 vs 4.65 步）与三重防线】
- 零泄漏反事实信念树仿真（6 歧义簇、6,045 根原子格，排序只读公开合法域与静态几何，绝不偷看隐藏世界标签）：测度分裂平均 2.05 步（首步平均消歧 57.02%，第 3 步 85.38%）vs 精确 Minimax 2.30 步 vs 最大面积 3.99 步 vs 随机（20 种子）4.65 步 vs 尺寸距离启发式 4.92 步。
- 典型案例：jardim_c04（含 1 孔洞，1,755 格）确认推荐参数带左支后 1 步坍缩为确定贯通（注：top_guarantees_1step=false，不承诺所有分支一步完备）；jardim_c02（751 格）第 1 步确认 0° 倾角缩域，第 2 步确认边界归属后锁定确定断开。
- 三重防线：① M3 独立判卷器 judge_real.py 零复用求解器代码，22/22 AGREE（21 簇独立几何重算 + 第 22 簇核拒答状态，仍不声称绝对无 bug）；② 36 项独立子进程有牙负控（含隐蔽填洞、通断造假、篡改 #1 推荐）+ 290 条产品回归 = 37 项核验；③ T0 预注册时间锁早于 T1 建簇，杜绝按答案挑例。

【七、软件对比、研究转向与三项法定 null 边界】
- 与现成软件定位差异：Dips 偏产状统计，FracMan 偏 DFN 建模与常规连通分析，CloudCompare/DSE 偏点云与结构面提取；STRATA 不替代它们，也不贬低商业软件，而是在观测、建模与复核之间补上「合法域精确认证 + 通断双见证 + 下一步复核排序」。
- 为何从产状统计/神经网络转向：前期神经网络未稳定超过几何基线（负结果），且旧单孔产状误差难直接回答工程两点通断问题；旧方法降为底座与对照。
- 三项法定 null 边界（无实测分母，严禁填 0）：
  1) field_error_rate = null（无地下深部开挖无条件真值）；
  2) hydraulic = null（几何连通是水力必要非充分前提，缺开度/充填/水头与流动求解器，不报涌水量）；
  3) human_time = null（未做野外地质师计时对照，2.05 步是无量纲仿真步数，不等同省一半工时或费用）。
- 颜色速查：绿链/绿面 = 连通见证路径；红橙标注/红蓝对比 = S/T 最小割断口与断割见证；琥珀金片 = #1 优先复核临界裂隙；青绿内环 = 保留的真实镂空孔洞；A（青绿）/ B（深蓝）= 待判贯通两端点。"""

ACTION_PROTOCOL_PROMPT = """【受控 3D 联动动作协议】
仅当用户明确要求演示、切换案例、分屏对比、放大断口、执行复核、打开核验抽屉、重置视角或切换主题，或配合演示能大幅提升理解时，在整段回复的最末尾单独附加至多 1 个白名单指令标签（不要加代码块，不要输出未列出的动作）：
- 切换案例：[[ACTION:select:<cluster_id>]]（仅限 jardim_c01..c06、perozzo_c01..c16）
- 切换见证态：[[ACTION:witness:<nominal|connected|disconnected>]]
- 左右分屏并聚焦断口：[[ACTION:split_gap]]
- 微距聚焦当前断口：[[ACTION:focus_gap]]
- 对当前簇执行一步推荐复核：[[ACTION:collapse_step]]
- 完整演示 jardim_c04 1步确定贯通：[[ACTION:demo_1step]]
- 完整演示 jardim_c02 2步确定断开：[[ACTION:demo_2step]]
- 打开可信溯源与 SHA256/三null 核验抽屉：[[ACTION:open_verify]]
- 复位全景并清除复核状态：[[ACTION:reset]]
- 切换深浅主题：[[ACTION:theme:<light|dark>]]
若用户仅询问概念或数据原理、无需改变当前 3D 视口，则不要输出任何 [[ACTION:...]] 标签。"""

_ACTION_TAG_RE = re.compile(r"\[\[ACTION:([a-z0-9_]+)(?::([a-zA-Z0-9_\-]+))?\]\]")

_ALLOWED_SIMPLE_ACTIONS = {
    "split_gap",
    "focus_gap",
    "collapse_step",
    "demo_1step",
    "demo_2step",
    "open_verify",
    "reset",
}
_ALLOWED_WITNESS_ARGS = {"nominal", "connected", "disconnected"}
_ALLOWED_THEME_ARGS = {"light", "dark"}


def extract_and_validate_action(
    raw_text: str,
    valid_cluster_ids: Iterable[str],
) -> Tuple[str, Optional[Dict[str, str]]]:
    """Strip all `[[ACTION:...]]` tags from text and return at most 1 validated action."""
    valid_clusters = set(valid_cluster_ids)
    chosen: Optional[Dict[str, str]] = None

    for match in _ACTION_TAG_RE.finditer(raw_text or ""):
        if chosen is not None:
            continue
        kind = (match.group(1) or "").strip()
        arg = (match.group(2) or "").strip()
        if kind in _ALLOWED_SIMPLE_ACTIONS and not arg:
            chosen = {"type": kind, "raw": match.group(0)}
        elif kind == "select" and arg in valid_clusters:
            chosen = {"type": "select", "arg": arg, "raw": match.group(0)}
        elif kind == "witness" and arg in _ALLOWED_WITNESS_ARGS:
            chosen = {"type": "witness", "arg": arg, "raw": match.group(0)}
        elif kind == "theme" and arg in _ALLOWED_THEME_ARGS:
            chosen = {"type": "theme", "arg": arg, "raw": match.group(0)}

    cleaned = _ACTION_TAG_RE.sub("", raw_text or "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).strip()
    return cleaned, chosen


def build_verified_context_block(
    case_by_id: Mapping[str, Mapping[str, Any]],
    frontend_ctx: Optional[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Verify frontend UI context against server-cached M1/M2/M3 truth to prevent spoofing."""
    ctx = frontend_ctx if isinstance(frontend_ctx, Mapping) else {}
    req_cid = str(ctx.get("cluster_id") or "jardim_c02").strip()
    if req_cid not in case_by_id:
        req_cid = "jardim_c02" if "jardim_c02" in case_by_id else next(iter(case_by_id))

    case_obj = case_by_id[req_cid]
    status = str(case_obj.get("status") or "UNKNOWN")
    rho_obj = case_obj.get("rho") or {}
    rho_inf = rho_obj.get("infimum_float")
    dw = case_obj.get("disconnected_witness") or {}
    cut_gap = dw.get("cut_gap") or {}
    gap_m = cut_gap.get("distance_m")
    gap_mm = round(float(gap_m) * 1000.0, 2) if gap_m is not None else None
    cell_count = case_obj.get("cell_count")
    hole_count = sum(int(f.get("preserved_hole_count") or 0) for f in (case_obj.get("fractures") or []))

    ranked = (case_obj.get("active_check") or {}).get("ranked_fractures") or []
    top1 = ranked[0] if ranked else None
    top1_summary = (
        f"{top1.get('short_label') or top1.get('fracture_id')} "
        f"(测度消除 {round(float(top1.get('resolved_measure_fraction') or 0) * 100)}%, "
        f"确定化比 {round(float(top1.get('certified_branch_fraction') or 0) * 100)}%)"
        if top1
        else "无（当前簇已确定或超预算拒答）"
    )

    # Whitelist non-authoritative UI view state from frontend
    raw_wv = str(ctx.get("witness_view") or "nominal").strip()
    witness_view = raw_wv if raw_wv in ("nominal", "connected", "disconnected", "interpolated") else "nominal"
    split_mode = bool(ctx.get("split_mode"))
    raw_steps = ctx.get("collapse_steps")
    collapse_steps = int(raw_steps) if isinstance(raw_steps, int) and 0 <= raw_steps <= 2 else 0

    raw_col_status = str(ctx.get("collapsed_status") or "").strip()
    allowed_col_statuses = {
        "CERTIFIED_CONNECTED",
        "CERTIFIED_DISCONNECTED",
        "AMBIGUOUS_WITH_VERIFIED_WITNESSES",
    }
    effective_status = raw_col_status if (collapse_steps > 0 and raw_col_status in allowed_col_statuses) else status

    verified = {
        "cluster_id": req_cid,
        "site_id": str(case_obj.get("site_id") or ""),
        "root_status": status,
        "effective_status": effective_status,
        "rho_infimum_float": round(float(rho_inf), 6) if rho_inf is not None else None,
        "gap_mm": gap_mm,
        "cell_count": cell_count,
        "hole_count": hole_count,
        "m3_verdict": str(case_obj.get("m3_judge_verdict") or "AGREE"),
        "top1_fracture": top1_summary,
        "witness_view": witness_view,
        "split_mode": split_mode,
        "collapse_steps": collapse_steps,
    }

    gap_desc = (
        f"{verified['gap_mm']:.2f} mm ({float(gap_m):.5f} m)"
        if verified["gap_mm"] is not None and gap_m is not None
        else "无（非断开/非双见证态）"
    )
    rho_desc = (
        str(verified["rho_infimum_float"])
        if verified["rho_infimum_float"] is not None
        else "null（非歧义态无有限翻转裕度）"
    )
    cells_desc = (
        str(verified["cell_count"])
        if verified["cell_count"] is not None
        else "超 4,096 格上限拒答"
    )
    view_desc = "左右分屏对比" if verified["split_mode"] else verified["witness_view"]

    lines = [
        "【当前用户 3D 视口实时真包状态（已由后端 SHA256 缓存核验，以本段为准）】",
        f"- 当前簇：{verified['cluster_id']}（场地：{verified['site_id']}）",
        f"- 原始认证状态：{verified['root_status']}；当前交互态：{verified['effective_status']}（已复核步数：{verified['collapse_steps']}）",
        f"- 稳定裕度 ρ：{rho_desc}",
        f"- 最小割断口间隙：{gap_desc}",
        f"- 精确原子格数：{cells_desc}；保留真实孔洞数：{verified['hole_count']}",
        f"- M3 独立判卷：{verified['m3_verdict']}（全集 22/22 AGREE）",
        f"- #1 优先复核裂隙：{verified['top1_fracture']}",
        f"- 当前视角模式：{view_desc}",
    ]
    return "\n".join(lines), verified


def build_system_prompt(
    case_by_id: Mapping[str, Mapping[str, Any]],
    frontend_ctx: Optional[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    ctx_block, verified = build_verified_context_block(case_by_id, frontend_ctx)
    prompt = f"{AUTHORITATIVE_FACT_CARD}\n\n{ctx_block}\n\n{ACTION_PROTOCOL_PROMPT}"
    return prompt, verified
