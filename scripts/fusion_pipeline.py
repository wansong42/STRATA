# -*- coding: utf-8 -*-
"""T47: L4 孪生 demo 管道 —— 一键端到端 L4 融合.

串联:
  证据加载 (beishan L0 + FORGE L1)
  → 冲突审计 (跨源 Hungarian 匹配 + 一致性分档)
  → 门控融合 (加权 Frechet 均值 + bootstrap 置信锥)
  → 后验集合 (方向不确定 + 强度不确定 → M=20 DFN 实现 → p32_crit 区间)
  → HTML 报告 (融合组系表 + 置信锥 + 后验渗流区间 + 3D 示意 + 诚实边界)

用法:
    python scripts/fusion_pipeline.py \
        --npz data/real/beishan_wells.npz \
        --forge data/external/utah_forge_fmi/forge_fmi_2wells.pt \
        --site-name "beishan_forge_demo" \
        --K 4 --domain 50 50 50 --out-dir results/l4_demo/
"""

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def normal_to_dip_dipdir(nrm):
    """法向 → (dip, dip_direction)."""
    nrm = np.asarray(nrm, float)
    nrm = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12)
    dip = np.degrees(np.arccos(np.clip(np.abs(nrm[..., 2]), 0, 1)))
    dip_dir = np.degrees(np.arctan2(nrm[..., 0], nrm[..., 1])) % 360
    return dip, dip_dir


def _fig_to_base64(fig):
    """matplotlib Figure → base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# 阶段 1: 证据加载
# ---------------------------------------------------------------------------

def _aggregate_entries(entries):
    """将同类型的 EvidenceEntry 列表合并为单个条目 (池化法向).

    注: 合并后 lose 井级分辨率, 但融合需要"源级"组系而非"井级"组系.
    诚实边界: 同源池化在诚实口径下已证无增益 (P0 审计),
    但此处仅为获得合理 K 口径的融合输入, 非声称精度增益.

    structure_type: 全部条目同型且非 unknown 时透传; 否则回退 unknown.
    历史 bug (2026-08-28 修复): 曾无条件丢弃该字段 → FORGE natural/induced
    子条目被聚合成 unknown, --structure-type-aware 的类型隔离静默失效.
    """
    from fractureflow.l4.evidence import EvidenceEntry

    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    # 合并法向
    all_normals = []
    all_strikes = []
    for e in entries:
        if e.normals is not None:
            all_normals.append(e.normals)
        if e.strikes is not None:
            all_strikes.append(e.strikes)

    stypes = {getattr(e, "structure_type", "unknown") for e in entries}
    structure_type = stypes.pop() if len(stypes) == 1 else "unknown"

    merged = EvidenceEntry(
        source_type=entries[0].source_type,
        source_name=f"{entries[0].source_type}_aggregated",
        data_type=entries[0].data_type,
        normals=np.vstack(all_normals) if all_normals else None,
        strikes=np.concatenate(all_strikes) if all_strikes else None,
        weight=entries[0].weight,
        structure_type=structure_type,
        provenance={
            "aggregated_from": [e.source_name for e in entries],
            "n_original_entries": len(entries),
            "coordinate_system": entries[0].provenance.get("coordinate_system", "unknown"),
            "note": "同源井池化 (仅用于获得合理 K 口径融合输入, 非声称精度增益)",
        },
    )
    return merged


def stage1_load_evidence(args):
    """加载多源证据 → EvidenceBundle.

    策略: 按 source_type 聚合 (L0 井 → 单条, L1 井 → 单条, ...).
    这样 fuse_bundle 看到的是 2 个源 (L0 + L1) 而非 24 个,
    融合组系表将是 K 组而非 K×N_wells 组.
    """
    from fractureflow.l4.evidence import (
        from_beishan_npz, from_forge_fmi_pt, build_bundle, EvidenceEntry,
    )

    print("\n" + "=" * 60)
    print("  阶段 1: 证据加载 (按源类型聚合)")
    print("=" * 60)

    # 按类型收集
    by_type = {}  # type -> list[EvidenceEntry]

    # L0: beishan 钻孔
    if args.npz:
        l0_entries = from_beishan_npz(args.npz)
        by_type.setdefault("L0", []).extend(l0_entries)
        n_obs = sum(e.n_obs for e in l0_entries)
        print(f"  [L0] beishan: {len(l0_entries)} 井, {n_obs} 条法向观测")

    # L1: FORGE FMI
    if args.forge:
        l1_entries = from_forge_fmi_pt(args.forge)
        by_type.setdefault("L1", []).extend(l1_entries)
        n_obs = sum(e.n_obs for e in l1_entries)
        print(f"  [L1] FORGE FMI: {len(l1_entries)} 井, {n_obs} 条法向观测")

    if not by_type:
        raise ValueError("至少需要 --npz 或 --forge 之一")

    # 聚合各类型为单条证据
    aggregated_entries = []
    aware = getattr(args, "structure_type_aware", False)
    if aware:
        # T50 类型隔离开启时按 (source_type, structure_type) 分组聚合;
        # 否则 FORGE natural/induced 子条目会被池化抹掉 structure_type,
        # _fuse_type_aware 的 known_types 恒空 → 开关被架空 (2026-08-28 修复).
        groups = {}
        for stype_key, ents in by_type.items():
            for e in ents:
                groups.setdefault(
                    (stype_key, getattr(e, "structure_type", "unknown")), []).append(e)
    else:
        groups = {(stype_key, None): ents for stype_key, ents in by_type.items()}
    for (stype_key, struct_t), entries in sorted(groups.items()):
        merged = _aggregate_entries(entries)
        if merged is not None:
            aggregated_entries.append(merged)
            tag = f"{stype_key}" + (f"/{struct_t}" if struct_t else "")
            print(f"  → 聚合 {tag}: {len(entries)} 条目 → 1 条 "
                  f"({merged.n_obs} 观测)")

    # build_bundle expects *lists*, so wrap each entry in a list
    bundle = build_bundle(*([e] for e in aggregated_entries),
                          site_name=args.site_name)
    summary = bundle.summary()
    print(f"  → 证据集合: {summary['n_sources']} 源 (聚合后), "
          f"类型: {summary['source_types']}")

    return bundle, summary


# ---------------------------------------------------------------------------
# 阶段 2: 源内组系
# ---------------------------------------------------------------------------

def stage2_source_grouping(bundle, K=4, seed=42, well_detail=None):
    """各源独立 spherical k-means → 各源组系.

    well_detail: 可选, dict {source_type: list of (well_name, n_obs)} for display.
    """
    from fractureflow.l4.fuse import parse_source_table

    print("\n" + "=" * 60)
    print("  阶段 2: 源内组系 (spherical k-means, 聚合后源级)")
    print("=" * 60)

    source_tables = []
    for entry in bundle.entries:
        st = parse_source_table(entry, K, seed=seed)
        if st is not None and st.K > 0:
            source_tables.append(st)
            dip, dip_dir = normal_to_dip_dipdir(st.centers)
            # 打印各组方向
            group_dirs = ", ".join(
                f"g{k}({dip[k]:.0f}°/{dip_dir[k]:.0f}°)"
                for k in range(st.K)
            )
            n_wells = 1
            if well_detail and entry.source_type in well_detail:
                n_wells = len(well_detail[entry.source_type])
            print(f"  [{st.source_type}] {st.source_name}: "
                  f"K={st.K}, n={entry.n_obs} ({n_wells} 原始井)")
            print(f"    组系: {group_dirs}")

    return source_tables


# ---------------------------------------------------------------------------
# 阶段 3: 冲突审计
# ---------------------------------------------------------------------------

def stage3_conflict_audit(source_tables):
    """跨源配对 → consistent/partial/conflict."""
    from fractureflow.l4.fuse import hungarian_match_groups, classify_consistency

    print("\n" + "=" * 60)
    print("  阶段 3: 冲突审计 (跨源 Hungarian 匹配)")
    print("=" * 60)

    n_consistent = 0
    n_partial = 0
    n_conflict = 0
    all_matches = []

    for i in range(len(source_tables)):
        for j in range(i + 1, len(source_tables)):
            sa = source_tables[i]
            sb = source_tables[j]
            matches = hungarian_match_groups(sa.centers, sb.centers)
            for (gi, gj, angle) in matches:
                decision = classify_consistency(angle)
                if decision == "consistent":
                    n_consistent += 1
                elif decision == "partial":
                    n_partial += 1
                else:
                    n_conflict += 1
                all_matches.append({
                    "source_a": sa.source_type,
                    "source_b": sb.source_type,
                    "group_a": gi,
                    "group_b": gj,
                    "angle_deg": round(angle, 2),
                    "decision": decision,
                })
                print(f"  {sa.source_type}:g{gi} ↔ {sb.source_type}:g{gj} "
                      f"= {angle:.1f}° → {decision}")

    audit = {
        "n_consistent": n_consistent,
        "n_partial": n_partial,
        "n_conflict": n_conflict,
        "matches": all_matches,
    }
    print(f"  → 一致: {n_consistent}, 部分: {n_partial}, 冲突: {n_conflict}")

    return audit


# ---------------------------------------------------------------------------
# 阶段 4: 融合
# ---------------------------------------------------------------------------

def stage4_fuse(bundle, K=4, seed=42, type_aware=False):
    """冲突门控融合 → UnifiedSetTable + 置信锥."""
    from fractureflow.l4.fuse import fuse_bundle, unified_set_table_to_dict

    print("\n" + "=" * 60)
    mode = "类型隔离融合 (T50)" if type_aware else "冲突门控融合 (加权 Frechet 均值)"
    print(f"  阶段 4: {mode}")
    print("=" * 60)

    result = fuse_bundle(bundle, K=K, seed=seed, type_aware=type_aware)
    ust = result.unified_set_table

    dip, dip_dir = normal_to_dip_dipdir(ust.centers)
    for k in range(ust.K):
        sources = ", ".join(ust.source_map[k]) if ust.source_map[k] else "N/A"
        types_str = ""
        if ust.structure_types and k < len(ust.structure_types) and ust.structure_types[k]:
            types_str = f", types={ust.structure_types[k]}"
        print(f"  组 {k}: dip={dip[k]:.1f}°, dip_dir={dip_dir[k]:.1f}°, "
              f"κ={ust.concentrations[k]:.2f}, "
              f"锥95={ust.cones_95[k]:.1f}°, "
              f"来源=[{sources}]{types_str}")

    print(f"  → 融合组系: K={ust.K}, "
          f"一致对={result.meta['n_consistent_pairs']}, "
          f"部分对={result.meta['n_partial_pairs']}, "
          f"冲突对={result.meta['n_conflict_pairs']}")
    if type_aware:
        csr = result.conflicts_report.get("n_cross_subspace_merged", 0)
        nsub = result.conflicts_report.get("n_subspaces", 0)
        print(f"  → 类型隔离: {nsub} 子空间, {csr} 跨子空间合并")

    return result, ust


# ---------------------------------------------------------------------------
# 阶段 5: 后验集合
# ---------------------------------------------------------------------------

def stage5_ensemble(ust, p32_estimate=0.3, p32_range=(0.1, 0.6),
                    beta=3.5, domain=(50, 50, 50), M=20, seed=42):
    """锥形方向不确定 + 强度不确定 → M 实现 → p32_crit 区间."""
    from fractureflow.l4.ensemble import generate_ensemble, ensemble_to_dict

    print("\n" + "=" * 60)
    print(f"  阶段 5: 后验网络集合 (M={M} 实现)")
    print("=" * 60)

    t0 = time.time()
    ensemble = generate_ensemble(
        ust,
        p32_estimate=p32_estimate,
        p32_range=p32_range,
        beta=beta,
        domain=domain,
        M=M,
        seed=seed,
    )
    elapsed = time.time() - t0

    lo, med, hi = ensemble.p32_crit_interval
    print(f"  p32_crit 区间: [{lo:.3f}, {med:.3f}, {hi:.3f}] "
          f"(5%-50%-95%)")
    print(f"  方向不确定 (锥半角): "
          f"{', '.join(f'{a:.1f}°' for a in ensemble.direction_uncertainty)}")
    print(f"  强度不确定: {ensemble.intensity_uncertainty:.3f}")
    print(f"  平均裂隙数: {ensemble.meta['n_fractures_mean']:.0f} "
          f"± {ensemble.meta['n_fractures_std']:.0f}")
    print(f"  → 耗时: {elapsed:.1f}s")

    ens_dict = ensemble_to_dict(ensemble)
    ens_dict["elapsed_sec"] = round(elapsed, 2)

    return ensemble, ens_dict


# ---------------------------------------------------------------------------
# 阶段 6: 3D 可视化
# ---------------------------------------------------------------------------

def stage6_render_3d(ust, domain=(50, 50, 50), seed=42, max_disks=600):
    """生成 DFN 实现 + matplotlib 3D 图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import font_manager

    from fractureflow.dfn import generate_dfn

    print("\n" + "=" * 60)
    print("  阶段 6: 3D 可视化 (DFN 实现)")
    print("=" * 60)

    # 中文字体
    cn_font = None
    for fn in ["SimHei", "Microsoft YaHei", "SimSun", "PingFang SC"]:
        if any(f.name == fn for f in font_manager.fontManager.ttflist):
            cn_font = fn
            break
    if cn_font:
        plt.rcParams["font.sans-serif"] = [cn_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 生成一个 DFN 实现 (中值 P32)
    st = ust.to_settable()
    dfn = generate_dfn(st, p32=0.3, beta=3.5, domain=domain, seed=seed)
    print(f"  DFN 生成: {dfn.n_fractures} 条裂隙")

    K = ust.K
    cmap = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))

    # 子采样
    n_disk = dfn.n_fractures
    if n_disk > max_disks:
        rng = np.random.default_rng(seed)
        idx_disk = rng.choice(n_disk, max_disks, replace=False)
    else:
        idx_disk = np.arange(n_disk)

    Lx, Ly, Lz = domain

    # 找最大主走向方向
    main_set = int(np.argmin(np.abs(ust.centers[:, 2])))
    main_dir = ust.centers[main_set]
    # 法向水平方位 = 倾向; 沿走向看需 +90° (与 site_model.render_3d 同修复)
    main_azim = float((np.degrees(np.arctan2(main_dir[0], main_dir[1])) + 90.0) % 360.0)

    views = [
        ("top", 90, 0, "俯视"),
        ("side", 0, 0, "侧视"),
        ("oblique", 30, 45, "斜视"),
        ("along_strike", 0, main_azim, "沿最大主走向"),
    ]

    def _disk_polygon(center, normal, radius, n_vertices=12):
        normal = _unit(normal)
        if abs(normal[2]) < 0.9:
            ref = np.array([0, 0, 1.0])
        else:
            ref = np.array([1.0, 0, 0])
        u = np.cross(normal, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        verts = (center[:, None]
                 + radius * (np.cos(angles)[None, :] * u[:, None]
                             + np.sin(angles)[None, :] * v[:, None]))
        return verts.T

    b64_images = {}
    for view_name, elev, azim, view_cn in views:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 画裂隙盘
        if len(idx_disk) > 0:
            verts_list = []
            colors_list = []
            for idx in idx_disk:
                c = dfn.centers[idx]
                n = dfn.normals[idx]
                r = dfn.radii[idx]
                k = int(dfn.sets[idx])
                verts = _disk_polygon(c, n, r)
                verts_list.append(verts)
                colors_list.append(cmap[k % len(cmap)])

            poly = Poly3DCollection(verts_list, alpha=0.25,
                                    linewidths=0.1, edgecolors="gray")
            poly.set_facecolor(colors_list)
            ax.add_collection3d(poly)

        # 域边框 (DFN 中心由 rng.uniform(-0.5,0.5)*domain 生成, 范围 [-L/2, L/2])
        ax.set_xlim(-Lx / 2, Lx / 2)
        ax.set_ylim(-Ly / 2, Ly / 2)
        ax.set_zlim(-Lz / 2, Lz / 2)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"L4 融合 DFN — {view_cn} "
                     f"(elev={elev}°, azim={azim}°)", fontsize=11)
        ax.view_init(elev=elev, azim=azim)

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap[k], alpha=0.5,
                                 label=f"组 {k}") for k in range(K)]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        b64 = _fig_to_base64(fig)
        b64_images[view_name] = b64
        print(f"  {view_cn} → 渲染完成")

    return b64_images, dfn


# ---------------------------------------------------------------------------
# 阶段 7: HTML 报告
# ---------------------------------------------------------------------------

def stage7_generate_html(site_name, summary, audit, fuse_result, ust,
                         ens_dict, b64_images, args, elapsed_total):
    """生成 HTML 报告."""
    from fractureflow.l4.fuse import unified_set_table_to_dict

    print("\n" + "=" * 60)
    print("  阶段 7: HTML 报告生成")
    print("=" * 60)

    ust_dict = unified_set_table_to_dict(ust)
    dip, dip_dir = normal_to_dip_dipdir(ust.centers)

    # 融合组系表 HTML
    set_table_rows = ""
    for k in range(ust.K):
        sources = ", ".join(ust.source_map[k]) if ust.source_map[k] else "N/A"
        set_table_rows += f"""
        <tr>
          <td>{k}</td>
          <td>{dip[k]:.1f}°</td>
          <td>{dip_dir[k]:.1f}°</td>
          <td>{ust.concentrations[k]:.2f}</td>
          <td>{ust.cones_95[k]:.1f}°</td>
          <td>{ust.proportions[k]:.3f}</td>
          <td>{sources}</td>
        </tr>"""

    # 冲突审计行
    conflict_rows = ""
    for m in audit.get("matches", []):
        conflict_rows += f"""
        <tr>
          <td>{m['source_a']}:g{m['group_a']}</td>
          <td>{m['source_b']}:g{m['group_b']}</td>
          <td>{m['angle_deg']:.1f}°</td>
          <td>{m['decision']}</td>
        </tr>"""

    # 3D 图 HTML
    imgs_html = ""
    view_names_cn = {
        "top": "俯视", "side": "侧视",
        "oblique": "斜视", "along_strike": "沿最大主走向",
    }
    for vname, b64 in b64_images.items():
        cn = view_names_cn.get(vname, vname)
        imgs_html += f"""
        <div style="display:inline-block; margin:10px; text-align:center;">
          <img src="data:image/png;base64,{b64}" width="480" />
          <p><b>{cn}</b></p>
        </div>"""

    # p32_crit 区间
    p32_lo = ens_dict["p32_crit_interval"]["lower"]
    p32_med = ens_dict["p32_crit_interval"]["median"]
    p32_hi = ens_dict["p32_crit_interval"]["upper"]

    # 场景指标
    scenario_html = ""
    for scen, sdata in ens_dict.get("scenario_intervals", {}).items():
        scenario_html += f"""
        <tr>
          <td>{scen}</td>
          <td>{sdata['lower']:.3f}</td>
          <td>{sdata['median']:.3f}</td>
          <td>{sdata['upper']:.3f}</td>
          <td>{sdata['n_valid']}</td>
        </tr>"""

    # 证据摘要
    evidence_html = ""
    for stype, sinfo in summary.get("by_type", {}).items():
        evidence_html += f"""
        <tr>
          <td>{stype}</td>
          <td>{sinfo['n_entries']}</td>
          <td>{sinfo['n_obs']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>L4 多源融合智能重构报告 — {site_name}</title>
<style>
body {{ font-family: "Microsoft YaHei", "SimHei", sans-serif; margin: 40px; background: #fafafa; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
th {{ background: #16213e; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 15px 0; }}
.honest {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 15px 0; }}
.metric {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; margin: 15px 0; }}
.container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<div class="container">

<h1>L4 多源融合智能重构报告</h1>
<p><b>场地:</b> {site_name} | <b>生成时间:</b> {time.strftime("%Y-%m-%d %H:%M:%S")} | <b>总耗时:</b> {elapsed_total:.1f}s</p>

<div class="metric">
  <b>核心结论:</b> 融合 K={ust.K} 组系, p32_crit 区间 = [{p32_lo:.3f}, {p32_med:.3f}, {p32_hi:.3f}] (5%-50%-95%).
  跨源一致对: {audit['n_consistent']}, 部分对: {audit['n_partial']}, 冲突对: {audit['n_conflict']}.
</div>

<!-- 章节 1: 证据摘要 -->
<h2>章节 1: 证据摘要</h2>
<table>
  <tr><th>源类型</th><th>条目数</th><th>观测数</th></tr>
  {evidence_html}
</table>
<p class="note">
  <b>来源与口径:</b> L0 = 钻孔编录表 (beishan npz), L1 = FMI 迹线 (FORGE pt).
  两源法向均在同一统一约定帧 (well-local ENU; T58 血统仲裁验证与 beishan 同帧, 无需旋转).
  各源初始等权 (weight=1.0).
</p>

<!-- 章节 2: 冲突审计 -->
<h2>章节 2: 冲突审计</h2>
<div class="metric">
  一致 (<20°): <b>{audit['n_consistent']}</b> 对 |
  部分 (20-35°): <b>{audit['n_partial']}</b> 对 |
  冲突 (>35°): <b>{audit['n_conflict']}</b> 对
</div>
<table>
  <tr><th>源 A</th><th>源 B</th><th>角距</th><th>判定</th></tr>
  {conflict_rows}
</table>
<p class="note">
  <b>来源与口径:</b> 跨源组配对使用匈牙利算法 (代价 = acos|&lt;ci,cj&gt;|).
  决策阈值: &lt;20° 一致 (加权 Frechet 合并), 20-35° 部分 (仅合并最一致对), &gt;35° 冲突 (独立保持).
</p>

<!-- 章节 3: 融合组系表 -->
<h2>章节 3: 融合组系表</h2>
<table>
  <tr><th>组</th><th>倾角 (°)</th><th>倾向 (°)</th><th>浓度 κ</th>
      <th>95% 置信锥 (°)</th><th>占比</th><th>贡献源</th></tr>
  {set_table_rows}
</table>
<p class="note">
  <b>来源与口径:</b> 方向 = 加权 Frechet 均值 (权重 = √n × 源质量先验).
  置信锥 = bootstrap 1000 次重采样的 95% 分位数.
  κ = vMF 浓度估计 (越大越集中).
</p>

<!-- 章节 4: p32_crit 区间 -->
<h2>章节 4: 后验渗流区间</h2>
<div class="metric">
  p32_crit 90% 区间: <b>[{p32_lo:.3f}, {p32_hi:.3f}]</b>, 中值 = <b>{p32_med:.3f}</b> m²/m³
</div>
<table>
  <tr><th>场景</th><th>下限 (5%)</th><th>中值 (50%)</th><th>上限 (95%)</th><th>有效实现数</th></tr>
  {scenario_html}
</table>
<p class="note">
  <b>来源与口径:</b> M={ens_dict['meta']['M']} 个 DFN 后验实现.
  方向不确定性 = 在 cone_95 内随机扰动; 强度不确定性 = 在 [{ens_dict['meta']['p32_range'][0]}, {ens_dict['meta']['p32_range'][1]}] 内独立抽样 P32.
  每个实现跑渗流曲线 (6 种子粗网格) → 收集 p32_crit 分布.
</p>

<!-- 章节 5: 3D 示意 -->
<h2>章节 5: DFN 三维示意</h2>
<div style="text-align:center;">
  {imgs_html}
</div>

<!-- 诚实边界 -->
<h2>诚实边界</h2>
<div class="honest">
  <p><b>1. 融合增益:</b> 仅在跨源互补时成立. 同源池化 (多井联合) 在诚实口径下已证无增益 (P0 审计).
  本管道只融合异源 (L0 钻孔 + L1 FMI).</p>
  <p><b>2. 置信锥:</b> = bootstrap 1000 次重采样的 95% 分位数, 非"分布无关保证".
  实际覆盖率依赖观测分布假设.</p>
  <p><b>3. DFN 实现:</b> = Baecher 盘模型随机实现 (裂隙位置随机), 非实测定位.
  渗流阈值仅供筛查级参考, 不可用于工程判定.</p>
  <p><b>4. FORGE 坐标:</b> FMI 法向为井局部 ENU; T58 血统仲裁验证其与
   beishan/统一约定同帧, 直接使用, 无旋转矩阵 (旧版"已旋转到 NED"为陈旧声明).</p>
  <p><b>5. 后验区间:</b> 方向不确定性与强度不确定性解耦报告, 不混淆.
   β={ens_dict['meta']['beta']}, r_min={ens_dict['meta']['r_min']}.</p>
  <p><b>6. 筛查级结论:</b> 非真实连通性判定, 需现场标定.</p>
  <p><b>7. 连通判据与区间语义 (T66 声明):</b> 连通性采用球近似判据
   (中心距 &lt; r_i+r_j, 不使用裂隙法向), 因此方向不确定性在结构上无法影响渗流阈值;
   p32_crit 区间宽度可能为零 (退化), 此时区间仅表示"模型内无可传播的不确定性",
   不表示"真实不确定性为零". 真实不确定性需现场数据标定.</p>
</div>

<hr>
<p style="color:#666; font-size:12px;">
  T47 L4 孪生 demo 管道 | seed={args.seed} | K={args.K} | domain={list(args.domain)} |
  不训网络、不逐点预测 | 全管线耗时 {elapsed_total:.1f}s
</p>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="T47: L4 孪生 demo 管道 — 一键端到端 L4 融合")
    ap.add_argument("--npz", default=None,
                    help="beishan_wells.npz 路径 (L0)")
    ap.add_argument("--forge", default=None,
                    help="forge_fmi_2wells.pt 路径 (L1)")
    ap.add_argument("--site-name", default="beishan_forge_demo",
                    help="场地名称")
    ap.add_argument("--K", type=int, default=4, help="组系数")
    ap.add_argument("--domain", nargs=3, type=float, default=[50, 50, 50],
                    help="域尺寸 (m)")
    ap.add_argument("--p32-estimate", type=float, default=0.3,
                    help="P32 中值估计")
    ap.add_argument("--p32-range", nargs=2, type=float, default=[0.1, 0.6],
                    help="P32 估计区间 (lower upper)")
    ap.add_argument("--beta", type=float, default=3.5, help="幂律指数")
    ap.add_argument("--M", type=int, default=20, help="后验集合实现数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--structure-type-aware", action="store_true",
                    help="T50 类型隔离融合: 先按 (source_type, structure_type) 划分子空间,"
                         "子空间内做既有融合, 再跨子空间匈牙利匹配合并同向组")
    ap.add_argument("--out-dir", default="results/l4_demo/",
                    help="输出目录")
    args = ap.parse_args()

    t_start = time.time()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    domain = tuple(args.domain)
    p32_range = tuple(args.p32_range)

    # 收集原始井信息 (用于阶段 2 展示)
    well_detail = {}
    if args.npz:
        from fractureflow.l4.evidence import from_beishan_npz
        for e in from_beishan_npz(args.npz):
            well_detail.setdefault("L0", []).append((e.source_name, e.n_obs))
    if args.forge:
        from fractureflow.l4.evidence import from_forge_fmi_pt
        for e in from_forge_fmi_pt(args.forge):
            well_detail.setdefault("L1", []).append((e.source_name, e.n_obs))

    # 阶段 1: 证据加载 (按源类型聚合)
    bundle, summary = stage1_load_evidence(args)

    # 阶段 2: 源内组系
    source_tables = stage2_source_grouping(
        bundle, K=args.K, seed=args.seed, well_detail=well_detail)

    # 阶段 3: 冲突审计 (跨源配对)
    audit = stage3_conflict_audit(source_tables)

    # 阶段 4: 融合 (T50: 类型隔离模式开关)
    fuse_result, ust = stage4_fuse(
        bundle, K=args.K, seed=args.seed,
        type_aware=getattr(args, "structure_type_aware", False))

    # 阶段 5: 后验集合
    ensemble, ens_dict = stage5_ensemble(
        ust,
        p32_estimate=args.p32_estimate,
        p32_range=p32_range,
        beta=args.beta,
        domain=domain,
        M=args.M,
        seed=args.seed,
    )

    # 阶段 6: 3D 可视化
    b64_images, dfn = stage6_render_3d(ust, domain=domain, seed=args.seed)

    # 阶段 7: HTML 报告
    elapsed_total = time.time() - t_start
    html = stage7_generate_html(
        args.site_name, summary, audit, fuse_result, ust,
        ens_dict, b64_images, args, elapsed_total,
    )

    # 落盘
    html_path = os.path.join(args.out_dir, "l4_fusion_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML 报告: {html_path}")

    # JSON 落盘
    from fractureflow.l4.fuse import unified_set_table_to_dict
    output_json = {
        "site_name": args.site_name,
        "K": args.K,
        "domain": list(domain),
        "seed": args.seed,
        "structure_type_aware": getattr(args, "structure_type_aware", False),
        "evidence_summary": summary,
        "conflict_audit": audit,
        "unified_set_table": unified_set_table_to_dict(ust),
        "ensemble": ens_dict,
        "dfn_n_fractures": dfn.n_fractures,
        "elapsed_total_sec": round(elapsed_total, 2),
        "honest_boundary": {
            "fusion_gain": "仅在跨源互补时成立, 同源池化已证无增益",
            "confidence_cone": "bootstrap 1000 次 95% 分位数, 非分布无关保证",
            "dfn_realization": "Baecher 盘模型随机实现, 非实测定位",
            "forge_coordinate": "井局部 ENU 直接使用 (T58 血统仲裁: 与统一约定同帧, 无旋转)",
            "screening_level": "筛查级结论, 非真实连通性判定, 需现场标定",
            "connectivity_criterion": (
                "球近似判据 (中心距 < r_i+r_j, 不使用法向) — 方向不确定性结构上"
                "无法影响渗流阈值; p32_crit 区间可能零宽度退化, 零宽度 = 模型内"
                "无可传播不确定性, 非真实不确定性为零 (T66 声明)"
            ),
        },
    }
    json_path = os.path.join(args.out_dir, "l4_fusion_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"  JSON 结果: {json_path}")

    print("\n" + "=" * 60)
    print(f"  T47 L4 孪生 demo 管道完成!")
    print(f"  总耗时: {elapsed_total:.1f}s")
    print(f"  输出目录: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
