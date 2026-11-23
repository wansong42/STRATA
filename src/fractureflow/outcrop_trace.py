# -*- coding: utf-8 -*-
"""R106 · L2 露头迹线产线 单入口 CLI（产品化收口）

定位: L2（露头 2D 迹线档）从 R95 MVP 到「产品线模块」——单入口 CLI + 能力声明 + 守卫测试。
R95 管线本体（scripts/l2_outcrop_trace_pipeline.py, 39KB）只读引用，不改核心逻辑。

用法:
    python -m fractureflow.outcrop_trace <影像目录> [--out-dir <dir>]
    python -m fractureflow.outcrop_trace <单张影像> [--out-dir <dir>]   # 也接受单文件

能力边界 (详见 results/v_r106_capability.md):
    能   = 2D 影像迹线自动统计 (方向域 [0,180)，分钟级，公开影像演示 4/4 例)
    不能 = 真三维产状 (dip) 与长度/尺度 (单图信息上限)
    待定 = 真实露头照片人工标注对照 (R102 在飞)
    定位 = 筛查级/演示口径

血统: 复用 R95 现役函数 (scripts/l2_outcrop_trace_pipeline.py)，核心逻辑零改动。
      走向约定 = 2D 影像平面方位 (模180°，从北顺时针)；影像"上方向=北"假设写死并声明。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# 锚 ROOT 与 R95 管线导入（只读引用，零改动）
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
if os.path.join(_PROJECT_ROOT, "vision") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vision"))

# R95 现役函数 (冻结，不修改)
from l2_outcrop_trace_pipeline import (  # noqa: E402
    DET_PARAMS,
    build_set_table,
    cluster_sets,
    detect_trace_lines,
    filter_short,
    merge_collinear,
    run_detection,
    save_csv,
    save_rose_diagram,
    save_trace_overlay,
    segs_to_traces,
)

# 中文路径安全读图 (vision 模块)
from imio import imread_robust  # noqa: E402

# ---------------------------------------------------------------------------
# 版本 / 血统元数据
# ---------------------------------------------------------------------------
VERSION = "1.0.0"
TASK = "R106_L2_outcrop_trace_productized"
POSITIONING = "筛查级/演示口径"

# 能力声明（三色卡，语义镜像 results/v_r106_capability.md）
CAPABILITY = {
    "positioning": POSITIONING,
    "can": {
        "summary": "2D 影像迹线自动统计",
        "details": [
            "方向域 [0,180)：读图→CLAHE→Canny→概率Hough→共线合并→走向统计→组系表→三件套可视化→报告",
            "分钟级 (单张 224x224 ~秒级；512x512 数十秒级)",
            "公开影像演示 4/4 例 (Khibiny 11 / Alishan 12 / Oman 8 / Åland 25 条检测迹线)",
        ],
        "backing": "R95 ACCEPTED (2026-08-27)；results/l2_outcrop_trace_mvp/summary.json",
    },
    "cannot": {
        "summary": "真三维产状 (dip) 与长度/尺度",
        "details": [
            "单图信息上限：2D 影像平面无法恢复真三维产状 (dip/倾向)，需相机位姿/多视图/已知产状",
            "无比例尺标定，长度/尺度仅像素或渲染比例，无长度承诺",
        ],
        "backing": "R95 诚实边界；results/l2_outcrop_trace_mvp/summary.json honesty_notes",
    },
    "pending": {
        "summary": "真实露头照片人工标注对照",
        "details": [
            "R102 在飞 (预注册 + 自标对照 + 台账)，转正后并入本产品线能力声明",
        ],
        "backing": "tasks/R102_任务书_★★☆_L2升格TierA_真实露头对照验证.md",
    },
    "forbidden_claims": [
        "不得出现『三维产状/尺寸/长度承诺』类能力主张",
        "不得升格为『生产/勘察结论』口径，维持筛查级/演示口径",
    ],
}

# 易变键（不进指纹）
_VOLATILE_KEYS = {
    "V_time", "V_duration_sec", "generated_at", "timestamp",
    "elapsed_sec", "V_input_path_abs", "V_duration",
}


def _sha16_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _sha16_of_json(obj) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return _sha16_of_bytes(text.encode("utf-8"))


def fingerprint_of(outputs: dict) -> str:
    """幂等指纹：剔除易变键后对稳定字段求 sha16。同输入同输出即逐位相等。"""
    stable = {}
    for k, v in outputs.items():
        if k in _VOLATILE_KEYS:
            continue
        stable[k] = v
    return _sha16_of_json(stable)


def compute_image_fingerprint(img_bgr) -> str:
    """输入影像血统指纹 (sha16 of raw pixels)."""
    return _sha16_of_bytes(img_bgr.tobytes())


def _imwrite_robust(path: str, img_bgr) -> None:
    """中文路径安全的 cv2.imwrite (imencode + 二进制写, 兼容 R95 既有坑)."""
    import cv2
    ok, enc = cv2.imencode(os.path.splitext(path)[1], img_bgr)
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(enc.tobytes())


def _rel_out_dir(out_dir: str) -> str:
    """把 out_dir 归一为相对路径 (避免 cv2.imwrite 在 Windows 中文绝对路径下静默失败)."""
    if os.path.isabs(out_dir):
        try:
            rel = os.path.relpath(out_dir, _PROJECT_ROOT)
            # 仅当在项目根内才转为相对路径；否则保留原绝对路径 (用户自定义外部目录)
            if not rel.startswith(".."):
                return rel
        except ValueError:
            pass
    return out_dir


# ---------------------------------------------------------------------------
# 核心：对单张影像跑全链（复用 R95 run_detection，零改动）
# ---------------------------------------------------------------------------

def process_image(img_bgr, case: str, out_dir: str,
                  scale_m_per_px: float = 0.012,
                  min_seg_len_px: float = 12.0) -> dict:
    """对单张影像跑完整产线：检测→三件套→组系表。

    Args:
        img_bgr: BGR 影像 (cv2 读入)
        case: 案例标识 (用于输出文件名)
        out_dir: 输出目录 (在项目根内时自动转相对路径, 规避中文绝对路径写入坑)
        scale_m_per_px: 长度尺度 (占位/渲染比例；无比例尺标定时不承诺)
        min_seg_len_px: 最短迹线过滤 (R95 产品参数，冻结)

    Returns:
        record: 与 R95 summary 同构的案例记录 (含 n_detected_traces / detected_sets 等)
    """
    out_dir = _rel_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # R95 run_detection: 线段→合并→过滤→迹线→分组 (seed=0 固定，确定性)
    segs, traces, assigns, centers, n_sets = run_detection(
        img_bgr, scale_m_per_px, min_seg_len_px=min_seg_len_px)

    # 三件套 (复用 R95 现役 API; out_dir 已归一为相对路径, 规避中文绝对路径写入坑)
    overlay_path = os.path.join(out_dir, f"overlay_{case}.png")
    save_trace_overlay(img_bgr, traces, assigns, overlay_path,
                       title=f"L2 trace overlay | {case}")
    rose_path = os.path.join(out_dir, f"rose_{case}.png")
    save_rose_diagram([t["strike_deg"] for t in traces], assigns, rose_path,
                      title=f"走向玫瑰图 · {len(traces)} 条迹线")
    set_rows = build_set_table(traces, assigns, n_sets, scale_m_per_px)
    st_path, tr_path = save_csv(out_dir, case, set_rows, traces, assigns,
                                scale_m_per_px)

    return {
        "case": case,
        "n_detected_traces": len(traces),
        "scale_m_per_px": round(float(scale_m_per_px), 6),
        "detected_sets": [
            {"set_id": int(s["set_id"]),
             "strike_center_deg": s["strike_center_deg"],
             "n_traces": s["n_traces"]}
            for s in set_rows
        ],
        "set_table_csv": os.path.relpath(st_path, _PROJECT_ROOT),
        "traces_csv": os.path.relpath(tr_path, _PROJECT_ROOT),
        "overlay_png": os.path.relpath(
            os.path.join(out_dir, f"overlay_{case}.png"), _PROJECT_ROOT),
        "rose_png": os.path.relpath(
            os.path.join(out_dir, f"rose_{case}.png"), _PROJECT_ROOT),
        "strike_domain": "[0,180)",
        "input_fingerprint": _sha16_of_bytes(img_bgr.tobytes()),
        "_honest": "2D 影像平面方位 (模180°, 从北顺时针)；单图不可得真三维产状 (dip)；"
                   "无比例尺标定时长度/尺度不承诺.",
    }


# ---------------------------------------------------------------------------
# 目录级批量入口
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(image_path: str):
    """返回 (case, abs_path) 列表：目录则遍历影像文件；单文件则单张。"""
    if os.path.isdir(image_path):
        items = []
        for fn in sorted(os.listdir(image_path)):
            if os.path.splitext(fn)[1].lower() in _IMG_EXTS:
                items.append((os.path.splitext(fn)[0], os.path.join(image_path, fn)))
        if not items:
            raise FileNotFoundError(f"目录中无影像文件: {image_path}")
        return items
    if os.path.isfile(image_path):
        base = os.path.splitext(os.path.basename(image_path))[0]
        return [(base, image_path)]
    raise FileNotFoundError(f"路径不存在: {image_path}")


def process_directory(image_dir: str, out_dir: str,
                      scale_m_per_px: float = 0.012,
                      min_seg_len_px: float = 12.0) -> dict:
    """一键全链：批量跑目录内全部影像。

    Returns:
        doc: 结果文档 (含指纹 / cases / 能力声明指针)
    """
    # 以项目根为工作目录（cwd 无关：无论从 src/ 或项目根以 -m 调用，输出都落项目根）
    os.chdir(_PROJECT_ROOT)
    records = []
    for case, path in _iter_images(image_dir):
        img = imread_robust(path)
        rec = process_image(img, case, out_dir,
                            scale_m_per_px=scale_m_per_px,
                            min_seg_len_px=min_seg_len_px)
        rec["input_path"] = os.path.relpath(path, _PROJECT_ROOT)
        records.append(rec)

    doc = {
        "task": TASK,
        "version": VERSION,
        "positioning": POSITIONING,
        "input": image_dir,
        "n_images": len(records),
        "cases": records,
        "capability_ref": "results/v_r106_capability.md",
        "lineage": {
            "pipeline": "scripts/l2_outcrop_trace_pipeline.py (R95 现役, 只读引用)",
            "entry": "python -m fractureflow.outcrop_trace <影像目录>",
        },
    }
    # 幂等指纹 (剔除易变键后求 sha16)
    doc["fingerprint"] = fingerprint_of(doc)

    # 落盘 per-run summary
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "v_r106_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)

    # 简洁控制台输出
    for r in records:
        sets_str = "; ".join(
            f"Set{s['set_id']+1}@{s['strike_center_deg']}°({s['n_traces']}条)"
            for s in r["detected_sets"])
        print(f"[outcrop_trace] {r['case']}: det={r['n_detected_traces']} "
              f"sets={len(r['detected_sets'])} [{sets_str}]")
    print(f"[outcrop_trace] -> {os.path.relpath(json_path, _PROJECT_ROOT)}")
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fractureflow.outcrop_trace",
        description="R106 · L2 露头迹线产线 单入口 CLI (产品化收口; 复用 R95 现役管线)")
    parser.add_argument("image_dir", help="影像目录（或单张影像路径）")
    parser.add_argument("--out-dir", default=os.path.join("results", "v_r106_out"),
                        help="输出目录 (默认 results/v_r106_out)")
    parser.add_argument("--scale-m-per-px", type=float, default=0.012,
                        help="长度尺度 m/px (占位值; 无比例尺标定时不承诺长度)")
    parser.add_argument("--min-seg-len-px", type=float, default=12.0,
                        help="最短迹线过滤像素 (R95 产品参数, 冻结)")
    args = parser.parse_args(argv)

    doc = process_directory(args.image_dir, args.out_dir,
                            scale_m_per_px=args.scale_m_per_px,
                            min_seg_len_px=args.min_seg_len_px)
    if doc["n_images"] == 0:
        print("[outcrop_trace] 无影像被处理", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
