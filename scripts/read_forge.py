# -*- coding: utf-8 -*-
"""读 FORGE 16A/16B 钻孔 FMI 裂隙测井 xlsx -> 勘察剖面 + route-A 就绪 net.

FORGE xlsx 只有单列共享字符串, 无 openpyxl 依赖, 用 zipfile 解析.
列: MD, TVD, Azimuth, Dip Direction, Dip, Type  (986 行裂隙, 单井).

法向由 (Dip Direction, Dip) 还原 (无向, ±n 等价):
    n = [ sin(dd)*sin(dip), cos(dd)*sin(dip), -cos(dip) ]

用法:
  python scripts/read_forge.py --xlsx data/external/FORGE_16A-32_FMI_fracture_log.xlsx
"""
import argparse
import json
import os
import sys
import zipfile
import re
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def _read_xlsx(path):
    z = zipfile.ZipFile(path)
    ss = [''.join(x.text or '' for x in si.iter(NS + 't'))
          for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(NS + 'si')]
    data = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))

    def colref(r):
        return re.match(r'[A-Z]+', r).group() if r else None

    rows = {}
    for c in data.iter(NS + 'c'):
        ref = c.get('r')
        if not ref:
            continue
        col = colref(ref)
        t = c.get('t')
        v = c.find(NS + 'v')
        isn = c.find(NS + 'is')
        if t == 's' and v is not None:
            val = ss[int(v.text)]
        elif isn is not None:
            val = ''.join(x.text or '' for x in isn.iter(NS + 't'))
        elif v is not None:
            val = v.text
        else:
            val = ''
        rows.setdefault(int(re.search(r'\d+', ref).group()), {})[col] = val
    # 列字母 -> 表头名
    hdr = rows.get(1, {})
    cols = sorted(hdr.keys())
    names = [hdr[c] for c in cols]
    body = []
    for r in range(2, max(rows) + 1):
        body.append([rows.get(r, {}).get(c, '') for c in cols])
    return names, np.array(body, dtype=object)


def _to_floats(body, names, name):
    j = names.index(name)
    out = np.empty(len(body), dtype=float)
    for i, x in enumerate(body[:, j]):
        try:
            out[i] = float(x)
        except (ValueError, TypeError):
            out[i] = np.nan
    return out


def azimuth_dipdir_to_normal(az, dipdir, dip):
    """(azimuth, dip_direction, dip) -> 无向法向 [N,3]. 仅 dip dir + dip 决定平面."""
    a = np.radians(dipdir)
    d = np.radians(dip)
    n = np.stack([np.sin(a) * np.sin(d), np.cos(a) * np.sin(d), -np.cos(d)], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx",
                    default=os.path.join(ROOT, "data/external/FORGE_16A-32_FMI_fracture_log.xlsx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/real/forge16A_net.pt"))
    args = ap.parse_args()

    names, body = _read_xlsx(args.xlsx)
    print("[forge] columns:", names, " rows:", len(body))

    md = _to_floats(body, names, "MD")
    az = _to_floats(body, names, "Azimuth")
    dd = _to_floats(body, names, "Dip Direction")
    dip = _to_floats(body, names, "Dip")
    from collections import Counter
    types = [str(r[names.index("Type")]) for r in body]
    type_counts = Counter(types)

    # Azimuth(走向) 在 982 行有值, Dip Direction 仅 51 行; 抽样显示
    # Dip Direction = Azimuth + 90 (走向+90=倾向). 故缺失倾向时由走向派生.
    dd_eff = np.where(np.isfinite(dd), dd, az + 90.0)
    valid = np.isfinite(az) & np.isfinite(dip)
    n_valid = int(valid.sum())
    print(f"[forge] 有效裂隙 (az+dip, 倾向缺失时由走向派生): {n_valid}/{len(body)}")

    nrm = azimuth_dipdir_to_normal(az, dd_eff, dip)

    def _nz(a):
        return int(np.isfinite(a).sum())

    azv, ddv, dipv, mdv = az[valid], dd_eff[valid], dip[valid], md[valid]
    recon = {
        "source": os.path.basename(args.xlsx),
        "n_fractures": int(len(body)),
        "n_valid": n_valid,
        "columns": names,
        "nonempty_counts": {
            "MD": _nz(md), "TVD": _nz(np.array(
                [r[names.index("TVD")] for r in body], dtype=float)),
            "Azimuth": _nz(az), "Dip Direction": _nz(dd), "Dip": _nz(dip),
            "Type": int(sum(1 for t in types if t)),
        },
        "md_range": [float(mdv.min()), float(mdv.max())],
        "azimuth_range": [float(azv.min()), float(azv.max())],
        "dip_direction_range": [float(ddv.min()), float(ddv.max())],
        "dip_range": [float(dipv.min()), float(dipv.max())],
        "dip_mean": float(dipv.mean()),
        "type_counts": dict(type_counts),
        "has_fracture_id": False,
        "has_xyz": False,
        "note": "单井 FMI 测井 (depth-only, 无 fracture_id/无 XYZ). "
                "可喂路线 A (法向→球面 kmeans 定组), 不可喂路线 B (无 fracture_id). "
                "单井=单一场地, 无跨场地泛化. 倾向缺失时由 Azimuth+90 派生.",
    }
    print(json.dumps(recon, ensure_ascii=False, indent=2, allow_nan=False))
    recon_path = os.path.join(ROOT, "results/forge_recon.json")
    with open(recon_path, "w", encoding="utf-8") as f:
        json.dump(recon, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[forge] recon -> {recon_path}")

    # route-A 就绪 net (单井 = 一个 net, 法向+深度属性)
    net = {
        "pos": np.zeros((n_valid, 3), float),   # 无 XYZ, 仅深度
        "md": md[valid].astype(float),
        "nrm": nrm[valid].copy(),
        "nrm_full": nrm[valid].copy(),
        "dip": dip[valid].astype(float),
        "dipdir": dd[valid].astype(float),
        "azimuth": az[valid].astype(float),
        "ftype": np.array([types[i] for i in range(len(types)) if valid[i]]),
        "len": np.ones(n_valid),
        "lith": np.zeros(n_valid, int),
        "s1": np.ones(n_valid), "s3": np.ones(n_valid),
        "wid": os.path.basename(args.xlsx), "src": "forge_fmi",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import torch
    torch.save([net], args.out)
    print(f"[forge] route-A 就绪 net -> {args.out}  (N={n_valid})")


if __name__ == "__main__":
    main()
