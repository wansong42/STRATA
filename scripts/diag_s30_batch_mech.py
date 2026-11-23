# -*- coding: utf-8 -*-
"""S30 FAIL 机制隔离: 共享init / 批级kn / 批级Kk 谁在起作用."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fractureflow.geo_prior import geo_prior_dirs_gpu


def make_net(seed, n=40):
    r = np.random.default_rng(seed)
    pos = r.normal(size=(n, 3))
    nrm = np.vstack([np.tile([1.0, 0, 0], (n // 2, 1)),
                     np.tile([0.0, 1, 0], (n - n // 2, 1))])
    nrm = nrm + r.normal(0, 0.05, nrm.shape)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    mask = r.random(n) < 0.5
    return pos.astype(np.float64), (nrm * mask[:, None]).astype(np.float64), mask


def ang(a, b):
    c = np.clip(np.abs((a * b).sum(-1)), 0, 1)
    return float(np.degrees(np.arccos(c)).max())


nets = [make_net(s) for s in (101, 202, 303, 404)]
for i, (_, _, m) in enumerate(nets):
    print(f"net{i}: L={len(m)}, n_occ={int(m.sum())}")

# B=1 各网单独
outs_b1 = []
for pos, nrm, mask in nets:
    o = geo_prior_dirs_gpu(torch.tensor(pos)[None], torch.tensor(nrm)[None],
                           torch.tensor(mask, dtype=torch.float64)[None], fast=False)
    outs_b1.append(o[0].numpy())

# B=4
pos4 = torch.tensor(np.stack([x[0] for x in nets]))
nrm4 = torch.tensor(np.stack([x[1] for x in nets]))
mask4 = torch.tensor(np.stack([x[2] for x in nets]), dtype=torch.float64)
out_b4 = geo_prior_dirs_gpu(pos4, nrm4, mask4, fast=False).numpy()

print("\n--- 全隐伏点最大角差 ---")
for b in range(4):
    hid = ~nets[b][2]
    print(f"net{b} @pos{b}: {ang(outs_b1[b][hid], out_b4[b][hid]):.3f}°")

# 隔离实验A: net0 放在不同批位置 (同批同伴固定) -> 测共享init效应
companions = [nets[1], nets[2], nets[3]]
print("\n--- 隔离A: net0 在批内位置 0..3 ---")
for slot in range(4):
    batch = companions[:slot] + [nets[0]] + companions[slot:]
    pb = torch.tensor(np.stack([x[0] for x in batch]))
    nb = torch.tensor(np.stack([x[1] for x in batch]))
    mb = torch.tensor(np.stack([x[2] for x in batch]), dtype=torch.float64)
    ob = geo_prior_dirs_gpu(pb, nb, mb, fast=False).numpy()[slot]
    hid = ~nets[0][2]
    print(f"slot={slot}: vs B1 {ang(outs_b1[0][hid], ob[hid]):.3f}°")

# 隔离实验B: 同一 net0 与不同同伴组批 -> 测批组成敏感度
print("\n--- 隔离B: net0 + 不同同伴 ---")
for tag, comp in [("B", [nets[1]]), ("C", [nets[2]]), ("D", [nets[3]])]:
    batch = [nets[0]] + comp
    pb = torch.tensor(np.stack([x[0] for x in batch]))
    nb = torch.tensor(np.stack([x[1] for x in batch]))
    mb = torch.tensor(np.stack([x[2] for x in batch]), dtype=torch.float64)
    ob = geo_prior_dirs_gpu(pb, nb, mb, fast=False).numpy()[0]
    hid = ~nets[0][2]
    print(f"{tag}: vs B1 {ang(outs_b1[0][hid], ob[hid]):.3f}°")
