# -*- coding: utf-8 -*-
"""E3GT v4 — 等变向量传播回归网络 (重构核心, 目标逼近 13° 天花板)。

设计动机
--------
前作 SetProp 的解码被"组中心锚点混合" (K 组中心 + 最近观测 + 弱残差) 限制,
理论上界 = 真值指派 + 观测组中心 ≈ 13.76°(理想) / 真实 18~23°, 实际上被卡在
~34°. 其 dir 损失 (1-|cos|) 长期停在 0.27 (cos≈0.73), 说明连续方向回归几乎没学出来.

v4 的关键重构: **去掉锚点混合, 改为连续等变向量回归**。
  - 观测法向作为"协变向量通道" v_i 进入网络 (隐伏点 v_i=0, 必须从邻居传播重建).
  - 在每点局部坐标系 (kNN-PCA 标架) 内做等变消息传递: 标量不变量 + 向量协变量
    一起传播; 输出在局部标架内回归 3D 方向, 再旋回全局.
  - 全程不依赖脆短的球面 k-means 组发现, 不必被组中心均值封顶 —— 可表达组内
    个体差异, 理论上下界低于组中心均值.

等变性
------
输入整体旋转 R: 标架 evec -> R·evec, 局部向量 v_i(局部) 不变, 全局向量 -> R·向量.
网络在局部标架内的运算 (点积/叉积/范数/MLP) 均为 SO(3) 等变/不变, 故输出方向
随输入同步旋转. 配合 |cos| 无向度量, 评测对旋转一致.

注: 不引入 e3nn 等外部库; 全部用纯 torch 实现 (符合项目约束).

R173 (2026-09-05, D-N1 闭环): EquivLayer.Wv 增加 wv_mode 三模式 ——
  frozen (默认) = 冻结行为逐位不变; ortho = Wv 极分解投影到最近 SO(3)
  (任务书原案); scalar = c·I Schur 等变形式 (R173 实证: 层级等变 ⇔ c·I,
  正交化不充分: polar 89.51° vs scalar 2e-6°)。管线内 v 为局部标架不变
  坐标, Wv 任何形态都不破坏该不变性; 端到端等变的真实阻断是 D-N2 协变
  特征与 D-N5 (e2 符号钉不稳健), 见 results/r173_wv_fix/verdict.md。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import knn_graph, local_frames, build_graph_features

EPS = 1e-8


def unit(x, eps=EPS):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _polar_so3(W):
    """W [3,3] -> 最近 SO(3) 旋转 (极分解; Frobenius 意义下正交化漂移最小)。

    R173 (D-N1): 任务书原案的正交化路径。注意 (R173 实证, Schur 引理):
    与 SO(3) 全体可交换的 3x3 实矩阵只有 c·I, 故正交化**不足以**恢复
    "帧固定/v 等变"构架下的向量通道等变 —— 等变形式是 scalar 模式 (c·I)。
    """
    U, _S, Vh = torch.linalg.svd(W)
    U = U.clone()
    if float(torch.det(U @ Vh)) < 0:
        U[:, -1] = -U[:, -1]
    return U @ Vh


class NormAct(nn.Module):
    """Vector-Neuron 风格向量激活: v -> vhat * rho(||v||), rho>0 保证等变。

    rho 是一个以 ||v|| 为输入的标量正函数 (softplus 输出), 使向量通道具备
    可学的幅度门控, 同时保持方向等变。
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, v):
        # v: [..., 3]
        nv = v.norm(dim=-1, keepdim=True)            # [..., 1]
        vhat = v / (nv + EPS)                        # [..., 3]
        scale = self.rho(nv)                         # [..., 1] > 0
        return vhat * scale


class EquivLayer(nn.Module):
    """单层局部标架等变消息传递。

    输入: s [B,L,d] 标量不变量, v [B,L,3] 局部标架内的协变向量。
    边: 用 kNN 相对位置 (旋入 i 的局部标架) 与不变量构造注意力 + 消息。
    输出: 更新后的 (s', v')。
    """

    def __init__(self, d, k, edge_dim=5, dropout=0.1, wv_mode="frozen"):
        super().__init__()
        self.k = k
        self.d = d
        if wv_mode not in ("frozen", "ortho", "scalar"):
            raise ValueError(
                f"未知 wv_mode {wv_mode!r} (仅接受 frozen/ortho/scalar)")
        self.wv_mode = wv_mode
        self.dropout = nn.Dropout(dropout)

        # 注意力: 输入 (s_i, s_j, edge) -> 标量分数
        self.attn = nn.Sequential(
            nn.Linear(2 * d + edge_dim, d), nn.ReLU(),
            nn.Linear(d, 1))

        # 标量消息 MLP: (s_j, edge) -> d
        self.msg_s = nn.Sequential(
            nn.Linear(d + edge_dim, d), nn.ReLU(),
            nn.Linear(d, d))

        # 向量线性变换: 3->3。⚠️ R173/D-N1: 作用在 3D 分量上的通用线性层
        # 不是 SO(3) 等变映射 (等变 ⇔ c·I, Schur); 等变语义由 wv_mode 控制,
        # 默认 frozen = 冻结行为逐位不变 (ckpt 兼容), 见 forward。
        self.Wv = nn.Linear(3, 3, bias=False)
        self.normact = NormAct()

        # 节点更新 (残差): (s_i, m_s) -> d
        self.upd_s = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(),
            nn.Linear(d, d))

    def forward(self, s, v, idx, dist, rel_local, evec, radius):
        B, L, k = idx.shape
        # 邻居 gather
        s_nb = torch.gather(s.unsqueeze(1).expand(B, L, L, self.d), 2,
                            idx.unsqueeze(-1).expand(B, L, k, self.d))      # [B,L,k,d]
        v_i = v.unsqueeze(2)                                               # [B,L,1,3]
        v_nb = torch.gather(v.unsqueeze(1).expand(B, L, L, 3), 2,
                            idx.unsqueeze(-1).expand(B, L, k, 3))          # [B,L,k,3]

        # ---- 不变量边特征 ----
        dist_n = (dist / radius.unsqueeze(-1)).unsqueeze(-1)               # [B,L,k,1]
        nv = v.norm(dim=-1, keepdim=True).unsqueeze(2)                     # [B,L,1,1]
        nv_nb = v_nb.norm(dim=-1, keepdim=True)                            # [B,L,k,1]
        dot_vv = (v_i * v_nb).sum(-1, keepdim=True)                        # [B,L,k,1]
        rel_norm = rel_local.norm(dim=-1, keepdim=True)                    # [B,L,k,1]
        rel_dot_v = (rel_local * v_nb).sum(-1, keepdim=True)               # [B,L,k,1]
        edge = torch.cat([dist_n, nv_nb, dot_vv, rel_norm, rel_dot_v], -1)  # [B,L,k,5]

        # ---- 注意力 (不变量) ----
        s_i_exp = s.unsqueeze(2).expand(B, L, k, self.d)                   # [B,L,k,d]
        attn_in = torch.cat([s_i_exp, s_nb, edge], -1)                     # [B,L,k,2d+5]
        a = F.softmax(self.attn(attn_in), dim=2)                           # [B,L,k,1]

        # ---- 标量消息 ----
        msg_s_in = torch.cat([s_nb, edge], -1)                             # [B,L,k,d+5]
        m_s = (a * self.msg_s(msg_s_in)).sum(2)                            # [B,L,d]

        # ---- 向量消息 (协变): 邻向量 + 叉积 (均在 i 局部标架) ----
        cross = torch.cross(v_i.expand_as(v_nb), v_nb, dim=-1)             # [B,L,k,3]
        m_v = (a * (v_nb + cross)).sum(2)                                  # [B,L,3]

        # ---- 更新 (R173/D-N1 三模式; 默认 frozen 与冻结 ckpt 逐位一致) ----
        # 管线中 v 为局部标架坐标 (场景旋转下不变), 任何固定 Wv 都不破坏该
        # 不变性; wv_mode 只影响"帧固定/v 等变"构架下的声明 (层级测试口径)。
        if self.wv_mode == "frozen":
            v = self.normact(self.Wv(v) + m_v)                             # [B,L,3]
        elif self.wv_mode == "ortho":
            # 任务书原案: Wv -> 最近 SO(3) (R173 实证: 不能恢复层级等变, 仅登记)
            v = self.normact(F.linear(v, _polar_so3(self.Wv.weight)) + m_v)
        else:  # scalar: Schur 等变形式 c·I (c 由原参数 Frobenius 范数导出)
            c = torch.linalg.norm(self.Wv.weight) / (3.0 ** 0.5)
            v = self.normact(c * v + m_v)
        s = s + self.dropout(self.upd_s(torch.cat([s, m_s], -1)))          # [B,L,d]
        return s, v

    def set_wv_mode(self, mode):
        """运行时切换 wv_mode (不改变参数; 供诊断/评测显式选择)。"""
        if mode not in ("frozen", "ortho", "scalar"):
            raise ValueError(
                f"未知 wv_mode {mode!r} (仅接受 frozen/ortho/scalar)")
        self.wv_mode = mode


class E3GTV4(nn.Module):
    def __init__(self, d_model=256, k_knn=16, n_layers=4, dropout=0.1,
                 obs_drop=0.25, wv_mode="frozen"):
        super().__init__()
        self.k = k_knn
        self.d = d_model
        self.obs_drop = obs_drop  # 训练时对观测法向的随机丢弃比例 (迫使传播)

        # 节点标量提升: build_graph_features 产出 31 维不变量
        self.lift = nn.Sequential(
            nn.Linear(31, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))

        self.layers = nn.ModuleList(
            [EquivLayer(d_model, k_knn, dropout=dropout, wv_mode=wv_mode)
             for _ in range(n_layers)])

        # 读头: (s, |v|, v) -> 局部标架内 3D 方向
        self.readout = nn.Sequential(
            nn.Linear(d_model + 1 + 3, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3))

        # 辅助头: 预测"所在裂隙组"的方向 (set_dirs[set_id])。
        # 用 3D 方向而非组索引作监督 -> 跨网络语义一致, 可直接迁移到真实数据
        # (真实数据无 set_ids, 该头仅在合成预训练阶段激活, 塑造 s 的"组身份"表征)。
        self.set_dir_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 3))

    def _dropped_nrm(self, nrm, mask, rng=None):
        """训练时随机丢弃部分观测法向 (输入置零, 目标仍用 nrm_full 监督)。"""
        if self.training and self.obs_drop > 0:
            drop = (torch.rand_like(mask) < self.obs_drop) & (mask > 0.5)
            return torch.where(drop.unsqueeze(-1), torch.zeros_like(nrm), nrm)
        return nrm

    def forward(self, b):
        pos = b["pos"]; nrm = b["nrm"]; mask = b["mask"]
        log_len = b["log_len"]; lith = b["lith"]; s1 = b["s1"]; s3 = b["s3"]
        B, L, _ = pos.shape

        nrm_in = self._dropped_nrm(nrm, mask)

        # 图 + 局部标架 (一次性计算, 供消息传递与节点特征共用, 避免重复 kNN/PCA)
        idx, dist = knn_graph(pos, self.k)
        idx3 = idx.unsqueeze(-1).expand(B, L, self.k, 3)
        knn_pos = torch.gather(pos.unsqueeze(1).expand(B, L, L, 3), 2, idx3)
        evec, eval_ratio = local_frames(knn_pos, self.k)            # [B,L,3,3]
        graph = (idx, dist, knn_pos, evec, eval_ratio)
        fnode, fedge, _, _ = build_graph_features(
            pos, nrm_in, log_len, lith, mask, s1, s3, self.k,
            graph=graph)                                           # [B,L,31]
        radius = pos.std(1).mean(-1, keepdim=True).clamp_min(1e-3)  # [B,1]

        # 协变向量通道: 观测法向旋入局部标架; 隐伏点=0
        nrm_u = unit(torch.where(nrm_in.norm(dim=-1, keepdim=True) > 1e-6,
                                 nrm_in, torch.zeros_like(nrm_in)))
        has_obs = (nrm_in.norm(dim=-1, keepdim=True) > 1e-6).float()
        v0 = torch.einsum("blc,blcd->bld", nrm_u, evec) * has_obs  # [B,L,3]

        s = self.lift(fnode)                                       # [B,L,d]
        v = v0                                                     # [B,L,3]

        rel = knn_pos - pos.unsqueeze(2)                           # [B,L,k,3]
        rel_local = torch.einsum("blke,blcd->blkd", rel, evec)     # [B,L,k,3]

        for layer in self.layers:
            s, v = layer(s, v, idx, dist, rel_local, evec, radius)

        # 读头: 局部标架内回归方向
        nv = v.norm(dim=-1, keepdim=True)                          # [B,L,1]
        n_local = self.readout(torch.cat([s, nv, v], -1))         # [B,L,3]
        # 旋回全局: global = evec @ local  (evec 列向量 = 局部基, 故 E @ n_local)
        n_global = torch.einsum("blcd,bld->blc", evec, n_local)    # [B,L,3]
        pred = unit(n_global)

        # 位置邻域各向异性 (PCA 特征值比): 局部几何平面度, 越接近 1 越可预测
        aniso = (eval_ratio[:, :, 0] - eval_ratio[:, :, 2]).clamp(0.0, 1.0)  # [B,L]

        # 辅助: 预测所在裂隙组方向 (仅合成数据有 set_ids/set_dirs)
        set_dir_pred = unit(self.set_dir_head(torch.cat([s, v], -1)))        # [B,L,3]

        return {"pred": pred, "v": v, "s": s, "aniso": aniso, "set_dir_pred": set_dir_pred}

    # ---------------- 损失 ----------------
    def loss(self, b, w_hid=1.0, w_obs=0.3, w_aniso=1.0, aniso_floor=0.2,
             w_set=0.5):
        out = self.forward(b)
        pred = out["pred"]
        mask = b["mask"]
        nrm_full = b["nrm_full"]
        aniso = out["aniso"]
        hid = (1.0 - mask) > 0.5
        obs = mask > 0.5

        cos = (pred * nrm_full).sum(-1).abs().clamp(0, 1)         # [B,L]
        err1 = 1.0 - cos

        total = 0.0
        comp = {}
        # 隐伏点方向损失 (主目标, 按各向异性加权聚焦可预测点)
        if hid.any():
            aw = torch.clamp(aniso / 0.5, aniso_floor, 1.0) * w_aniso
            l_hid = (err1[hid] * aw[hid]).mean() / max(float(aw[hid].mean()), 1e-3)
            total = total + w_hid * l_hid
            comp["l_hid"] = float(err1[hid].mean())
        # 观测点重构 (辅助, 含去噪): 让模型学会保留/重建观测法向
        if obs.any():
            l_obs = err1[obs].mean()
            total = total + w_obs * l_obs
            comp["l_obs"] = float(l_obs)

        # 辅助: 合成数据有 set_ids/set_dirs 时, 监督"组方向"预测
        # (攻击指派瓶颈: 让 s 学会表达裂隙组身份, 迁移到真实数据)
        if "set_ids" in b and "set_dirs" in b:
            Bb, Lb, _ = b["pos"].shape
            set_ids = b["set_ids"]                                        # [B,L] long
            set_dirs = b["set_dirs"]                                    # [B,G,3]
            valid = set_ids >= 0
            sid = set_ids.clamp(0, set_dirs.shape[1] - 1)
            idx = sid.unsqueeze(-1).expand(Bb, Lb, 3).to(torch.long)
            tgt = torch.gather(set_dirs, 1, idx)                        # [B,L,3]
            cos_s = (out["set_dir_pred"] * tgt).sum(-1).abs().clamp(0, 1)
            err_s = 1.0 - cos_s
            if valid.any():
                l_set = err_s[valid].mean()
                total = total + w_set * l_set
                comp["l_set"] = float(l_set)

        comp["total"] = float(total)
        return total, comp
