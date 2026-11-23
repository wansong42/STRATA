# -*- coding: utf-8 -*-
"""S0 · 统一观测核（唯一新核心观测模块）。

本件把 results/research_direction_20261018/研究方向裁定与核心技术路线_v2.0.md
§5（统一观测制度）落成可执行的最小观测核: 有限井段曝光 → 声明离散深度/产状
误差 → 量化到声明格 → 匿名计数向量公开记录。判据权威 = 本线
results/s0s1_minimal_vertical_chain/s0/observation_spec.json（冻结后其
canonical sha12 与代码钢印共同构成 kernel_id）。

硬语义（与任务单 §3.1 逐条对应，违反即实现缺陷）:
  1. 状态 = 完整候选字典每个盘的存在性位; 本件不读取真值、不读取 frac_id。
  2. 行动 = 有限井段 (entry, direction, length); 曝光段显式裁剪到声明域与 [0,length],
     禁止"无限长线 + 负深度过滤"旧语义。
  3. 曝光: 交点参数 t∈[0,L_eff] 且落入盘内才算曝光; t<0 / t>L_eff 一律"未曝光",
     未曝光盘对本次行动检测概率为 0（只贡献 ∅ 分支），不得当作负证据。
  4. 误差与量化顺序: 真交点 → 声明测量误差（深度格点误差 + 无向法向切平面旋转误差）
     → 量化到声明格 → 释放。禁止先量化再加噪，禁止记录真法向。
  5. 公开记录 = 匿名计数向量（深度格 × 产状格的计数）+ design_id/well_id/
     exposure_segment/quantization/detected; 不含 frac_id/真值集合/未曝光盘心/
     隐藏标识/真位置。原始逐盘记录只留在评估侧（private audit）。
  6. 检测模型: 每盘以声明概率落格或 ∅; r_iab/r_ia∅ 由声明离散误差表精确给出
     （Fraction 精确算术）; 给定状态下各盘检测条件独立; shared_bias=false; λ=0。
  7. 误检: 首版 λ=0（§3.1-7 允许）; 若 λ>0 须带显式 overflow 类（本件构造期硬拒）。
  8. 似然: log_prob 走精确计数向量 PMF（生成函数乘积 / 稀疏多重卷积 / 系数抽取），
     禁止把"同一盘只进一个格"偷换成"各格独立 Bernoulli"; count_pmf() 在声明枚举
     预算内显式物化 PMF 并对 sum(pmf)=1 做精确机器检查; log_prob_upper 首版返回
     UNSUPPORTED（不得把优化器当前最大值冒充上界）。
  9. kernel_id = 规范化 observation_spec（去掉 kernel_identity 键）的 sha12 +
     observation_kernel.py 代码字节 sha12; 任何格界/误差表/量化顺序变化必须换
     kernel_id 并重开 prereg, 旧读数不沿用（STALE_KERNEL）。

允许的实现边界（§3.1 末条）: 首版只支持 d<=6 的完整枚举; 全 PMF 物化受
pmf_max_entries 限制, 超出即 ComputationUnfinished（COMPUTATION_UNFINISHED），
不猜、不删、不截断。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from fractions import Fraction

import numpy as np

__all__ = [
    "KernelIdentityError", "KernelScopeError", "PublicRecordViolation",
    "ComputationUnfinished", "UNSUPPORTED", "EMPTY_OUTCOME",
    "GRID_KERNEL_FAMILY", "GridDetectionKernel", "ObservationKernel",
    "canonical_bytes", "canonical_sha12", "kernel_identity", "sha12_file",
    "sha12_bytes", "FORBIDDEN_PUBLIC_KEYS",
]

EMPTY_OUTCOME = "EMPTY"
GRID_KERNEL_FAMILY = "GRID_DETECTION_KERNEL"
UNSUPPORTED = "UNSUPPORTED"
ALLOWED_RECORD_KEYS = {
    "design_id", "well_id", "exposure_segment", "quantization",
    "counts", "detected_total", "detected", "kernel_id",
}
FORBIDDEN_PUBLIC_KEYS = {
    "frac_id", "frac_ids", "truth", "truth_sets", "truth_set", "true_state",
    "state_bits", "candidate_id", "candidate_index", "disc_id", "disc_index",
    "center", "centers", "normal", "normals", "radius", "radii",
    "unexposed_centers", "hidden_id", "truth_label", "answer", "g_q",
}
_JSON_SCALARS = (str, int, float, bool, type(None))


class KernelIdentityError(RuntimeError):
    """spec 冻结内容 / 代码钢印与 kernel_id 登记不符（STALE_KERNEL）。"""


class KernelScopeError(ValueError):
    """模型/行动超出声明核作用域（validate_scope 判红）。"""


class PublicRecordViolation(ValueError):
    """公开记录含禁键 / 非 JSON 原生对象 / 结构非法。"""


class ComputationUnfinished(RuntimeError):
    """精确 PMF 枚举超出声明预算（返回该状态，不猜、不删、不截断）。"""


def sha12_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha12_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha12_bytes(f.read())


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def canonical_sha12(obj) -> str:
    return sha12_bytes(canonical_bytes(obj))


def _code_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return sha256_bytes(f.read())


def kernel_identity(spec_base: dict) -> dict:
    """由 spec 内容（不含 kernel_identity 键）与代码字节计算 kernel 身份。"""
    if "kernel_identity" in spec_base:
        raise KernelIdentityError("kernel_identity(spec_base) 不得预含 kernel_identity 键")
    spec_sha12 = canonical_sha12(spec_base)
    spec_sha256 = hashlib.sha256(canonical_bytes(spec_base)).hexdigest()
    code_sha256 = _code_sha256()
    return {
        "formula": "kernel_id = 'K-' + sha12(canonical_json(spec_minus_kernel_identity)) + '-' + sha12(observation_kernel.py bytes)",
        "spec_content_sha12": spec_sha12,
        "spec_content_sha256": spec_sha256,
        "code_sha12": code_sha256[:12],
        "code_sha256": code_sha256,
        "code_file": "src/fractureflow/observation_kernel.py",
        "kernel_id": "K-%s-%s" % (spec_sha12, code_sha256[:12]),
    }


def _as_vec3(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,):
        raise KernelScopeError("%s 形状非法: %r (应为 (3,))" % (name, np.shape(x)))
    if not np.all(np.isfinite(a)):
        raise KernelScopeError("%s 含 NaN/inf" % name)
    return a


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not (n > 0.0 and math.isfinite(n)):
        raise KernelScopeError("%s 零向量/非有限" % name)
    return v / n


def _canonical_undirected_normal(n) -> np.ndarray:
    """无向法向规范化: n ~ -n；先验符号规则 (z, y, x) 依次取正。"""
    v = _unit(_as_vec3(n, "normal"), "normal")
    eps = 1e-12
    if v[2] < -eps:
        return -v
    if abs(v[2]) <= eps:
        if v[1] < -eps:
            return -v
        if abs(v[1]) <= eps and v[0] < 0.0:
            return -v
    return v


def _tangent_basis(n: np.ndarray):
    idx = int(np.argmin(np.abs(n)))
    axis = np.zeros(3, dtype=np.float64)
    axis[idx] = 1.0
    u1 = _unit(np.cross(n, axis), "tangent_u1")
    u2 = _unit(np.cross(n, u1), "tangent_u2")
    return u1, u2


def _orientation_params(n) -> dict:
    m = _canonical_undirected_normal(n)
    dec = math.degrees(math.acos(max(-1.0, min(1.0, float(m[2])))))
    az = math.degrees(math.atan2(float(m[1]), float(m[0]))) % 360.0
    return {"normal": m, "declination_deg": dec, "azimuth_deg": az}


def _direction_from_orientation(declination_deg: float, azimuth_deg: float) -> np.ndarray:
    d = math.radians(declination_deg)
    a = math.radians(azimuth_deg)
    return np.array([math.sin(d) * math.cos(a),
                     math.sin(d) * math.sin(a),
                     math.cos(d)], dtype=np.float64)


def _rotate_normal(n: np.ndarray, delta_deg: float, psi_deg: float) -> np.ndarray:
    """把无向法向 n 在切平面方位 psi 上旋转 polar 角 delta（先噪声后量化）。"""
    u1, u2 = _tangent_basis(n)
    delta = math.radians(delta_deg)
    psi = math.radians(psi_deg)
    w = math.sin(psi) * u1 + math.cos(psi) * u2
    m = math.cos(delta) * n + math.sin(delta) * w
    return _unit(m, "noisy_normal")


def _line_plane_intersection(entry, direction, center, normal):
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-14:
        return None
    return -float(np.dot(entry - center, normal)) / denom


def _clip_segment(entry, direction, length, lo, hi) -> dict:
    t0, t1 = 0.0, float(length)
    for k in range(3):
        p, d = float(entry[k]), float(direction[k])
        if abs(d) < 1e-15:
            if p < lo[k] - 1e-12 or p > hi[k] + 1e-12:
                return {"ok": False, "reason": "ENTRY_OUTSIDE_DOMAIN_AXIS_%d" % k}
            continue
        ta = (lo[k] - p) / d
        tb = (hi[k] - p) / d
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
    if t1 <= t0:
        return {"ok": False, "reason": "SEGMENT_OUTSIDE_DOMAIN"}
    return {"ok": True, "t0": t0, "t1": t1}


def _segment_point(entry, direction, t):
    return entry + t * direction


class ObservationKernel:
    """v2.0 §5.1 拟定最小接口的可执行基类（签名即契约）。"""

    kernel_id = "UNSET"

    def simulate(self, state, action, history=None, rng=None):
        raise NotImplementedError

    def log_prob(self, public_record, state, action, history=None):
        raise NotImplementedError

    def log_prob_upper(self, public_record, region, action, history=None):
        raise NotImplementedError

    def validate_scope(self, model_spec, action_spec):
        raise NotImplementedError


class GridDetectionKernel(ObservationKernel):
    """声明深度格 × 无向产状格的有限分辨率几何记录核（KNOWN_SYNTHETIC）。"""

    def __init__(self, model_spec: dict, spec: dict):
        self.spec = spec
        self.model_spec = model_spec
        if spec.get("kernel_family") != GRID_KERNEL_FAMILY:
            raise KernelIdentityError("spec.kernel_family=%r 非 %s"
                                      % (spec.get("kernel_family"), GRID_KERNEL_FAMILY))
        self._verify_kernel_identity(spec)
        ki = spec["kernel_identity"]
        self.kernel_id = ki["kernel_id"]
        self.spec_content_sha12 = ki["spec_content_sha12"]
        self.code_sha12 = ki["code_sha12"]

        self.domain_lo = np.asarray(spec["domain"]["lo"], dtype=np.float64)
        self.domain_hi = np.asarray(spec["domain"]["hi"], dtype=np.float64)
        self.depth_bin_width = float(spec["depth_quantization"]["bin_width"])
        op = spec["orientation_partition"]
        self.azimuth_sectors = int(op["azimuth_sectors"])
        self.declination_bands = int(op["declination_bands"])
        self.azimuth_edges = [float(x) for x in op["azimuth_edges_deg"]]
        self.declination_edges = [float(x) for x in op["declination_edges_deg"]]
        self.n_orientation_cells = self.azimuth_sectors * self.declination_bands
        self.noise_tiers = dict(spec["noise_model"]["noise_tiers"])
        self._depth_tables = self._parse_noise_tables()
        self._pmf_max_entries = int(spec["pmf"]["pmf_max_entries"])
        self._max_d = int(spec["computational_limits"]["max_enumeration_d"])
        self._discs = self._parse_model(model_spec)
        self._outcome_cache = {}
        self._measurement_cache = {}

    @staticmethod
    def _verify_kernel_identity(spec: dict) -> None:
        ki = spec.get("kernel_identity")
        if not isinstance(ki, dict):
            raise KernelIdentityError("spec 缺 kernel_identity 冻结块")
        spec_base = {k: v for k, v in spec.items() if k != "kernel_identity"}
        want = canonical_sha12(spec_base)
        if ki.get("spec_content_sha12") != want:
            raise KernelIdentityError(
                "spec 内容 sha12 与 kernel_identity 登记不符 (STALE_KERNEL): %s vs %s"
                % (want, ki.get("spec_content_sha12")))
        code_sha256 = _code_sha256()
        if ki.get("code_sha256") != code_sha256:
            raise KernelIdentityError(
                "observation_kernel.py 代码字节与 kernel_identity 登记不符 (STALE_KERNEL): %s vs %s"
                % (code_sha256[:12], str(ki.get("code_sha256"))[:12]))
        want_id = "K-%s-%s" % (want, code_sha256[:12])
        if ki.get("kernel_id") != want_id:
            raise KernelIdentityError("kernel_id 复核不符: %r vs %r"
                                      % (ki.get("kernel_id"), want_id))

    def _parse_model(self, model_spec: dict) -> list:
        if not isinstance(model_spec, dict):
            raise KernelScopeError("model_spec 必须为 dict")
        dom = model_spec.get("domain")
        if dom is not None:
            lo = np.asarray(dom["lo"], dtype=np.float64)
            hi = np.asarray(dom["hi"], dtype=np.float64)
            if not (np.allclose(lo, self.domain_lo) and np.allclose(hi, self.domain_hi)):
                raise KernelScopeError("model_spec.domain 与观测规格声明域不一致")
        discs = model_spec.get("discs")
        if not isinstance(discs, list) or not discs:
            raise KernelScopeError("model_spec.discs 必须为非空列表")
        if len(discs) > self._max_d:
            raise KernelScopeError("候选字典 d=%d 超出首版完整枚举上限 %d"
                                   % (len(discs), self._max_d))
        out = []
        for i, d in enumerate(discs):
            c = _as_vec3(d["center"], "discs[%d].center" % i)
            n = _unit(_as_vec3(d["normal"], "discs[%d].normal" % i), "discs[%d].normal" % i)
            r = float(d["radius"])
            if not (r > 0.0 and math.isfinite(r)):
                raise KernelScopeError("discs[%d].radius 非正/非有限: %r" % (i, r))
            out.append({"index": i, "center": c, "normal": n, "radius": r})
        return out

    def _parse_noise_tables(self) -> dict:
        tables = {}
        for tier in ("N0", "N1", "N2"):
            t = self.noise_tiers.get(tier)
            if not isinstance(t, dict):
                raise KernelIdentityError("noise_tiers 缺 %s" % tier)
            depth_w = {int(k): int(v) for k, v in t["depth"]["weights"].items()}
            dden = int(t["depth"]["denominator"])
            if sum(depth_w.values()) != dden:
                raise KernelIdentityError("%s depth 权重和 != 分母" % tier)
            pol = t["angle"]["polar_offsets_deg"]
            pw = [int(x) for x in t["angle"]["polar_weights"]]
            pden = int(t["angle"]["polar_denominator"])
            if len(pol) != len(pw) or sum(pw) != pden:
                raise KernelIdentityError("%s angle 极角表非法" % tier)
            j = int(t["angle"]["azimuth_count"])
            if j <= 0:
                raise KernelIdentityError("%s azimuth_count 非正" % tier)
            det = t["detection"]
            det_fr = Fraction(int(det["detection_probability_numerator"]),
                              int(det["detection_probability_denominator"]))
            if not (0 <= det_fr <= 1):
                raise KernelIdentityError("%s detection_probability 越界" % tier)
            lam = Fraction(int(det["false_detection"]["lambda_numerator"]),
                           int(det["false_detection"]["lambda_denominator"]))
            if lam != 0:
                raise KernelIdentityError(
                    "%s 声明 λ=%s>0: 首版只支持 λ=0（显式 overflow 类未实现）" % (tier, lam))
            tables[tier] = {
                "depth_weights": depth_w, "depth_denominator": dden,
                "polar_offsets_deg": [float(x) for x in pol], "polar_weights": pw,
                "polar_denominator": pden, "azimuth_count": j,
                "detection_probability": det_fr, "lambda": lam,
                "quantization_still_applied": bool(t["quantization_still_applied"]),
            }
        return tables

    def _parse_action(self, action_spec: dict) -> dict:
        if not isinstance(action_spec, dict):
            raise KernelScopeError("action_spec 必须为 dict")
        entry = _as_vec3(action_spec["entry"], "action.entry")
        direction = _unit(_as_vec3(action_spec["direction"], "action.direction"),
                          "action.direction")
        length = float(action_spec["length"])
        if not (length > 0.0 and math.isfinite(length)):
            raise KernelScopeError("action.length 非正/非有限: %r" % length)
        clip = _clip_segment(entry, direction, length, self.domain_lo, self.domain_hi)
        if not clip["ok"]:
            raise KernelScopeError("曝光段裁剪失败: %s" % clip["reason"])
        t0, t1 = float(clip["t0"]), float(clip["t1"])
        if t0 > 1e-12:
            raise KernelScopeError(
                "action 起点在声明域外/曝光段被裁剪掉前段 (t0=%r): 首版要求 entry 在域内" % t0)
        t1 = min(t1, length)
        L_eff = t1 - t0
        n_bins_f = L_eff / self.depth_bin_width
        n_bins = int(round(n_bins_f))
        if abs(n_bins_f - n_bins) > 1e-9 or n_bins < 1:
            raise KernelScopeError(
                "曝光段长 %r 不是深度格宽 %r 的整数倍 (n_bins=%r)"
                % (L_eff, self.depth_bin_width, n_bins_f))
        return {"action_id": str(action_spec.get("action_id", "A")),
                "well_id": int(action_spec["well_id"]),
                "entry": entry, "direction": direction,
                "length_declared": length, "exposure_segment": [t0, t1],
                "length_exposed": L_eff, "n_depth_bins": n_bins}

    def _cell_of(self, depth: float, normal, n_depth_bins: int) -> int:
        w = self.depth_bin_width
        z = min(max(float(depth), 0.0), n_depth_bins * w)
        db = int(math.floor(z / w))
        if db >= n_depth_bins:
            db = n_depth_bins - 1
        if db < 0:
            db = 0
        op = _orientation_params(normal)
        az, dec = op["azimuth_deg"], op["declination_deg"]
        az_idx = int(math.floor(az / (360.0 / self.azimuth_sectors)))
        if az_idx >= self.azimuth_sectors:
            az_idx = self.azimuth_sectors - 1
        dec_idx = int(math.floor(dec / (90.0 / self.declination_bands)))
        if dec_idx >= self.declination_bands:
            dec_idx = self.declination_bands - 1
        orient = az_idx * self.declination_bands + dec_idx
        return db * self.n_orientation_cells + orient

    def _exposure_of(self, disc: dict, action: dict) -> dict:
        t = _line_plane_intersection(action["entry"], action["direction"],
                                     disc["center"], disc["normal"])
        if t is None:
            return {"exposed": False, "reason": "PARALLEL_TO_DISC_PLANE", "t": None}
        L = action["length_exposed"]
        if t < 0.0 or t > L:
            return {"exposed": False, "reason": "OUTSIDE_FINITE_SEGMENT", "t": t}
        q = _segment_point(action["entry"], action["direction"], t)
        if not bool(np.linalg.norm(q - disc["center"]) <= disc["radius"] + 1e-12):
            return {"exposed": False, "reason": "SEGMENT_MISSES_DISC", "t": t}
        return {"exposed": True, "reason": "EXPOSED", "t": float(t)}

    def _raw_outcomes(self, action: dict, disc: dict, tier: str) -> list:
        """[(depth_offset:int|None, polar_idx, azimuth_idx, detected:bool,
             cell:int|None, probability:Fraction, polar_delta_deg, psi_deg), ...]"""
        key = (tier, action["action_id"], action["well_id"], disc["index"],
               tuple(np.round(action["entry"], 12)),
               tuple(np.round(action["direction"], 12)), action["length_exposed"])
        if key in self._measurement_cache:
            return self._measurement_cache[key]
        tb = self._depth_tables[tier]
        expo = self._exposure_of(disc, action)
        if not expo["exposed"]:
            out = [(None, None, None, False, None, Fraction(1, 1), None, None)]
            self._measurement_cache[key] = out
            return out
        t_true = expo["t"]
        n_true = disc["normal"]
        det = tb["detection_probability"]
        miss = 1 - det
        J = tb["azimuth_count"]
        pden = tb["polar_denominator"] * J
        out = []
        if miss > 0:
            out.append((None, None, None, False, None, miss, None, None))
        w = self.depth_bin_width
        for l, wl in sorted(tb["depth_weights"].items()):
            t_meas = t_true + l * w
            for k, delta in enumerate(tb["polar_offsets_deg"]):
                wk = tb["polar_weights"][k]
                for j in range(J):
                    psi = 360.0 * j / J
                    m = _rotate_normal(n_true, delta, psi)
                    cell = self._cell_of(t_meas, m, action["n_depth_bins"])
                    p = det * Fraction(wl, tb["depth_denominator"]) * Fraction(wk, pden)
                    out.append((l, k, j, True, cell, p, float(delta), float(psi)))
        s = sum(item[5] for item in out)
        if s != 1:
            raise KernelIdentityError("tier=%s 逐盘结局概率和 != 1: %s" % (tier, s))
        self._measurement_cache[key] = out
        return out

    def _outcome_table(self, action: dict, disc: dict, tier: str) -> dict:
        key = (tier, action["action_id"], action["well_id"], disc["index"],
               action["length_exposed"])
        if key in self._outcome_cache:
            return self._outcome_cache[key]
        probs = {}
        for item in self._raw_outcomes(action, disc, tier):
            detected, cell = item[3], item[4]
            o = EMPTY_OUTCOME if not detected else int(cell)
            probs[o] = probs.get(o, Fraction(0, 1)) + item[5]
        if sum(probs.values()) != 1:
            raise KernelIdentityError("tier=%s 聚合分布和 != 1" % tier)
        self._outcome_cache[key] = probs
        return probs

    def public_record_schema(self) -> dict:
        return {"allowed_keys": sorted(ALLOWED_RECORD_KEYS),
                "forbidden_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
                "counts_encoding": "ascending [[cell_id:int, count:int], ...]",
                "json_only": True}

    @staticmethod
    def _assert_json_native(obj, path: str = "$") -> None:
        if isinstance(obj, _JSON_SCALARS):
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise PublicRecordViolation("%s 含非字符串键 %r（真实对象引用）"
                                                % (path, k))
                if k in FORBIDDEN_PUBLIC_KEYS:
                    raise PublicRecordViolation("%s 含公开记录禁键 %r（真值/隐藏信息泄漏）"
                                                % (path, k))
                GridDetectionKernel._assert_json_native(v, path + "." + k)
            return
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                GridDetectionKernel._assert_json_native(v, "%s[%d]" % (path, i))
            return
        raise PublicRecordViolation(
            "%s 含非 JSON 原生对象 %s（隐藏世界对象引用/序列化边界被绕过）"
            % (path, type(obj).__name__))

    def make_public_record(self, action: dict, counts: dict) -> dict:
        cl = [[int(c), int(n)] for c, n in sorted(counts.items()) if n]
        rec = {
            "design_id": action["action_id"],
            "well_id": int(action["well_id"]),
            "exposure_segment": [round(float(action["exposure_segment"][0]), 12),
                                 round(float(action["exposure_segment"][1]), 12)],
            "quantization": {"depth_bin_width": self.depth_bin_width,
                             "orientation_partition": self.spec["orientation_partition"]["algorithm"],
                             "orientation_cells": self.n_orientation_cells},
            "counts": cl,
            "detected_total": int(sum(n for _, n in cl)),
            "detected": bool(cl),
            "kernel_id": self.kernel_id,
        }
        self._assert_json_native(rec)
        return rec

    def _validate_public_record(self, public_record: dict, action: dict) -> dict:
        if not isinstance(public_record, dict):
            raise PublicRecordViolation("公开记录必须为 dict")
        self._assert_json_native(public_record)
        extra = set(public_record) - ALLOWED_RECORD_KEYS
        if extra:
            raise PublicRecordViolation("公开记录含未声明键: %s" % sorted(extra))
        if public_record.get("kernel_id") != self.kernel_id:
            raise PublicRecordViolation("公开记录 kernel_id 与现读数不符 (STALE_KERNEL)")
        if int(public_record.get("well_id", -1)) != int(action["well_id"]):
            raise PublicRecordViolation("公开记录 well_id 与行动不符")
        if public_record.get("design_id") != action["action_id"]:
            raise PublicRecordViolation("公开记录 design_id 与行动不符")
        counts_raw = public_record.get("counts", [])
        counts = {}
        for item in counts_raw:
            if not (isinstance(item, list) and len(item) == 2):
                raise PublicRecordViolation("counts 项必须为 [cell_id,count]")
            c, n = item
            if not (isinstance(c, int) and isinstance(n, int) and n > 0):
                raise PublicRecordViolation("counts 项非法: %r" % (item,))
            if c < 0 or c >= action["n_depth_bins"] * self.n_orientation_cells:
                raise PublicRecordViolation("cell_id 越界: %d" % c)
            if c in counts:
                raise PublicRecordViolation("counts 含重复 cell_id: %d" % c)
            counts[c] = n
        if int(public_record.get("detected_total", -1)) != sum(counts.values()):
            raise PublicRecordViolation("detected_total 与 counts 和不等")
        if bool(public_record.get("detected")) != bool(counts):
            raise PublicRecordViolation("detected 与 counts 非空性不等")
        return counts

    def simulate(self, state, action_spec, history=None, rng=None,
                 tier: str = "N1", return_private: bool = False):
        action = self._parse_action(action_spec)
        state_bits = self._validate_state(state)
        if rng is None:
            raise KernelScopeError("simulate 必须显式传入 rng（声明随机流身份）")
        if not isinstance(rng, np.random.Generator):
            raise KernelScopeError("rng 必须为 numpy.random.Generator")
        counts = {}
        private = {"tier": tier, "per_disc": []} if return_private else None
        for disc in self._discs:
            if not state_bits[disc["index"]]:
                continue
            outs = self._raw_outcomes(action, disc, tier)
            r = float(rng.random())
            acc = 0.0
            chosen = outs[-1]
            for item in outs:
                acc += float(item[5])
                if r < acc:
                    chosen = item
                    break
            l, k, j, detected, cell = chosen[0], chosen[1], chosen[2], chosen[3], chosen[4]
            if private is not None:
                expo = self._exposure_of(disc, action)
                tb = self._depth_tables[tier]
                true_cell = (self._cell_of(expo["t"], disc["normal"], action["n_depth_bins"])
                             if expo["exposed"] else None)
                measured_depth = None
                measured_normal = None
                if detected:
                    measured_depth = float(expo["t"] + int(l) * self.depth_bin_width)
                    measured_normal = [float(x) for x in _rotate_normal(
                        disc["normal"], tb["polar_offsets_deg"][int(k)],
                        360.0 * int(j) / tb["azimuth_count"])]
                private["per_disc"].append({
                    "candidate_index": disc["index"],
                    "exposed": bool(expo["exposed"]),
                    "exposure_reason": expo["reason"],
                    "true_t": expo["t"],
                    "true_normal": [float(x) for x in disc["normal"]],
                    "true_cell_unnoised": true_cell,
                    "detected": bool(detected),
                    "measured_depth": measured_depth,
                    "measured_normal": measured_normal,
                    "depth_offset_bins": l,
                    "polar_index": k,
                    "azimuth_index": j,
                    "measured_cell": int(cell) if detected else None,
                })
            if detected:
                counts[int(cell)] = counts.get(int(cell), 0) + 1
        rec = self.make_public_record(action, counts)
        if return_private:
            return rec, private
        return rec

    # -- 公开审计接口（评估侧与 S0-3/4 审计用；候选字典几何本身是公开的）----
    def public_quantize(self, action_spec, depth: float, normal) -> int:
        """把 (测量深度, 测量无向法向) 量化到声明格；生成与预测共用同一码路。"""
        action = self._parse_action(action_spec)
        return self._cell_of(float(depth), normal, action["n_depth_bins"])

    def disc_exposure(self, action_spec, disc_index: int) -> dict:
        """逐盘曝光分类（公开候选字典 × 公开行动）：t / 是否曝光 / 原因。"""
        action = self._parse_action(action_spec)
        if not (0 <= int(disc_index) < len(self._discs)):
            raise KernelScopeError("disc_index 越界: %r" % (disc_index,))
        return self._exposure_of(self._discs[int(disc_index)], action)

    def disc_outcome_table(self, action_spec, disc_index: int, tier: str = "N1") -> dict:
        """逐盘精确结局表 {cell_id|EMPTY: Fraction}（预测侧同定义枚举）。"""
        action = self._parse_action(action_spec)
        if not (0 <= int(disc_index) < len(self._discs)):
            raise KernelScopeError("disc_index 越界: %r" % (disc_index,))
        return dict(self._outcome_table(action, self._discs[int(disc_index)], tier))

    def candidate_geometry(self) -> list:
        """公开候选字典几何（模型声明面；不含真值存在性位）。"""
        return [{"candidate_index": d["index"],
                 "center": [float(x) for x in d["center"]],
                 "normal": [float(x) for x in d["normal"]],
                 "radius": float(d["radius"])} for d in self._discs]

    def _validate_state(self, state) -> tuple:
        arr = np.asarray(state)
        if arr.shape != (len(self._discs),):
            raise KernelScopeError("state 形状非法: %r (应为 (%d,))"
                                   % (np.shape(state), len(self._discs)))
        vals = tuple(int(x) for x in arr)
        if any(v not in (0, 1) for v in vals):
            raise KernelScopeError("state 必须为 0/1 存在性位")
        return vals

    def record_probability_exact(self, public_record, state, action_spec,
                                 tier: str = "N1", history=None) -> Fraction:
        """P_θ(y | a) 的精确有理数（计数向量联合 PMF 的系数；不压平、不独立化）。"""
        action = self._parse_action(action_spec)
        bits = self._validate_state(state)
        y = self._validate_public_record(public_record, action)
        cells = sorted(y)
        ytuple = tuple(y[c] for c in cells)
        cell_pos = {c: i for i, c in enumerate(cells)}
        dp = {tuple(0 for _ in cells): Fraction(1, 1)}
        for disc in self._discs:
            if not bits[disc["index"]]:
                continue
            table = self._outcome_table(action, disc, tier)
            ndp = {}
            for partial, p0 in dp.items():
                for outcome, p in table.items():
                    if outcome == EMPTY_OUTCOME:
                        key = partial
                    else:
                        pos = cell_pos.get(int(outcome))
                        if pos is None:
                            continue
                        cnt = partial[pos] + 1
                        if cnt > ytuple[pos]:
                            continue
                        key = partial[:pos] + (cnt,) + partial[pos + 1:]
                    ndp[key] = ndp.get(key, Fraction(0, 1)) + p0 * p
            dp = ndp
            if not dp:
                return Fraction(0, 1)
        return dp.get(ytuple, Fraction(0, 1))

    def log_prob(self, public_record, state, action_spec, history=None,
                 tier: str = "N1"):
        p = self.record_probability_exact(public_record, state, action_spec,
                                          tier=tier, history=history)
        if p == 0:
            return -math.inf
        return math.log(float(p.numerator)) - math.log(float(p.denominator))

    def log_prob_upper(self, public_record, region, action_spec, history=None):
        return {"status": UNSUPPORTED,
                "reason": "first version (KNOWN_SYNTHETIC grid kernel) does not "
                          "implement a certified density upper envelope; the "
                          "current optimizer maximum is NOT a valid upper bound",
                "region": region,
                "kernel_id": self.kernel_id}

    def count_pmf(self, state, action_spec, tier: str = "N1") -> dict:
        action = self._parse_action(action_spec)
        bits = self._validate_state(state)
        n_cells = action["n_depth_bins"] * self.n_orientation_cells
        tables = []
        for disc in self._discs:
            if bits[disc["index"]]:
                tables.append(self._outcome_table(action, disc, tier))
        dp = {tuple(0 for _ in range(n_cells)): Fraction(1, 1)}
        for table in tables:
            ndp = {}
            for vec, p0 in dp.items():
                for outcome, p in table.items():
                    if outcome == EMPTY_OUTCOME:
                        key = vec
                    else:
                        c = int(outcome)
                        key = vec[:c] + (vec[c] + 1,) + vec[c + 1:]
                    ndp[key] = ndp.get(key, Fraction(0, 1)) + p0 * p
                    if len(ndp) > self._pmf_max_entries:
                        raise ComputationUnfinished(
                            "精确 PMF 物化超出声明预算 pmf_max_entries=%d "
                            "(action=%s, state=%r): 返回 COMPUTATION_UNFINISHED"
                            % (self._pmf_max_entries, action["action_id"], bits))
            dp = ndp
        total = sum(dp.values())
        if total != 1:
            raise KernelIdentityError("count_pmf 归一化机器检查失败: sum=%s != 1" % total)
        if any(p < 0 for p in dp.values()):
            raise KernelIdentityError("count_pmf 含负概率")
        return dp

    def validate_scope(self, model_spec: dict, action_spec, tier: str = "N1") -> dict:
        checks = []
        errors = []
        try:
            discs = self._parse_model(model_spec)
            checks.append({"check": "model_parsed", "n_discs": len(discs), "ok": True})
        except (KernelScopeError, KeyError) as exc:
            return {"ok": False, "checks": [], "errors": [
                {"check": "model_parsed", "error": "%s: %s" % (type(exc).__name__, exc)}]}
        r_max = float(self.spec["geometry"]["radius_range"][1])
        w = self.depth_bin_width
        act_list = action_spec if isinstance(action_spec, list) else [action_spec]
        parsed_actions = []
        exposure_rows = []
        min_margin_deg = math.inf
        min_depth_margin_bins = math.inf
        for spec_a in act_list:
            try:
                action = self._parse_action(spec_a)
            except (KernelScopeError, KeyError) as exc:
                errors.append({"check": "action_parsed", "action": str(spec_a),
                               "error": "%s: %s" % (type(exc).__name__, exc)})
                continue
            parsed_actions.append(action)
            for disc in discs:
                expo = self._exposure_of(disc, action)
                row = {"action_id": action["action_id"], "well_id": action["well_id"],
                       "disc_index": disc["index"], "exposed": bool(expo["exposed"]),
                       "reason": expo["reason"], "t": expo["t"],
                       "n_depth_bins": action["n_depth_bins"]}
                if expo["exposed"]:
                    margin = min(abs(expo["t"] - kk * w)
                                 for kk in range(action["n_depth_bins"] + 1))
                    row["min_depth_margin_bins"] = margin / w
                    min_depth_margin_bins = min(min_depth_margin_bins, margin / w)
                    for item in self._raw_outcomes(action, disc, tier):
                        if not item[3]:
                            continue
                        m = _rotate_normal(disc["normal"], item[6], item[7])
                        op = _orientation_params(m)
                        az, dec = op["azimuth_deg"], op["declination_deg"]
                        for edge in self.azimuth_edges[:-1]:
                            d_az = abs((az - edge + 180.0) % 360.0 - 180.0)
                            min_margin_deg = min(min_margin_deg, d_az, 360.0 - d_az)
                        for edge in self.declination_edges[1:-1]:
                            min_margin_deg = min(min_margin_deg, abs(dec - edge))
                exposure_rows.append(row)
        for disc in discs:
            r = disc["radius"]
            if r > r_max + 1e-12:
                errors.append({"check": "disc_radius_in_range",
                               "disc_index": disc["index"],
                               "error": "radius %r > 声明上限 %r" % (r, r_max)})
            for k in range(3):
                if disc["center"][k] - r < self.domain_lo[k] - 1e-12 or \
                   disc["center"][k] + r > self.domain_hi[k] + 1e-12:
                    errors.append({"check": "disc_inside_domain",
                                   "disc_index": disc["index"],
                                   "axis": k,
                                   "error": "轴 %d 包围盒越界 (center=%r r=%r)"
                                            % (k, disc["center"].tolist(), r)})
        checks.append({"check": "discs_in_domain_radius_range", "ok": True,
                       "n_errors_here": sum(1 for e in errors
                                            if e["check"].startswith("disc_"))})
        checks.append({"check": "exposure_rows", "rows": exposure_rows,
                       "n_exposed_rows": sum(1 for x in exposure_rows if x["exposed"])})
        checks.append({"check": "min_orientation_cell_margin_deg",
                       "value": (None if min_margin_deg is math.inf else min_margin_deg)})
        checks.append({"check": "min_depth_bin_margin_bins",
                       "value": (None if min_depth_margin_bins is math.inf
                                 else min_depth_margin_bins)})
        checks.append({"check": "actions_parsed", "n": len(parsed_actions),
                       "n_depth_bins": [a["n_depth_bins"] for a in parsed_actions]})
        return {"ok": not errors, "checks": checks, "errors": errors,
                "kernel_id": self.kernel_id, "tier": tier}
