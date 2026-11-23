# -*- coding: utf-8 -*-
"""S1-XR 新引擎：精确覆盖的全轨迹枚举（整数/精确有理数，前缀共享 DFS）。

语义逐条对齐 src/fractureflow/action_expansion.py::enumerate_sequence + sequential_evidence.EProcess：
- E-process:  E_s^(t) = Pi_{u<=t} q_u / p_{s,u}；q_u = S_u/(W_{u-1} L_u)，p_{s,u}=A_{u,s}/L_u
  => E_s^(t) = U_t / (V_t * N_s^(t))，其中 U_t=Pi S_u、V_t=Pi_{u=0..t-1} W_u、N_s^(t)=Pi A_{u,s}
  均为整数（精确，无浮点）。
- S_t = {theta: max_{u<=t} E_u(theta) < 1/alpha}  <=>  对每个 s 单次判据 U_t*alpha_num >= alpha_den*V_t*N_s^(t)
  （Emax 单调不减 => 一旦出 S 永久出 S，用位掩码记）。
- 冲突路径（q_u==0）：E/混合后验冻结不更新，但概率账 p_path 仍按 A 连乘；冲突标志永久置位。
- 终步 status 判据与 certified_correct/wrong 切分与库件一致。
"""
from __future__ import annotations

import itertools
import math
import time
from fractions import Fraction

STATUS_CERTIFIED = "CERTIFIED"
STATUS_EMPTY = "EMPTY"
STATUS_DUAL = "DUAL"
STATUS_CONFLICT = "MODEL_CONFLICT"
STATUS_UNRESOLVED = "SEARCH_UNRESOLVED"


def _fs(fr):
    """Fraction -> 本项目统一字符串形式 n/d（与 action_expansion.frac_str 相同）。"""
    return "%d/%d" % (fr.numerator, fr.denominator)


def action_integer_tables(kernel, actions, tier, states=tuple(range(16))):
    """把 sweep_action 的 pmat（Fraction）转成公共分母整数表 A[r][s]。"""
    from fractureflow.action_expansion import sweep_action
    tabs = []
    d_seq = 1
    for a in actions:
        sw = sweep_action(kernel, a, tier, states, max(states).bit_length())
        rows = []
        den = 1
        for rk, _p in sw["support"]:
            for v in sw["pmat"][rk]:
                den = den * v.denominator // math.gcd(den, v.denominator) if v else den
        for rk, _p in sw["support"]:
            vals = []
            for v in sw["pmat"][rk]:
                x = v * den
                assert x.denominator == 1, "pmat 分母未对齐"
                vals.append(int(x))
            rows.append(tuple(vals))
        tabs.append({"action_id": a["action_id"], "n": len(rows), "A": rows, "L": den})
        d_seq *= den
    return {"tier": tier, "states": tuple(states), "tables": tabs,
            "sizes": [t["n"] for t in tabs], "D_seq": d_seq,
            "support_actions": [t["action_id"] for t in tabs]}


def enumerate_exact(tabs, alpha_seq, labels, tier=None, collect_first_exclusion=True,
                    drop_final_statuses=None):
    """精确全轨迹枚举（前缀共享）。返回与库件 enumerate_sequence 同构的 payload。"""
    states = tabs["states"]
    n_s = len(states)
    assert n_s == 16
    actions = tabs["tables"]
    T = len(actions)
    alpha = Fraction(alpha_seq)
    a_num, a_den = alpha.numerator, alpha.denominator
    full_mask = (1 << n_s) - 1
    labels_t = tuple(int(labels[s]) for s in states)
    t0 = time.time()

    ex = [Fraction(0, 1)] * n_s
    sv = [Fraction(0, 1)] * n_s
    cc = [Fraction(0, 1)] * n_s
    cw = [Fraction(0, 1)] * n_s
    du = [Fraction(0, 1)] * n_s
    em = [Fraction(0, 1)] * n_s
    cf = [Fraction(0, 1)] * n_s
    un = [Fraction(0, 1)] * n_s
    first = [[Fraction(0, 1)] * (T + 1) for _ in range(n_s)]
    n_nonzero = [0] * n_s
    status_counts = {}
    n_paths = [0]
    max_n_S = [0]
    n_all_excl = [0]
    d_seq = tabs["D_seq"]

    def leaf(N_acc, outmask, first_out, conflict, maxn):
        n_paths[0] += 1
        S_list = [i for i in range(n_s) if not (outmask >> i) & 1]
        if maxn > max_n_S[0]:
            max_n_S[0] = maxn
        if conflict:
            st = STATUS_CONFLICT
        elif not S_list:
            st = STATUS_EMPTY
        else:
            g = sorted({labels_t[i] for i in S_list})
            st = STATUS_CERTIFIED if len(g) == 1 else (STATUS_DUAL if len(g) == 2 else STATUS_UNRESOLVED)
        status_counts[st] = status_counts.get(st, 0) + 1
        if drop_final_statuses and st in drop_final_statuses:
            return
        if outmask == full_mask:
            n_all_excl[0] += 1
        for i in range(n_s):
            w = N_acc[i]
            if w == 0:
                continue
            n_nonzero[i] += 1
            if (outmask >> i) & 1:
                ex[i] += w
                if collect_first_exclusion:
                    first[i][first_out[i]] += w
            else:
                sv[i] += w
            if st == STATUS_CERTIFIED:
                if g[0] == labels_t[i]:
                    cc[i] += w
                else:
                    cw[i] += w
            elif st == STATUS_DUAL:
                du[i] += w
            elif st == STATUS_EMPTY:
                em[i] += w
            elif st == STATUS_CONFLICT:
                cf[i] += w
            else:
                un[i] += w

    def dfs(k, N_acc, N_eff, U, V, outmask, first_out, conflict, maxn):
        if k == T:
            leaf(N_acc, outmask, first_out, conflict, maxn)
            return
        A_list = actions[k]["A"]
        w_eff = sum(N_eff)
        for A in A_list:
            # q 分子：sum_s N_eff[s]*A[s]
            q_num = 0
            for i in range(n_s):
                ni = N_eff[i]
                if ni:
                    q_num += ni * A[i]
            N_acc2 = tuple(N_acc[i] * A[i] for i in range(n_s))
            if q_num == 0:
                dfs(k + 1, N_acc2, N_eff, U, V, outmask, first_out, True, maxn)
                continue
            N_eff2 = tuple(N_eff[i] * A[i] for i in range(n_s))
            U2 = U * q_num
            V2 = V * w_eff
            out2 = outmask
            first2 = first_out
            for i in range(n_s):
                if (out2 >> i) & 1:
                    continue
                if N_eff2[i] == 0 or U2 * a_num >= a_den * V2 * N_eff2[i]:
                    out2 |= (1 << i)
                    if first2 is first_out:
                        first2 = list(first_out)
                    first2[i] = k + 1
            if first2 is not first_out and isinstance(first2, list):
                first2 = tuple(first2)
            n_S2 = n_s - bin(out2).count("1")
            dfs(k + 1, N_acc2, N_eff2, U2, V2, out2, first2, conflict, maxn if maxn >= n_S2 else n_S2)

    N0 = tuple(1 for _ in range(n_s))
    dfs(0, N0, N0, 1, 1, 0, tuple(0 for _ in range(n_s)), False, 0)

    per_state = {}
    for i, s in enumerate(states):
        tot = ex[i] + sv[i]
        totp = tot / d_seq
        per_state[str(s)] = {
            "excluded_mass": _fs(ex[i] / d_seq),
            "survive_mass": _fs(sv[i] / d_seq),
            "certified_correct_mass": _fs(cc[i] / d_seq),
            "certified_wrong_mass": _fs(cw[i] / d_seq),
            "dual_mass": _fs(du[i] / d_seq),
            "empty_mass": _fs(em[i] / d_seq),
            "conflict_mass": _fs(cf[i] / d_seq),
            "unresolved_mass": _fs(un[i] / d_seq),
            "first_exclusion_mass": {str(t): _fs(first[i][t] / d_seq) for t in range(1, T + 1)},
            "n_paths_nonzero": n_nonzero[i],
            "total_prob": _fs(totp),
            "prob_closed": bool(totp == 1),
            "coverage_exact": _fs(ex[i] / d_seq),
            "coverage_float": float(ex[i] / d_seq),
            "coverage_le_alpha": bool(ex[i] / d_seq <= alpha),
            "certified_correct_rate": float(cc[i] / tot) if tot else None,
            "dual_rate": float(du[i] / tot) if tot else None,
            "empty_rate": float(em[i] / tot) if tot else None,
            "conflict_rate": float(cf[i] / tot) if tot else None,
            "unresolved_rate": float(un[i] / tot) if tot else None,
        }
    n_leaves = 1
    for t in actions:
        n_leaves *= t["n"]
    return {
        "status": "EXACT",
        "tier": tier if tier is not None else tabs["tier"],
        "alpha_seq": str(alpha),
        "alpha_seq_float": float(alpha),
        "n_leaves": n_leaves,
        "support_sizes": list(tabs["sizes"]),
        "support_actions": list(tabs["support_actions"]),
        "elapsed_s": round(time.time() - t0, 3),
        "totals": {
            "n_paths": n_paths[0],
            "max_n_S": max_n_S[0],
            "status_counts": dict(status_counts),
            "n_paths_all_states_excluded": n_all_excl[0],
        },
        "per_state": per_state,
    }
