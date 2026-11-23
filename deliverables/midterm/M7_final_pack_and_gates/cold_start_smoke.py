# -*- coding: utf-8 -*-
"""M7 · STRATA 断网冷启动压力冒烟（无窗，127.0.0.1:8866）。

覆盖：
  1. M5 一键启动器副本的 --self-check（证明无窗一键路径可用，且自清端口）
  2. 直接无窗冷启动 workbench_server，GET /cases、GET /case × 4、POST /apply_check × 3
  3. 绿色 M3 判卷徽章：headless Chrome 实测 4 簇徽章文本/类名/底色 + 截图，外链全阻断
  4. 干净关停 + 端口/进程零残留

退出码：0 全过；2 启动失败；3 HTTP/时延不达标；4 浏览器徽章不达标；5 关停不干净
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CREATE_NO_WINDOW = 134217728
ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
PY = r"E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
PYW = r"E:/app2/anacoda/Anacoda/envs/pytorch_env/pythonw.exe"
M7 = ROOT / "results" / "midterm_sprint_20261120" / "M7_final_pack_and_gates"
M5 = ROOT / "results" / "midterm_sprint_20261120" / "M5_strata_workbench_demo"
M3VB = ROOT / "results" / "midterm_sprint_20261120" / "M3_independent_judge" / "verdict_batch.json"
SRC = ROOT / "src"
RUNTIME = M7 / "runtime"
SHOTS = M7 / "screenshots"
LAT_LIMIT_MS = 2000.0
CASES = [
    ("jardim_c04", "jardim_garcia_uh", True),
    ("jardim_c02", "jardim_garcia_uh", True),
    ("perozzo_c02", "perozzo_finale_ligure", True),
    ("perozzo_c01", "perozzo_finale_ligure", False),
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_json(url: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None,
              timeout: float = 30.0) -> Tuple[int, Dict[str, Any], float]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw), (time.perf_counter() - t0) * 1000.0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw[:800]}
        return int(exc.code), body, (time.perf_counter() - t0) * 1000.0


def wait_ready(port: int, timeout_s: float = 40.0) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{port}/api/identifiability/cases"
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            status, body, ms = http_json(url, timeout=8.0)
            if status == 200 and body.get("ok") is True:
                return {"ok": True, "ms": round(ms, 1), "summary": {
                    "total_cases": body.get("total_cases"),
                    "sha_verified_count": body.get("sha_verified_count"),
                    "m3_present": body.get("m3_present"),
                    "m3_verdict_count": body.get("m3_verdict_count"),
                    "m3_verdict_sha256": body.get("m3_verdict_sha256"),
                }}
            last = f"status={status} ok={body.get('ok')}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.4)
    return {"ok": False, "error": last or "timeout"}


def port_closed(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        return s.connect_ex(("127.0.0.1", port)) != 0
    finally:
        s.close()


def pick_action(body: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    ac = body.get("active_check") or {}
    summary = ac.get("collapse_tree_summary") or {}
    ranked = ac.get("ranked_fractures") or []
    fid = None
    if ranked and isinstance(ranked[0], dict):
        item = ranked[0]
        fid = item.get("fracture_id") or item.get("fragment_id") or item.get("id")
    if not fid and summary:
        fid = sorted(summary, key=lambda k: (summary[k].get("rank") if summary[k].get("rank") is not None else 999))[0]
    branch = None
    if fid and fid in summary:
        bids = summary[fid].get("branch_ids") or []
        if bids:
            branch = bids[0]
    return fid, branch, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8866)
    ap.add_argument("--selfcheck-port", type=int, default=8867)
    args = ap.parse_args(argv)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "schema": "m7-cold-start-smoke/v1",
        "at_utc": started,
        "host": "127.0.0.1",
        "port": args.port,
        "no_console_window": True,
        "creationflags": CREATE_NO_WINDOW,
        "launcher_selfcheck": {},
        "server": {},
        "cases_endpoint": {},
        "case_gets": [],
        "apply_checks": [],
        "certified_probe": {},
        "badge_probe": {},
        "shutdown": {},
        "verdict": "FAILED",
    }

    # ---------- 1) 一键启动器副本 --self-check ----------
    launcher_copy = M7 / "launch_strata_demo_copy.py"
    src_launcher = M5 / "launch_strata_demo.py"
    shutil.copyfile(src_launcher, launcher_copy)
    same = sha256_file(src_launcher) == sha256_file(launcher_copy)
    log = RUNTIME / "launcher_selfcheck.log"
    t0 = time.perf_counter()
    with open(log, "wb") as fh:
        proc = subprocess.run([PY, "-B", "-X", "utf8", str(launcher_copy), "--port", str(args.selfcheck_port),
                               "--no-browser", "--self-check"], cwd=str(ROOT), stdin=subprocess.DEVNULL,
                              stdout=fh, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW,
                              shell=False, timeout=180)
    time.sleep(1.5)
    result["launcher_selfcheck"] = {
        "launcher_src": str(src_launcher),
        "copy": str(launcher_copy),
        "byte_identical": same,
        "sha256": sha256_file(launcher_copy),
        "exit_code": int(proc.returncode),
        "seconds": round(time.perf_counter() - t0, 2),
        "port_after": args.selfcheck_port,
        "port_closed_after": port_closed(args.selfcheck_port),
        "log": str(log.relative_to(M7)).replace("\\", "/"),
    }

    # ---------- 2) 直接无窗冷启动 ----------
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUTF8"] = "1"
    srv_out = RUNTIME / f"server_{args.port}.stdout.log"
    srv_err = RUNTIME / f"server_{args.port}.stderr.log"
    cmd = [PYW, "-B", "-m", "fractureflow.surface_product.workbench_server",
           "--host", "127.0.0.1", "--port", str(args.port),
           "--workspace", str(RUNTIME / f"workbench_home_{args.port}")]
    t_spawn = time.perf_counter()
    with open(srv_out, "wb") as fo, open(srv_err, "wb") as fe:
        srv = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                               stdout=fo, stderr=fe, creationflags=CREATE_NO_WINDOW, shell=False)
    ready = wait_ready(args.port)
    result["server"] = {"cmd": " ".join(cmd), "pid": int(srv.pid), "ready": ready,
                        "spawn_to_ready_s": round(time.perf_counter() - t_spawn, 2),
                        "stdout": str(srv_out.relative_to(M7)).replace("\\", "/"),
                        "stderr": str(srv_err.relative_to(M7)).replace("\\", "/")}
    if not ready.get("ok"):
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()
        (M7 / "cold_start_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2

    base = f"http://127.0.0.1:{args.port}"
    m3sha = sha256_file(M3VB)

    # ---------- 2b) health 端点（m3 在位性与计数） ----------
    st_h, b_h, ms_h = http_json(f"{base}/api/identifiability/health")
    result["health"] = {
        "http_status": st_h, "ok": b_h.get("ok"), "ms": round(ms_h, 1),
        "m1_present": b_h.get("m1_present"), "m2_present": b_h.get("m2_present"),
        "m3_present": b_h.get("m3_present"), "m3_verdict_count": b_h.get("m3_verdict_count"),
        "m3_verdict_sha256": b_h.get("m3_verdict_sha256"),
        "m3_sha_matches_disk": b_h.get("m3_verdict_sha256") == m3sha,
        "m2_run_id": b_h.get("m2_run_id"),
    }
    health_ok = (st_h == 200 and b_h.get("ok") is True and b_h.get("m3_present") is True
                 and b_h.get("m3_verdict_count") == 22 and b_h.get("m3_verdict_sha256") == m3sha)

    # ---------- 3) cases 端点（首击=冷；再击=热，分别记录） ----------
    st, body, ms = http_json(f"{base}/api/identifiability/cases")
    st2, body2, ms2 = http_json(f"{base}/api/identifiability/cases")
    result["cases_endpoint"] = {
        "http_status": st, "ok": body.get("ok"), "cold_ms": round(ms, 1), "warm_ms": round(ms2, 1),
        "warm_http_status": st2, "warm_ok": body2.get("ok"),
        "latency_under_2s": bool(ms2 < LAT_LIMIT_MS),
        "total_cases": body.get("total_cases"), "sha_verified_count": body.get("sha_verified_count"),
        "site_counts": body.get("site_counts"), "status_counts": body.get("status_counts"),
        "m3_verdict_sha256": body.get("m3_verdict_sha256"),
        "m3_sha_matches_disk": body.get("m3_verdict_sha256") == m3sha,
        "latency_note": "首击与热击均执行 22 簇全量 SHA256 复核；>2s 部分在 report.md §6 单列，不冒充 <2s",
    }

    # ---------- 4) 逐案例 GET + apply_check ----------
    http_ok = (st == 200 and body.get("ok") is True and st2 == 200 and body2.get("ok") is True)
    bodies: Dict[str, Dict[str, Any]] = {}
    for cid, site, do_apply in CASES:
        # 首触（冷）单独记录；第二次（热）作为 <2 秒判定口径
        st_c0, b_c0, ms_c0 = http_json(f"{base}/api/identifiability/case?id={cid}")
        st_c, b_c, ms_c = http_json(f"{base}/api/identifiability/case?id={cid}")
        bodies[cid] = b_c
        rec = {
            "cluster_id": cid, "site_id": site, "http_status": st_c, "ok": b_c.get("ok"),
            "ms": round(ms_c, 1), "cold_ms": round(ms_c0, 1),
            "cold_http_status": st_c0, "cold_ok": b_c0.get("ok"),
            "status": b_c.get("status"),
            "m3_judge_verdict": b_c.get("m3_judge_verdict"),
            "m3_verdict_sha256": b_c.get("m3_verdict_sha256"),
            "m3_sha_matches_disk": b_c.get("m3_verdict_sha256") == m3sha,
            "provenance_mode": b_c.get("provenance_mode"),
            "cell_count": b_c.get("cell_count"),
            "fractures": len(b_c.get("fractures") or []),
            "has_collapse_tree_summary": bool((b_c.get("active_check") or {}).get("collapse_tree_summary")),
        }
        result["case_gets"].append(rec)
        http_ok = http_ok and st_c0 == 200 and b_c0.get("ok") is True and st_c == 200 and b_c.get("ok") is True and ms_c < LAT_LIMIT_MS

        if do_apply:
            fid, branch, summary = pick_action(b_c)
            rec_apply = {"cluster_id": cid, "fracture_id": fid, "branch_id": branch,
                         "ranked_count": len(summary)}
            if fid and branch:
                st_a, b_a, ms_a = http_json(f"{base}/api/identifiability/apply_check", method="POST",
                                            payload={"cluster_id": cid, "fracture_id": fid, "branch_id": branch})
                err = b_a.get("error")
                rec_apply.update({
                    "http_status": st_a, "ok": b_a.get("ok"), "ms": round(ms_a, 1),
                    "previous_status": b_a.get("previous_status"),
                    "resulting_status": b_a.get("resulting_status"),
                    "step_count": b_a.get("step_count"),
                    "collapsed_to_certified": b_a.get("collapsed_to_certified"),
                    "check_receipt_sha256": b_a.get("check_receipt_sha256"),
                    "next_step_present": bool(b_a.get("next_step")),
                    "error_code": b_a.get("error_code") or (err.get("code") if isinstance(err, dict) else None),
                })
                http_ok = http_ok and st_a == 200 and b_a.get("ok") is True and ms_a < LAT_LIMIT_MS

                # 两步坍缩：若 step1 返回 next_step，继续确认 step2（答辩 jardim_c02 剧本）
                ns = b_a.get("next_step")
                if isinstance(ns, dict) and ns.get("branches"):
                    ns_fid = ns.get("fracture_id")
                    # 优先选“确认后即认证”的分支，复现答辩讲稿的两步坍缩
                    def _rank(kv):
                        v = kv[1] if isinstance(kv[1], dict) else {}
                        cert = bool(v.get("collapsed_to_certified")) or str(v.get("resulting_status", "")).startswith("CERTIFIED")
                        return (0 if cert else 1, kv[0])
                    ns_bid = sorted(ns["branches"].items(), key=_rank)[0][0]
                    st_a2, b_a2, ms_a2 = http_json(
                        f"{base}/api/identifiability/apply_check", method="POST",
                        payload={"cluster_id": cid, "fracture_id": fid, "branch_id": branch,
                                 "step2_fracture_id": ns_fid, "step2_branch_id": ns_bid})
                    rec_apply["step2"] = {
                        "step2_fracture_id": ns_fid, "step2_branch_id": ns_bid,
                        "http_status": st_a2, "ok": b_a2.get("ok"), "ms": round(ms_a2, 1),
                        "previous_status": b_a2.get("previous_status"),
                        "resulting_status": b_a2.get("resulting_status"),
                        "collapsed_to_certified": b_a2.get("collapsed_to_certified"),
                        "check_receipt_sha256": b_a2.get("check_receipt_sha256"),
                    }
                    http_ok = http_ok and st_a2 == 200 and b_a2.get("ok") is True and ms_a2 < LAT_LIMIT_MS
            else:
                rec_apply.update({"http_status": 0, "ok": False, "error": "NO_ACTION_PICKED"})
                http_ok = False
            result["apply_checks"].append(rec_apply)

    # 已认证簇的 apply_check 反向探针（设计上应拒答，不计入 200 分母）
    p01_fr = (bodies.get("perozzo_c01", {}).get("fractures") or [{}])[0].get("id")
    st_x, b_x, ms_x = http_json(f"{base}/api/identifiability/apply_check", method="POST",
                                payload={"cluster_id": "perozzo_c01", "fracture_id": p01_fr,
                                         "branch_id": "band_left"})
    result["certified_probe"] = {"cluster_id": "perozzo_c01", "http_status": st_x, "ok": b_x.get("ok"),
                                 "ms": round(ms_x, 1), "expected_reject": st_x in (400, 404, 409),
                                 "body_code": b_x.get("code") or b_x.get("error_code") or b_x.get("error")}

    # ---------- 5) 浏览器绿色徽章探针（外链全阻断） ----------
    node = shutil.which("node") or "node"
    js = M7 / "browser_badge_probe.js"
    bp_log = RUNTIME / "browser_badge_probe.stdout.log"
    t_b = time.perf_counter()
    with open(bp_log, "wb") as fh:
        bp = subprocess.run([node, str(js), base, str(SHOTS)] + [c[0] for c in CASES],
                            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=CREATE_NO_WINDOW, shell=False, timeout=900)
    raw = bp_log.read_text(encoding="utf-8", errors="replace")
    try:
        bj = json.loads(raw[raw.index("{"):]) if "{" in raw else {}
    except Exception:
        bj = {"parse_error": raw[-1200:]}
    result["badge_probe"] = {
        "node_exit": int(bp.returncode), "seconds": round(time.perf_counter() - t_b, 2),
        "ok": bj.get("ok"), "externalRequests": bj.get("externalRequests"),
        "abortedExternal": bj.get("abortedExternal"),
        "consoleErrors": bj.get("consoleErrors"), "pageErrors": bj.get("pageErrors"),
        "cases": bj.get("cases"), "error": bj.get("error"),
        "log": str(bp_log.relative_to(M7)).replace("\\", "/"),
    }

    # ---------- 6) 干净关停 ----------
    term_s = time.perf_counter()
    srv.terminate()
    try:
        srv.wait(timeout=20)
        forced = False
    except subprocess.TimeoutExpired:
        srv.kill()
        srv.wait(timeout=10)
        forced = True
    time.sleep(1.0)
    closed = port_closed(args.port)
    result["shutdown"] = {"terminate_seconds": round(time.perf_counter() - term_s, 2),
                          "forced_kill": forced, "returncode": srv.poll(),
                          "port_closed": closed, "residual_pid": None if srv.poll() is not None else srv.pid}

    # ---------- 7) 判定 ----------
    badges = result["badge_probe"].get("cases") or []
    badge_ok = (
        result["badge_probe"].get("ok") is True
        and len(badges) == len(CASES)
        and all(b.get("badge_text") == "M3 独立判卷：AGREE" for b in badges)
        and all("agree" in (b.get("badge_class") or "") for b in badges)
        and not (result["badge_probe"].get("externalRequests") or [])
        and not (result["badge_probe"].get("abortedExternal") or [])
    )
    comp_j = sum(1 for c in CASES if c[1] == "jardim_garcia_uh")
    comp_p = sum(1 for c in CASES if c[1] == "perozzo_finale_ligure")
    result["enumeration"] = {
        "cases_loaded": len(CASES), "jardim_cases": comp_j, "perozzo_cases": comp_p,
        "apply_check_ok": sum(1 for r in result["apply_checks"] if r.get("ok") is True and r.get("http_status") == 200),
        "apply_check_total": len(result["apply_checks"]),
        "step2_collapses_ok": sum(1 for r in result["apply_checks"]
                                  if isinstance(r.get("step2"), dict) and r["step2"].get("ok") is True),
        "badges_green_agree": sum(1 for b in badges if b.get("badge_text") == "M3 独立判卷：AGREE" and "agree" in (b.get("badge_class") or "")),
        "badges_total": len(badges),
        "external_requests_runtime": len(result["badge_probe"].get("externalRequests") or []),
        "cases_cold_ms": result["cases_endpoint"].get("cold_ms"),
        "cases_warm_ms": result["cases_endpoint"].get("warm_ms"),
        "cases_latency_under_2s": result["cases_endpoint"].get("latency_under_2s"),
    }
    ok = (
        result["launcher_selfcheck"].get("byte_identical")
        and result["launcher_selfcheck"].get("exit_code") == 0
        and result["launcher_selfcheck"].get("port_closed_after")
        and health_ok
        and result["cases_endpoint"].get("ok") is True
        and result["cases_endpoint"].get("total_cases") == 22
        and result["cases_endpoint"].get("m3_sha_matches_disk") is True
        and http_ok
        and result["certified_probe"].get("expected_reject") is True
        and badge_ok
        and result["shutdown"].get("port_closed") is True
        and result["shutdown"].get("residual_pid") is None
    )
    result["verdict"] = "PASS" if ok else "FAILED"
    result["pass_reasons"] = {
        "launcher_ok": bool(result["launcher_selfcheck"].get("byte_identical") and result["launcher_selfcheck"].get("exit_code") == 0 and result["launcher_selfcheck"].get("port_closed_after")),
        "health_m3_ok": bool(health_ok),
        "http_ok": bool(http_ok),
        "badge_ok": bool(badge_ok),
        "shutdown_ok": bool(result["shutdown"].get("port_closed") and result["shutdown"].get("residual_pid") is None),
        "two_sites": bool(comp_j >= 1 and comp_p >= 2),
        "cases_endpoint_latency_under_2s": bool(result["cases_endpoint"].get("latency_under_2s")),
    }
    (M7 / "cold_start_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "enumeration": result["enumeration"],
                      "pass_reasons": result["pass_reasons"],
                      "cases": result["cases_endpoint"], "shutdown": result["shutdown"]},
                     ensure_ascii=False, indent=1))
    if not http_ok:
        return 3
    if not badge_ok:
        return 4
    if not (result["shutdown"].get("port_closed") and result["shutdown"].get("residual_pid") is None):
        return 5
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
