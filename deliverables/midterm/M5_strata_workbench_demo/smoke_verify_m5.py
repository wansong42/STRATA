# -*- coding: utf-8 -*-
"""Comprehensive M5 verification: API contract, 5 toothed negative controls,
windowless launcher check, and real headless-Chrome E2E DOM + 3D canvas smoke.

Obeys AGENTS.md §11 strictly:
  * Every Windows subprocess uses creationflags=subprocess.CREATE_NO_WINDOW (134217728)
    and shell=False.
  * Never touches reserved running ports (8791/8801/8810); tests on isolated 8855/8856-8859.
  * Never mutates M1/M2/M3 source directories (negative-control tampering operates solely on
    throw-away copies inside _nc_scratch/).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 134217728)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
M1_REAL_ROOT = REPO_ROOT / "results" / "midterm_sprint_20261120" / "M1_real_to_exact_bridge"
M2_REAL_ROOT = REPO_ROOT / "results" / "midterm_sprint_20261120" / "M2_active_identifiability"
M3_REAL_ROOT = REPO_ROOT / "results" / "midterm_sprint_20261120" / "M3_independent_judge"
SCREENSHOT_DIR = THIS_DIR / "screenshots"
NC_SCRATCH = THIS_DIR / "_nc_scratch"
RUNTIME_DIR = THIS_DIR / "runtime"
PYTHON_EXE = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
CHROME_EXE = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
PLAYWRIGHT_CANDIDATES = [
    REPO_ROOT / "results" / "w179_midterm_consolidation_20261117" / "execution" / "demo" / "node_modules" / "playwright-core",
    Path(r"C:\Users\晚菘\AppData\Roaming\npm\node_modules\agent-browser\node_modules\playwright-core"),
    Path(r"C:\Users\晚菘\AppData\Roaming\npm\node_modules\playwright-core"),
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fractureflow.surface_product.identifiability_bridge import IdentifiabilityBridge  # noqa: E402
from fractureflow.surface_product.workbench_server import make_server  # noqa: E402


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _http_req(url: str, method: str = "GET", body: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
    data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"ok": False, "raw": raw}
        return int(exc.code), parsed


def _start_isolated_server(
    port: int,
    workspace: Path,
    *,
    m1_root: Optional[Path] = None,
    m2_root: Optional[Path] = None,
    m3_root: Optional[Path] = None,
) -> Tuple[Any, threading.Thread, str]:
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    bridge = IdentifiabilityBridge(
        m1_root=m1_root or M1_REAL_ROOT,
        m2_root=m2_root or M2_REAL_ROOT,
        m3_root=m3_root or M3_REAL_ROOT,
        repo_root=REPO_ROOT,
    )
    server = make_server("127.0.0.1", port, workspace, identifiability_bridge=bridge)
    th = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, th, base


def _copy_isolated_subset(dst_root: Path, cluster_ids: Tuple[str, ...]) -> Tuple[Path, Path]:
    if dst_root.exists():
        shutil.rmtree(dst_root, ignore_errors=True)
    m1_dst = dst_root / "M1"
    m2_dst = dst_root / "M2"
    (m1_dst / "clusters").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(M1_REAL_ROOT / "real_network_format.py", m1_dst / "real_network_format.py")
    reg = json.loads((M1_REAL_ROOT / "cluster_register.json").read_text(encoding="utf-8"))
    reg["clusters"] = [c for c in reg["clusters"] if c["cluster_id"] in cluster_ids]
    (m1_dst / "cluster_register.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for cid in cluster_ids:
        cdir = m1_dst / "clusters" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(M1_REAL_ROOT / "clusters" / cid / "model.json", cdir / "model.json")

    m2_runs = M2_REAL_ROOT / "runs"
    if m2_runs.is_dir():
        runs = [p for p in m2_runs.iterdir() if p.is_dir() and (p / "cases").is_dir()]
        if runs:
            src_run = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            dst_cases = m2_dst / "runs" / src_run.name / "cases"
            for cid in cluster_ids:
                src_case = src_run / "cases" / cid
                if src_case.is_dir():
                    (dst_cases / cid).mkdir(parents=True, exist_ok=True)
                    for fname in ("certification.json", "active_check.json"):
                        if (src_case / fname).is_file():
                            shutil.copyfile(src_case / fname, dst_cases / cid / fname)
    return m1_dst, m2_dst


def run_negative_controls_n1_n2_n3() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # ---- M5-N1: 1-byte tamper in model.json (alone & coordinated with register) ----
    n1_dir = NC_SCRATCH / "m5_n1"
    m1_n1, m2_n1 = _copy_isolated_subset(n1_dir, ("jardim_c02", "perozzo_c01"))
    model_p = m1_n1 / "clusters" / "jardim_c02" / "model.json"
    raw_bytes = model_p.read_bytes()
    # Flip 1 byte in an endpoint coordinate string ("19668023/..." -> "19668024/...")
    needle = b"19668023/50000000"
    assert needle in raw_bytes, "expected coordinate literal missing in jardim_c02"
    tampered_bytes = raw_bytes.replace(needle, b"19668024/50000000", 1)
    assert len(tampered_bytes) == len(raw_bytes)
    model_p.write_bytes(tampered_bytes)

    srv1, th1, base1 = _start_isolated_server(8856, n1_dir / "ws", m1_root=m1_n1, m2_root=m2_n1)
    try:
        st_1a, body_1a = _http_req(f"{base1}/api/identifiability/case?id=jardim_c02")
        code_1a = (body_1a.get("error") or {}).get("code")
        assert st_1a == 409 and body_1a.get("ok") is False and code_1a == "MODEL_SHA256_MISMATCH", (st_1a, body_1a)
        results.append({
            "id": "M5-N1a",
            "scenario": "篡改隔离副本 jardim_c02/model.json 1 个字节（保留原 register 与 M2 证书）",
            "expected_reject": True,
            "http_status": st_1a,
            "ok": body_1a.get("ok"),
            "error_code": code_1a,
            "reason": (body_1a.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })
    finally:
        srv1.shutdown()
        srv1.server_close()

    # M5-N1b: Update cluster_register.json to match tampered file, leaving M2 certification.json untouched
    reg_p = m1_n1 / "cluster_register.json"
    reg_doc = json.loads(reg_p.read_text(encoding="utf-8"))
    tampered_obj = json.loads(model_p.read_text(encoding="utf-8"))
    for c in reg_doc["clusters"]:
        if c["cluster_id"] == "jardim_c02":
            c["model_canonical_sha256"] = _canon_sha256(tampered_obj)
            c["model_file_sha256"] = _sha256_file(model_p)
    reg_p.write_text(json.dumps(reg_doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    srv1b, th1b, base1b = _start_isolated_server(8856, n1_dir / "ws_b", m1_root=m1_n1, m2_root=m2_n1)
    try:
        st_1b, body_1b = _http_req(f"{base1b}/api/identifiability/case?id=jardim_c02")
        code_1b = (body_1b.get("error") or {}).get("code")
        assert st_1b == 409 and body_1b.get("ok") is False and code_1b == "CERT_INPUT_SHA256_MISMATCH", (st_1b, body_1b)
        results.append({
            "id": "M5-N1b",
            "scenario": "篡改 1 字节并同步刷新 register 哈希，但 M2 certification.json#input_sha256 仍为原值",
            "expected_reject": True,
            "http_status": st_1b,
            "ok": body_1b.get("ok"),
            "error_code": code_1b,
            "reason": (body_1b.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })
    finally:
        srv1b.shutdown()
        srv1b.server_close()

    # ---- M5-N2: Nonexistent cluster_id & illegal branch_id & apply on certified cluster ----
    n2_dir = NC_SCRATCH / "m5_n2"
    srv2, th2, base2 = _start_isolated_server(8857, n2_dir / "ws")
    try:
        st_2a, body_2a = _http_req(f"{base2}/api/identifiability/case?id=nonexistent_cluster_999")
        code_2a = (body_2a.get("error") or {}).get("code")
        assert st_2a == 404 and body_2a.get("ok") is False and code_2a == "CLUSTER_NOT_FOUND", (st_2a, body_2a)
        results.append({
            "id": "M5-N2a",
            "scenario": "请求不存在的 cluster_id=nonexistent_cluster_999",
            "expected_reject": True,
            "http_status": st_2a,
            "ok": body_2a.get("ok"),
            "error_code": code_2a,
            "reason": (body_2a.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })

        st_2b, body_2b = _http_req(
            f"{base2}/api/identifiability/apply_check",
            method="POST",
            body={
                "cluster_id": "jardim_c02",
                "fracture_id": "frag_306f0d79a2c51ef819597b812f515062",
                "branch_id": "FAKE_ILLEGAL_BRANCH_999",
            },
        )
        code_2b = (body_2b.get("error") or {}).get("code")
        assert st_2b == 400 and body_2b.get("ok") is False and code_2b == "INVALID_BRANCH_ID", (st_2b, body_2b)
        results.append({
            "id": "M5-N2b",
            "scenario": "对有效歧义簇请求非法 branch_id=FAKE_ILLEGAL_BRANCH_999",
            "expected_reject": True,
            "http_status": st_2b,
            "ok": body_2b.get("ok"),
            "error_code": code_2b,
            "reason": (body_2b.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })

        st_2c, body_2c = _http_req(
            f"{base2}/api/identifiability/apply_check",
            method="POST",
            body={
                "cluster_id": "perozzo_c01",
                "fracture_id": "perozzo_row_0620",
                "branch_id": "any",
            },
        )
        code_2c = (body_2c.get("error") or {}).get("code")
        assert st_2c == 400 and body_2c.get("ok") is False and code_2c == "INVALID_CHECK_TARGET", (st_2c, body_2c)
        results.append({
            "id": "M5-N2c",
            "scenario": "对已精确认证的 perozzo_c01 (CERTIFIED_DISCONNECTED) 强行发起 apply_check",
            "expected_reject": True,
            "http_status": st_2c,
            "ok": body_2c.get("ok"),
            "error_code": code_2c,
            "reason": (body_2c.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })

        st_2d, body_2d = _http_req(
            f"{base2}/api/identifiability/apply_check",
            method="POST",
            body={
                "cluster_id": "jardim_c02",
                "fracture_id": "frag_786debfdca7bb1f52f2ff3766d61ca0c",
                "branch_id": "mode_TILT_0DEG",
                "step2_fracture_id": "frag_306f0d79a2c51ef819597b812f515062",
                "step2_branch_id": "FAKE_STEP2_BRANCH_999",
            },
        )
        code_2d = (body_2d.get("error") or {}).get("code")
        assert st_2d == 400 and body_2d.get("ok") is False and code_2d == "INVALID_BRANCH_ID", (st_2d, body_2d)
        results.append({
            "id": "M5-N2d",
            "scenario": "对两步坍缩分支提交非法 step2_branch_id=FAKE_STEP2_BRANCH_999",
            "expected_reject": True,
            "http_status": st_2d,
            "ok": body_2d.get("ok"),
            "error_code": code_2d,
            "reason": (body_2d.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })
    finally:
        srv2.shutdown()
        srv2.server_close()

    # ---- M5-N3: Synthetic scope impostor dropped into real clusters directory ----
    n3_dir = NC_SCRATCH / "m5_n3"
    m1_n3, m2_n3 = _copy_isolated_subset(n3_dir, ("jardim_c02",))
    impostor_dir = m1_n3 / "clusters" / "synthetic_impostor"
    impostor_dir.mkdir(parents=True, exist_ok=True)
    impostor_doc = json.loads((m1_n3 / "clusters" / "jardim_c02" / "model.json").read_text(encoding="utf-8"))
    impostor_doc["scope"] = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
    impostor_doc["snapshot_id"] = "synthetic_impostor"
    (impostor_dir / "model.json").write_text(
        json.dumps(impostor_doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    srv3, th3, base3 = _start_isolated_server(8858, n3_dir / "ws", m1_root=m1_n3, m2_root=m2_n3)
    try:
        st_3a, body_3a = _http_req(f"{base3}/api/identifiability/cases")
        code_3a = (body_3a.get("error") or {}).get("code")
        assert st_3a == 409 and body_3a.get("ok") is False and code_3a == "SCOPE_VIOLATION", (st_3a, body_3a)
        st_3b, body_3b = _http_req(f"{base3}/api/identifiability/case?id=synthetic_impostor")
        code_3b = (body_3b.get("error") or {}).get("code")
        assert st_3b == 409 and body_3b.get("ok") is False and code_3b == "SCOPE_VIOLATION", (st_3b, body_3b)
        results.append({
            "id": "M5-N3a",
            "scenario": "向真实 clusters/ 目录投放 scope=SYNTHETIC_DECLARED_TRIANGLE_UNION 冒充文件",
            "expected_reject": True,
            "http_status": st_3a,
            "ok": body_3a.get("ok"),
            "error_code": code_3a,
            "reason": (body_3a.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })
    finally:
        srv3.shutdown()
        srv3.server_close()

    # Also test registered cluster overwritten with synthetic scope
    shutil.rmtree(impostor_dir)
    reg_j = m1_n3 / "clusters" / "jardim_c02" / "model.json"
    jdoc = json.loads(reg_j.read_text(encoding="utf-8"))
    jdoc["scope"] = "SYNTHETIC_DECLARED_TRIANGLE_UNION"
    reg_j.write_text(json.dumps(jdoc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    srv3b, th3b, base3b = _start_isolated_server(8858, n3_dir / "ws_b", m1_root=m1_n3, m2_root=m2_n3)
    try:
        st_3c, body_3c = _http_req(f"{base3b}/api/identifiability/case?id=jardim_c02")
        code_3c = (body_3c.get("error") or {}).get("code")
        assert st_3c == 409 and body_3c.get("ok") is False and code_3c == "SCOPE_VIOLATION", (st_3c, body_3c)
        results.append({
            "id": "M5-N3b",
            "scenario": "已登记真实簇 jardim_c02 的 scope 被改写为 SYNTHETIC_DECLARED_TRIANGLE_UNION",
            "expected_reject": True,
            "http_status": st_3c,
            "ok": body_3c.get("ok"),
            "error_code": code_3c,
            "reason": (body_3c.get("error") or {}).get("message"),
            "correctly_rejected": True,
        })
    finally:
        srv3b.shutdown()
        srv3b.server_close()

    return results


NODE_E2E_SCRIPT = r'''
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const { createRequire } = require("node:module");

const pwIndex = process.argv[2];
const base = process.argv[3];
const shotDir = process.argv[4];
const chromePath = process.argv[5];

const { chromium } = require(pwIndex);

async function main() {
  fs.mkdirSync(shotDir, { recursive: true });
  const report = {
    base,
    started_at_utc: new Date().toISOString(),
    steps: [],
    externalRequests: [],
    consoleErrors: [],
    pageErrors: [],
    screenshots: [],
    addFragmentCheck: {},
    dualWitnessCheck: {},
    activeCollapseCheck: {},
    negativeUiCheck: {},
    ok: false,
  };

  let browser;
  try {
    browser = await chromium.launch({
      executablePath: chromePath,
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--window-size=1440,900",
        "--lang=zh-CN",
        "--force-device-scale-factor=1",
      ],
    });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      locale: "zh-CN",
      timezoneId: "Asia/Shanghai",
    });
    const page = await context.newPage();
    page.setDefaultTimeout(120000);

    let expectHttpReject = false;
    page.on("request", (req) => {
      const u = new URL(req.url());
      if (u.protocol !== "data:" && u.protocol !== "blob:" && u.origin !== new URL(base).origin) {
        report.externalRequests.push({ method: req.method(), url: req.url() });
      }
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        if (!expectHttpReject) report.consoleErrors.push(msg.text());
      }
    });
    page.on("pageerror", (err) => {
      report.pageErrors.push(String((err && err.stack) || err));
    });

    async function shot(name) {
      const p = path.join(shotDir, name);
      await page.screenshot({ path: p, fullPage: false });
      report.screenshots.push({ name, path: p, bytes: fs.statSync(p).size });
    }

    async function canvasFingerprint() {
      return await page.evaluate(() => {
        const c = document.getElementById("plate");
        return c ? c.toDataURL("image/png").slice(0, 4000) : "";
      });
    }

    // 1) Load Workbench mode & verify ADD_FRAGMENT option + transaction
    await page.goto(base + "/", { waitUntil: "domcontentloaded" });
    await page.locator('[data-testid="open-synthetic"]').waitFor({ state: "visible" });
    await page.locator('[data-testid="open-synthetic"]').click();
    await page.locator('[data-testid="object-row-surf_a"]').waitFor({ state: "visible", timeout: 30000 });
    await page.locator('[data-testid="object-row-frag_a"]').click();
    await page.waitForTimeout(300);
    await page.locator('[data-testid="object-row-surf_a"]').click();
    await page.locator('#ed-action').waitFor({ state: "visible" });

    const edOptions = await page.evaluate(() => {
      const sel = document.getElementById("ed-action");
      return sel ? Array.from(sel.options).map((o) => o.value) : [];
    });
    if (!edOptions.includes("ADD_FRAGMENT")) {
      throw new Error("ADD_FRAGMENT missing from #ed-action options: " + JSON.stringify(edOptions));
    }
    await page.selectOption("#ed-action", "ADD_FRAGMENT");
    await page.waitForTimeout(300);
    const payloadText = await page.locator("#ed-payload").inputValue();
    const parsedPayload = JSON.parse(payloadText);
    if (!parsedPayload.fragment || !parsedPayload.surface_id) {
      throw new Error("ADD_FRAGMENT skeleton invalid: " + payloadText);
    }
    await page.locator('[data-testid="preview"]').click();
    await page.locator('[data-testid="commit"]').waitFor({ state: "visible", timeout: 30000 });
    await page.locator('[data-testid="commit"]').click();
    await page.waitForFunction(() => {
      const el = document.querySelector('[data-testid="project-version"]');
      return el && Number(el.textContent.trim()) >= 2;
    }, null, { timeout: 30000 });
    await shot("00_workbench_add_fragment.png");
    report.addFragmentCheck = {
      hasOption: true,
      optionCount: edOptions.length,
      skeletonKeys: Object.keys(parsedPayload),
      committedVersion: await page.locator('[data-testid="project-version"]').textContent(),
    };

    // 2) Negative control on ADD_FRAGMENT (insufficient indices < 3 rejected HTTP 400)
    expectHttpReject = true;
    const negAdd = await page.evaluate(async () => {
      const r = await fetch("/api/edit/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "ADD_FRAGMENT",
          target_ids: ["frag_a"],
          payload: { indices: [0, 1], observation_id: "obs_synth" },
          base_version: Number(document.querySelector('[data-testid="project-version"]').textContent.trim()),
        }),
      });
      return { status: r.status, json: await r.json() };
    });
    expectHttpReject = false;
    if (negAdd.status < 400 || negAdd.json.ok !== false) {
      throw new Error("Expected ADD_FRAGMENT <3 indices to fail, got: " + JSON.stringify(negAdd));
    }
    report.negativeUiCheck.badAddFragment = { httpStatus: negAdd.status, error: negAdd.json.error };

    // 3) Switch to Identifiability Mode and verify all 22 real clusters load
    await page.locator('[data-testid="mode-tab-identifiability"]').click();
    await page.locator('[data-testid="id-cluster-row-jardim_c02"]').waitFor({ state: "visible", timeout: 30000 });
    await page.locator('[data-testid="id-status-badge"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 45000 });
    await page.waitForTimeout(450);

    const clusterCount = await page.locator('[data-testid^="id-cluster-row-"]').count();
    if (clusterCount !== 22) {
      throw new Error("Expected 22 real clusters, rendered: " + clusterCount);
    }
    const boundaryText = await page.locator('[data-testid="id-academic-boundary-bar"]').innerText();
    if (!boundaryText.includes("REAL_DERIVED_CONDITIONED_FINITE_UNION") || !boundaryText.includes("null")) {
      throw new Error("Academic boundary bar incomplete: " + boundaryText);
    }
    const fpNominal = await canvasFingerprint();
    await shot("01_overview_jardim_c02_ambiguous.png");

    // 4) Toggle Connected Witness (m+, t+) and Disconnected Witness (m-, t-)
    await page.locator('[data-testid="witness-btn-connected"]').click();
    await page.waitForTimeout(350);
    const connCap = await page.locator('#vp-caption').innerText();
    const fpConn = await canvasFingerprint();
    if (!connCap.includes("连通链") || fpConn === fpNominal) {
      throw new Error("Connected witness did not visibly update canvas/caption: " + connCap);
    }
    await shot("02_witness_connected_emerald.png");

    await page.locator('[data-testid="witness-btn-disconnected"]').click();
    await page.waitForTimeout(350);
    const discCap = await page.locator('#vp-caption').innerText();
    const fpDisc = await canvasFingerprint();
    if (!discCap.includes("断割见证") || fpDisc === fpConn) {
      throw new Error("Disconnected witness did not visibly update canvas/caption: " + discCap);
    }
    await shot("03_witness_disconnected_crimson.png");
    report.dualWitnessCheck = {
      clusterId: "jardim_c02",
      nominalChangedToConnected: fpConn !== fpNominal,
      connectedChangedToDisconnected: fpDisc !== fpConn,
      connectedCaption: connCap.replace(/\s+/g, " "),
      disconnectedCaption: discCap.replace(/\s+/g, " "),
    };

    // 5) Active Next-Best-Check & Real-time 2-step Collapse on jardim_c02 (AMBIGUOUS -> Step 1 -> Step 2 CERTIFIED_DISCONNECTED)
    const beforeBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    const provInitial = await page.locator('[data-testid="id-provenance-mode"]').innerText();
    await page.locator('[data-testid="apply-check-r1-b0"]').click();
    await page.locator('[data-testid="id-next-step-panel"]').waitFor({ state: "visible", timeout: 25000 });
    await page.waitForTimeout(3200);
    const step1Badge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    if (step1Badge !== "AMBIGUOUS_WITH_VERIFIED_WITNESSES") {
      throw new Error("Expected jardim_c02 rank#1 step1 to remain AMBIGUOUS pending step2, got: " + step1Badge);
    }
    await shot("04a_step1_narrowed_jardim_c02.png");

    await page.locator('[data-testid="apply-step2-b1"]').click();
    await page.locator('[data-testid="id-status-badge"][data-status="CERTIFIED_DISCONNECTED"]').waitFor({ state: "visible", timeout: 25000 });
    await page.locator('[data-testid="id-collapse-result"]').waitFor({ state: "visible" });
    await page.waitForTimeout(3200);
    const afterDiscBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    const provDisc = await page.locator('[data-testid="id-provenance-mode"]').innerText();
    await shot("04_active_collapse_certified_disconnected.png");

    // Reset back to unreviewed AMBIGUOUS state
    await page.locator('[data-testid="id-reset-collapse"]').click();
    await page.locator('[data-testid="id-status-badge"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 10000 });
    const resetBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");

    // 6) Select holed Jardim cluster jardim_c04 & collapse AMBIGUOUS -> CERTIFIED_CONNECTED (1-step)
    await page.locator('[data-testid="id-cluster-row-jardim_c04"]').click();
    await page.locator('[data-testid="id-status-badge"][data-cluster="jardim_c04"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 45000 });
    await page.waitForTimeout(450);
    const c04Title = await page.locator('#vp-caption').textContent();
    if (!String(c04Title).includes("jardim_c04")) {
      throw new Error("Expected jardim_c04 caption, got: " + c04Title);
    }
    await shot("05_jardim_c04_holed_before_collapse.png");

    await page.locator('[data-testid="apply-check-r1-b0"]').click();
    await page.locator('[data-testid="id-status-badge"][data-cluster="jardim_c04"][data-status="CERTIFIED_CONNECTED"]').waitFor({ state: "visible", timeout: 25000 });
    await page.waitForTimeout(3200);
    const afterConnBadge = await page.locator('[data-testid="id-status-badge"]').getAttribute("data-status");
    await shot("06_active_collapse_certified_connected.png");

    // 7) Switch site to Perozzo 2024 and inspect perozzo_c02
    await page.locator('[data-testid="ident-site-perozzo_finale_ligure"]').click();
    await page.locator('[data-testid="id-cluster-row-perozzo_c02"]').waitFor({ state: "visible" });
    await page.locator('[data-testid="id-cluster-row-perozzo_c02"]').click();
    await page.locator('[data-testid="id-status-badge"][data-cluster="perozzo_c02"][data-status="AMBIGUOUS_WITH_VERIFIED_WITNESSES"]').waitFor({ state: "visible", timeout: 30000 });
    await page.waitForTimeout(450);
    await shot("07_perozzo_c02_outcrop_overview.png");

    report.activeCollapseCheck = {
      jardim_c02_before: beforeBadge,
      jardim_c02_step1: step1Badge,
      jardim_c02_after_step2: afterDiscBadge,
      jardim_c02_after_reset: resetBadge,
      jardim_c02_provenance_initial: provInitial,
      jardim_c02_provenance_collapse: provDisc,
      jardim_c04_after: afterConnBadge,
      perozzo_c02_loaded: true,
      clusterCountRendered: clusterCount,
    };

    if (report.externalRequests.length > 0) {
      throw new Error("External network requests detected: " + JSON.stringify(report.externalRequests));
    }
    if (report.consoleErrors.length > 0 || report.pageErrors.length > 0) {
      throw new Error("Browser console/page errors: " + JSON.stringify({ console: report.consoleErrors, page: report.pageErrors }));
    }

    report.ok = true;
    report.finished_at_utc = new Date().toISOString();
    console.log(JSON.stringify(report, null, 2));
  } finally {
    if (browser) await browser.close();
  }
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
'''


def run_browser_e2e() -> Dict[str, Any]:
    assert CHROME_EXE.is_file(), f"Chrome not found at {CHROME_EXE}"
    pw_dir = next((p for p in PLAYWRIGHT_CANDIDATES if (p / "index.js").is_file()), None)
    assert pw_dir is not None, f"playwright-core not found in candidates: {PLAYWRIGHT_CANDIDATES}"

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    driver_path = RUNTIME_DIR / "m5_e2e_driver.js"
    driver_path.write_text(NODE_E2E_SCRIPT, encoding="utf-8")

    # 1) Launcher self-check on isolated port 8855
    launcher_proc = subprocess.run(
        [PYTHON_EXE, "-B", "-X", "utf8", str(THIS_DIR / "launch_strata_demo.py"), "--self-check", "--port", "8855", "--no-browser"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        shell=False,
        creationflags=CREATE_NO_WINDOW,
        timeout=60,
    )
    assert launcher_proc.returncode == 0, f"launcher self-check failed: {launcher_proc.stdout}\n{launcher_proc.stderr}"

    # 2) Start real isolated server on 127.0.0.1:8855 & run headless Chrome E2E
    ws = RUNTIME_DIR / "smoke_ws_8855"
    srv, th, base = _start_isolated_server(8855, ws)
    try:
        proc = subprocess.run(
            ["node", str(driver_path), str(pw_dir / "index.js"), base, str(SCREENSHOT_DIR), str(CHROME_EXE)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=False,
            creationflags=CREATE_NO_WINDOW,
            timeout=240,
        )
        assert proc.returncode == 0, f"Browser E2E failed (exit={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        e2e_report = json.loads(proc.stdout)
        assert e2e_report["ok"] is True
        assert len(e2e_report["screenshots"]) >= 4
        assert e2e_report["externalRequests"] == []
        assert e2e_report["consoleErrors"] == [] and e2e_report["pageErrors"] == []
        return e2e_report
    finally:
        srv.shutdown()
        srv.server_close()


def run_surface_product_regression(full_pytest: bool = False) -> Dict[str, Any]:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    targets = (
        ["tests/surface_product"]
        if full_pytest
        else [
            "tests/surface_product/test_contracts.py",
            "tests/surface_product/test_edit_engine.py",
            "tests/surface_product/test_workbench.py",
        ]
    )
    cmd = [PYTHON_EXE, "-B", "-X", "utf8", "-m", "pytest", *targets, "-q", "--tb=short"]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        shell=False,
        creationflags=CREATE_NO_WINDOW,
        timeout=900,
    )
    (RUNTIME_DIR / "pytest_surface_product.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (RUNTIME_DIR / "pytest_surface_product.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    assert proc.returncode == 0, f"pytest surface_product failed (exit={proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1000:]}"
    summary_line = [ln.strip() for ln in proc.stdout.splitlines() if "passed" in ln]
    return {
        "id": "M5-N4",
        "scenario": "复跑既有 tests/surface_product/ 定向回归测试（验证 T-D 编辑/保存/导出契约零破坏）",
        "cmd": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "summary": summary_line[-1] if summary_line else "passed",
        "stdout_file": "runtime/pytest_surface_product.stdout.txt",
        "correctly_passed": True,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run M5 negative controls + E2E browser smoke.")
    parser.add_argument("--full-pytest", action="store_true", help="Run entire tests/surface_product/ suite (takes ~8 min)")
    args = parser.parse_args(argv)

    NC_SCRATCH.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    started = _now_utc()
    nc_items = run_negative_controls_n1_n2_n3()
    reg_info = run_surface_product_regression(full_pytest=args.full_pytest)
    e2e_info = run_browser_e2e()

    nc_doc = {
        "schema": "m5-negative-controls/v1",
        "task_id": "M5",
        "started_at_utc": started,
        "finished_at_utc": _now_utc(),
        "no_console_window": True,
        "creationflags": int(CREATE_NO_WINDOW),
        "controls_summary": {
            "total_classes": 5,
            "total_checks": len(nc_items) + 2,
            "all_toothed_rejections_passed": all(x["correctly_rejected"] for x in nc_items),
            "surface_product_regression_passed": reg_info["correctly_passed"],
            "browser_e2e_and_bad_add_fragment_passed": e2e_info["ok"],
        },
        "M5_N1_to_M5_N3": nc_items,
        "M5_N4_regression": reg_info,
        "M5_N5_browser_e2e": e2e_info,
    }
    out_path = THIS_DIR / "negative_controls.json"
    out_path.write_text(json.dumps(nc_doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "negative_controls_file": str(out_path),
        "n1_to_n3_count": len(nc_items),
        "n4_regression": reg_info["summary"],
        "n5_screenshots": len(e2e_info["screenshots"]),
        "n5_external_requests": len(e2e_info["externalRequests"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
