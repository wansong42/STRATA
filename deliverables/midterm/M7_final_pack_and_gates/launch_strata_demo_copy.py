# -*- coding: utf-8 -*-
"""STRATA M5 Defense Demo Launcher (CLI / windowless-safe).

Starts an isolated STRATA workbench server on 127.0.0.1:8855 (never colliding
with 8791/8801/8810), verifies /api/identifiability/cases readiness (including
SHA256 & real-derived scope checks), and optionally opens the 3D Dual-Witness
& Active Collapse view in the browser.

Strictly obeys AGENTS.md §11:
  * NEVER uses shell=True, os.system, os.popen, cmd /c, start, .bat, or .cmd.
  * Every Windows child process uses creationflags=subprocess.CREATE_NO_WINDOW
    (134217728) and prefers pythonw.exe when detaching.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 134217728)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_PORT = 8855
FORBIDDEN_PORTS = frozenset({8791, 8801, 8810})
RUNTIME_DIR = THIS_DIR / "runtime"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_pythonw() -> str:
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    return str(pyw) if pyw.is_file() else str(exe)


def _http_get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return {"http_status": resp.status, "body": json.loads(raw)}


def wait_until_ready(host: str, port: int, timeout_s: float = 25.0) -> Dict[str, Any]:
    url = f"http://{host}:{port}/api/identifiability/cases"
    deadline = time.monotonic() + timeout_s
    last_err: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            res = _http_get_json(url, timeout=10.0)
            body = res["body"]
            if res["http_status"] == 200 and body.get("ok") is True:
                return {
                    "ok": True,
                    "url": url,
                    "http_status": res["http_status"],
                    "total_cases": body.get("total_cases"),
                    "sha_verified_count": body.get("sha_verified_count"),
                    "site_counts": body.get("site_counts"),
                    "status_counts": body.get("status_counts"),
                    "m2_run_id": body.get("m2_run_id"),
                }
            last_err = f"unexpected response: status={res['http_status']} ok={body.get('ok')}"
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            return {
                "ok": False,
                "url": url,
                "http_status": exc.code,
                "error": body_text or str(exc),
            }
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.35)
    return {"ok": False, "url": url, "http_status": 0, "error": last_err or "timeout waiting for server"}


def _spawn_windowless_server(host: str, port: int, workspace: Path) -> int:
    workspace.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log_out = RUNTIME_DIR / f"server_{port}.stdout.log"
    log_err = RUNTIME_DIR / f"server_{port}.stderr.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["PYTHONUTF8"] = "1"

    cmd = [
        _resolve_pythonw(),
        "-B",
        "-m",
        "fractureflow.surface_product.workbench_server",
        "--host",
        host,
        "--port",
        str(port),
        "--workspace",
        str(workspace),
    ]
    with log_out.open("ab") as fh_out, log_err.open("ab") as fh_err:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fh_out,
            stderr=fh_err,
            shell=False,
            creationflags=CREATE_NO_WINDOW,
        )
    return int(proc.pid)


def _run_in_process_server(host: str, port: int, workspace: Path) -> int:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from fractureflow.surface_product.workbench_server import make_server

    workspace.mkdir(parents=True, exist_ok=True)
    server = make_server(host, port, workspace)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Windowless launcher for STRATA M5 3D Dual-Witness Demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Isolated port (default: 8855)")
    parser.add_argument("--cluster", default="jardim_c02", help="Initial cluster to open (default: jardim_c02)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open web browser")
    parser.add_argument("--serve-foreground", action="store_true", help="Run server in current process")
    parser.add_argument("--self-check", action="store_true", help="Start isolated server, verify readiness, then shut it down cleanly")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("REFUSED: STRATA demo launcher only binds loopback (127.0.0.1/localhost).")
    if args.port in FORBIDDEN_PORTS:
        raise SystemExit(f"REFUSED: port {args.port} is reserved; use an isolated port such as {DEFAULT_PORT}.")

    workspace = RUNTIME_DIR / f"workbench_home_{args.port}"
    demo_url = f"http://{args.host}:{args.port}/?mode=identifiability&cluster={args.cluster}"

    if args.serve_foreground:
        return _run_in_process_server(args.host, args.port, workspace)

    # Check if already running on target port
    already = wait_until_ready(args.host, args.port, timeout_s=1.2)
    pid: Optional[int] = None
    started_here = False
    if already.get("ok"):
        ready = already
    else:
        pid = _spawn_windowless_server(args.host, args.port, workspace)
        started_here = True
        ready = wait_until_ready(args.host, args.port, timeout_s=25.0)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "m5-strata-launcher-receipt/v1",
        "at_utc": _now_utc(),
        "host": args.host,
        "port": args.port,
        "demo_url": demo_url,
        "started_new_process": started_here,
        "pid": pid,
        "no_console_window": True,
        "creationflags": int(CREATE_NO_WINDOW),
        "readiness": ready,
    }
    (RUNTIME_DIR / "launcher_status.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if not ready.get("ok"):
        if started_here and pid is not None:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
        print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
        return 2

    if not args.no_browser and not args.self_check:
        webbrowser.open(demo_url)

    if args.self_check and started_here and pid is not None:
        # Cleanly terminate the self-check child process without spawning any console window
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(1, False, int(pid))
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
        except Exception:
            try:
                os.kill(pid, 15)
            except Exception:
                pass

    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
