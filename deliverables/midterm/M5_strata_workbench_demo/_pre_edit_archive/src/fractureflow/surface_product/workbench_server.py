# -*- coding: utf-8 -*-
"""Loopback-only HTTP server for the PF05 same-origin workbench."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .contracts import ContractError, VersionConflictError, canonical_json
from .edit_engine import PreviewTokenError
from .workbench_service import MAX_IMPORT_BYTES, ROOT, WorkbenchService

STATIC_ROOT = ROOT / "demos" / "workbench"


def _envelope(path: str, result: Any) -> Dict[str, Any]:
    """One response contract for every /api/* POST.

    ``ok`` is computed from an explicit, route-specific receipt -- never from
    the mere absence of an exception.  A save without ``saved=true`` and a
    verified receipt is a failure; an export without a non-empty
    enumeration is a failure.
    """
    ok = True
    extra: Dict[str, Any] = {}
    if isinstance(result, dict):
        if path == "/api/projects/save":
            receipt = result.get("receipt") or {}
            ok = bool(result.get("saved")) and bool(receipt.get("verified"))
            extra = {"saved": bool(result.get("saved")), "dirty": bool(result.get("dirty", True)),
                     "receipt_present": bool(receipt), "bundle_id": result.get("bundle_id")}
        elif path == "/api/export":
            readback = result.get("readback") or {}
            ok = (bool(result.get("exported")) and bool(readback.get("verified"))
                  and int(result.get("enumeration_count") or 0) > 0)
            extra = {"exported": bool(result.get("exported")),
                     "enumeration_count": int(result.get("enumeration_count") or 0),
                     "verified": bool(readback.get("verified"))}
        elif path == "/api/portable/export":
            # Export is only "ok" when the package verifies and reopens cleanly.
            opened = result.get("open") or {}
            ok = bool(result.get("exported")) and bool(opened.get("ok"))
            extra = {"exported": bool(result.get("exported")),
                     "package": result.get("package"),
                     "package_status": opened.get("status")}
        elif path == "/api/portable/open":
            ok = bool(result.get("ok"))
            extra = {"package_status": result.get("status")}
        elif path == "/api/portable/relocate":
            relocated = result.get("relocated") or {}
            opened = result.get("open") or {}
            ok = bool(relocated.get("ok")) and bool(opened.get("ok"))
            extra = {"hash_relocate": relocated.get("hash_relocate"),
                     "package_status": opened.get("status")}
        elif result.get("saved") is False or result.get("exported") is False:
            ok = False
    return {"ok": ok, "result": result, **extra}


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple, handler: type, service: WorkbenchService) -> None:
        self.service = service
        super().__init__(address, handler)

    def server_close(self) -> None:
        try:
            self.service.shutdown()
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: WorkbenchHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        line = "%s %s\n" % (self.log_date_time_string(), fmt % args)
        path = self.server.service.workspace / "server.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _send_bytes(self, status: int, data: bytes, content_type: str, *, download_name: Optional[str] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # Plotly 2.35 injects its own <style> and calls sheet.insertRule; a
        # style-src without 'unsafe-inline' leaves sheet === null and aborts
        # the whole vendor bundle (measured: "Cannot read properties of null
        # (reading 'insertRule')", window.Plotly stays undefined).  Only the
        # style directive is relaxed: scripts remain same-origin only, no
        # 'unsafe-eval', no runtime CDN, no remote connect.
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        if download_name:
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % download_name)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, value: Any, status: int = 200) -> None:
        data = canonical_json(value).encode("utf-8")
        self._send_bytes(status, data, "application/json; charset=utf-8")

    def _error(self, exc: BaseException) -> None:
        status = 409 if isinstance(exc, (VersionConflictError, PreviewTokenError)) else 400
        if isinstance(exc, FileNotFoundError):
            status = 404
        self._json({
            "ok": False,
            "error": {
                "code": getattr(exc, "code", "WORKBENCH_ERROR"),
                "message": str(exc), "type": type(exc).__name__,
            },
        }, status)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_IMPORT_BYTES:
            raise ContractError("request body exceeds the 12 MiB boundary", "RESOURCE_LIMIT_EXCEEDED")
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("request body is not valid UTF-8 JSON", "PROJECT_JSON_INVALID") from exc
        if not isinstance(value, dict):
            raise ContractError("request body must be a JSON object", "OBJECT_REQUIRED")
        return value

    def _query(self, key: str) -> str:
        values = parse_qs(urlparse(self.path).query).get(key) or []
        if not values:
            raise ContractError("missing query parameter: " + key, "QUERY_REQUIRED")
        return values[0]

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            service = self.server.service
            if path == "/api/health":
                return self._json({
                    "ok": True, "service": "PF05_DEVELOPMENT", "loopback_only": True,
                    "runtime_cdn": False, "solver": "PF04_SHARED_RULE_SOLVER",
                    "model_registered": False, "project_open": service.state().get("project_open", False),
                })
            if path == "/api/catalog":
                return self._json(service.catalog())
            if path == "/api/state":
                return self._json(service.state())
            if path == "/api/library":
                return self._json({"library": service.library()})
            if path == "/api/scene":
                return self._json(service.scene())
            if path == "/api/object":
                return self._json(service.object_detail(self._query("id")))
            if path == "/api/evidence":
                return self._json(service.evidence(self._query("id")))
            if path == "/api/jobs":
                return self._json({"jobs": service.jobs.list(public=True)})
            if path == "/api/portable/status":
                return self._json(service.portable_status())
            if path.startswith("/api/jobs/"):
                return self._json(service.jobs.get(unquote(path.rsplit("/", 1)[-1]), public=True))
            if path.startswith("/download/"):
                parts = path.split("/")
                if len(parts) != 4:
                    raise FileNotFoundError(path)
                file_path = service.download_path(unquote(parts[2]), unquote(parts[3]))
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                if file_path.suffix in (".json", ".csv"):
                    content_type += "; charset=utf-8"
                return self._send_bytes(200, file_path.read_bytes(), content_type, download_name=file_path.name)
            return self._static(path)
        except Exception as exc:
            self._error(exc)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in ("", "/") else unquote(path.lstrip("/"))
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            raise FileNotFoundError(path)
        if not candidate.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix in (".html", ".css", ".js"):
            content_type += "; charset=utf-8"
        self._send_bytes(200, candidate.read_bytes(), content_type)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            body = self._body()
            service = self.server.service
            routes = {
                "/api/projects/open": lambda: service.open_catalog(str(body.get("catalog_id") or "")),
                "/api/projects/import": lambda: service.import_project(body.get("project"), declared_bytes=int(body.get("declared_bytes") or 0)),
                "/api/projects/migrate": lambda: service.migrate_current(),
                "/api/projects/close": lambda: service.close_project(),
                "/api/projects/open-saved": lambda: service.open_saved(str(body.get("bundle_id") or "")),
                "/api/projects/save": lambda: service.save_current(interrupt=body.get("interrupt")),
                "/api/edit/preview": lambda: service.preview(body),
                "/api/edit/commit": lambda: service.commit(body),
                "/api/edit/cancel": lambda: service.cancel_preview(body),
                "/api/edit/undo": lambda: service.undo(int(body.get("base_version"))),
                "/api/edit/redo": lambda: service.redo(int(body.get("base_version"))),
                "/api/solve/start": lambda: service.start_solve(body),
                "/api/export": lambda: service.export_current(),
                "/api/portable/export": lambda: service.portable_export(
                    distribution_mode=str(body.get("distribution_mode") or "REFERENCE_ONLY")),
                "/api/portable/open": lambda: service.portable_open(body.get("package")),
                "/api/portable/relocate": lambda: service.portable_relocate(
                    body.get("package"), body.get("asset_id"), body.get("new_path")),
                "/api/portable/close": lambda: service.portable_close(),
            }
            if path.startswith("/api/solve/cancel/"):
                return self._json({"ok": True, "result": service.jobs.cancel(unquote(path.rsplit("/", 1)[-1]))})
            if path.startswith("/api/solve/apply/"):
                # Match the /api/* route contract: the front end reads
                # payload.result.update. Returning the raw apply_job dict here
                # made every browser solve fail with "Cannot read properties of
                # undefined (reading 'update')" even when the job succeeded.
                return self._json({"ok": True, "result": service.apply_job(
                    unquote(path.rsplit("/", 1)[-1]), int(body.get("base_version")))})
            handler = routes.get(path)
            if handler is None:
                raise FileNotFoundError(path)
            result = handler()
            self._json(_envelope(path, result))
        except Exception as exc:
            self._error(exc)


def make_server(host: str, port: int, workspace: os.PathLike[str] | str,
                dataset_registry: Optional[os.PathLike[str] | str] = None) -> WorkbenchHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ContractError("PF05 server is loopback-only", "UNSAFE_BIND")
    service = WorkbenchService(workspace, dataset_registry=dataset_registry)
    return WorkbenchHTTPServer((host, port), Handler, service)


DEFAULT_DATASET_REGISTRY = ROOT / "results" / "w152_general_import" / "catalog" / "datasets.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", default=str(ROOT / "results" / "product_first_plan_20261106" / "runs" / "PF05" / "workbench_home"))
    parser.add_argument("--dataset-registry",
                        default=os.environ.get("FRACTUREFLOW_DATASET_REGISTRY")
                        or str(DEFAULT_DATASET_REGISTRY),
                        help="W152 dataset registry; new datasets need no source edit")
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port, args.workspace,
                         dataset_registry=args.dataset_registry)
    print("PF05 workbench: http://%s:%d" % (args.host, server.server_address[1]), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
