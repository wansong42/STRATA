# -*- coding: utf-8 -*-
"""Windowless (.pyw) one-click launcher for STRATA M5 3D Defense Demo.

Double-clicking or invoking via pythonw.exe starts STRATA on isolated port
127.0.0.1:8855 with zero console/cmd.exe windows (CREATE_NO_WINDOW=134217728),
verifies /api/identifiability/cases SHA256 readiness, and opens the browser to
the 3D Dual-Witness & Active Collapse workspace.
"""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from launch_strata_demo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
