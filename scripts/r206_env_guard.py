# -*- coding: utf-8 -*-
"""R206-T3 · 环境毒丸（坑位 43）—— 解释器 + 数值库版本锚。

问题（R198 已实证、R203 终验再次踩中、R206 本单 R203 §0.3 第三次复现）:
  PATH 上的 `python` 是 base Anaconda（numpy 2.5.0 / scipy 1.18.0），而本项目
  一律使用 pytorch_env（numpy 2.2.5 / scipy 1.15.3）。位级复现类断言在 base 上
  必然假红（R200 亲历：`test_v_r140_::test_t2_probe_rerun_bitwise` max_abs_dev
  1e-15 级），而假红会被误读成"代码坏了"→ 触发无谓返工。

处置（本单 T3）: 把"跑错解释器"从**纪律**变成**构造上不可能**——
终验入口 / 守卫入口在开工前先断言:
  ① `sys.executable` 归一化后 == pytorch_env 解释器;
  ② `numpy.__version__ == 2.2.5`;
  ③ `scipy.__version__ == 1.15.3`。
任一不符 → `SystemExit`（硬停，退出码 4），且理由里带坑位号与修复命令。

用法:
  python scripts/r206_env_guard.py            # 自检: 合规 → rc 0; 不合规 → rc 4
  python scripts/r206_env_guard.py --json     # 打印环境签名 (JSON)
  from r206_env_guard import assert_pinned_env   # 终验/守卫入口第一行调用
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PIT_ID = 43
PIT_NAME = "解释器/数值库毒丸"

PINNED_PYTHON = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
PINNED_NUMPY = "2.2.5"
PINNED_SCIPY = "1.15.3"
KNOWN_BAD_PYTHON = "E:/app2/anacoda/Anacoda/python.exe"
KNOWN_BAD_VERSIONS = {"numpy": "2.5.0", "scipy": "1.18.0"}

EXIT_RC = 4


def _norm(p):
    try:
        return os.path.normcase(os.path.realpath(str(p)))
    except OSError:
        return os.path.normcase(os.path.abspath(str(p)))


def _version(mod_name):
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, "__version__", "?"))
    except Exception as exc:              # 缺包也算"不符"（禁静默放行）
        return "IMPORT_FAILED:%s" % type(exc).__name__


def env_signature():
    """环境指纹（纯读，供证据件与守卫直读）。"""
    exe = sys.executable
    np_v = _version("numpy")
    sp_v = _version("scipy")
    return dict(
        unit="R206", pit=PIT_ID, pit_name=PIT_NAME,
        executable=exe,
        executable_norm=_norm(exe),
        pinned_python=PINNED_PYTHON,
        pinned_python_norm=_norm(PINNED_PYTHON),
        executable_ok=(_norm(exe) == _norm(PINNED_PYTHON)),
        numpy=np_v, numpy_anchor=PINNED_NUMPY, numpy_ok=(np_v == PINNED_NUMPY),
        scipy=sp_v, scipy_anchor=PINNED_SCIPY, scipy_ok=(sp_v == PINNED_SCIPY),
        python_version="%d.%d.%d" % sys.version_info[:3],
        known_bad_python=KNOWN_BAD_PYTHON,
        known_bad_versions=KNOWN_BAD_VERSIONS,
    )


def violations(sig=None):
    """返回违规项清单（空 = 合规）。"""
    sig = sig or env_signature()
    bad = []
    if not sig["executable_ok"]:
        bad.append("sys.executable=%s 不是钉死解释器 %s" % (sig["executable"], PINNED_PYTHON))
    if not sig["numpy_ok"]:
        bad.append("numpy=%s != %s" % (sig["numpy"], PINNED_NUMPY))
    if not sig["scipy_ok"]:
        bad.append("scipy=%s != %s" % (sig["scipy"], PINNED_SCIPY))
    return bad


def pit_message(bad, context=""):
    return (
        "坑 %d (%s) 触发%s: %s\n"
        "  → 本项目的位级复现/终验**必须在钉死环境下跑**, 否则产出 1e-15 级假红。\n"
        "  → 修复: $env:TZ=''; $env:PYTHONUTF8='1'; "
        "%s -m pytest <suite> -q\n"
        "  → 反例实证: %s (base Anaconda, numpy %s / scipy %s) 跑 r140 位级断言假红。"
        % (PIT_ID, PIT_NAME, (" @ " + context) if context else "", "; ".join(bad),
           PINNED_PYTHON, KNOWN_BAD_PYTHON, KNOWN_BAD_VERSIONS["numpy"],
           KNOWN_BAD_VERSIONS["scipy"]))


def assert_pinned_env(context="", quiet=False):
    """终验/守卫入口的开跑前断言: 不合规即 SystemExit(4)。"""
    bad = violations()
    if bad:
        raise SystemExit(pit_message(bad, context))
    if not quiet:
        sys.stderr.write("[R206-T3] 坑 %d 通过: %s | numpy %s / scipy %s\n"
                         % (PIT_ID, os.path.basename(sys.executable),
                            PINNED_NUMPY, PINNED_SCIPY))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="打印环境签名")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    sig = env_signature()
    if a.json:
        print(json.dumps(sig, ensure_ascii=False, indent=2, allow_nan=False))
    bad = violations(sig)
    if bad:
        sys.stderr.write(pit_message(bad, "r206_env_guard CLI") + "\n")
        return EXIT_RC
    if not a.quiet:
        print("坑 %d PASS: %s | numpy %s | scipy %s"
              % (PIT_ID, sig["executable"], sig["numpy"], sig["scipy"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
