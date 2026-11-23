"""R322 公共件：哈希 / 严格 UTF-8 / 原子写 / 读数。

零第三方依赖（stdlib only）。所有落盘件走 dump_json（allow_nan=False + 原子替换 +
落盘前断言严格 UTF-8 且 NUL=0），对应 S31 / 军规「产物严格 UTF-8」。
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "r322_final_settlement")

PINNED = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
NNS_PREFIX = "NNS-"


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_text(rel_or_abs):
    p = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(ROOT, rel_or_abs)
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def read_bytes(rel_or_abs):
    p = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(ROOT, rel_or_abs)
    with open(p, "rb") as f:
        return f.read()


def ufffd_count(text):
    return text.count(chr(0xFFFD))


def assert_env_pinned():
    """坑 43：入口断言解释器与 numpy/scipy 版本，不符即 SystemExit。"""
    exe = os.path.normcase(os.path.abspath(sys.executable))
    if exe != os.path.normcase(os.path.abspath(PINNED.replace("/", os.sep))):
        raise SystemExit("坑 43 违规: sys.executable=%s != %s" % (sys.executable, PINNED))
    import numpy
    import scipy
    if (numpy.__version__, scipy.__version__) != ("2.2.5", "1.15.3"):
        raise SystemExit("坑 43 违规: numpy=%s scipy=%s" % (numpy.__version__, scipy.__version__))
    return {"python": sys.executable, "numpy": numpy.__version__, "scipy": scipy.__version__}


def dump_json(path, obj):
    """严格 UTF-8 + allow_nan=False + 原子写；写后即读回校验。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    blob = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False)
    raw = blob.encode("utf-8")
    _assert_clean(raw, path)
    fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    with open(path, "rb") as f:
        back = f.read()
    if back != raw:
        raise SystemExit("写后回读不一致: %s" % path)
    return {"path": _rel(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def write_text_atomic(path, text):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    raw = text.encode("utf-8")
    _assert_clean(raw, path)
    fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return {"path": _rel(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _assert_clean(raw, path):
    if b"\x00" in raw:
        raise SystemExit("产物含 NUL: %s" % path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit("产物非严格 UTF-8: %s (%s)" % (path, e))
    if chr(0xFFFD) in text:
        raise SystemExit("产物含 U+FFFD 替换符: %s" % path)


def _rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def df_c():
    """C 盘读数（GiB + 十进制 G 两口径并报，坑 100：计数须报口径）。"""
    st = os.statvfs("C:/") if hasattr(os, "statvfs") else None
    if st is None:
        import shutil
        t = shutil.disk_usage("C:/")
        free_b, total_b, used_b = t.free, t.total, t.used
    else:
        free_b = st.f_bavail * st.f_frsize
        total_b = st.f_blocks * st.f_frsize
        used_b = total_b - free_b
    gib = 1024.0 ** 3
    return {
        "total_G_decimal": round(total_b / 1e9, 1),
        "used_G_decimal": round(used_b / 1e9, 1),
        "avail_G_decimal": round(free_b / 1e9, 1),
        "use_pct": round(100.0 * used_b / total_b, 1),
        "avail_GiB_binary": round(free_b / gib, 2),
        "method": "os.statvfs(C:/) f_bavail" if st is not None else "shutil.disk_usage",
    }


FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_reparse_or_link(abspath):
    """junction / symlink 一律不进遍历：冷迁位点在 E:，走进去既慢又非本仓面。"""
    try:
        st = os.stat(abspath, follow_symlinks=False)
    except OSError:
        return True
    if getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    return os.path.islink(abspath)


def walk_face_files():
    """仓面 = ROOT 下全部普通文件（排除 .git / __pycache__ / junction·symlink）。

    相对路径按字符串累加（禁 os.path.relpath：盘上有 `\\\\.\\NUL` 之类
    特殊命名件会让 relpath 直接 ValueError）。
    """
    skip_dirs = {".git", "__pycache__"}
    root_prefix = os.path.abspath(ROOT) + os.sep
    for dirpath, dirnames, filenames in os.walk(root_prefix, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs
                       and not _is_reparse_or_link(os.path.join(dirpath, d))]
        for fn in filenames:
            ap = os.path.join(dirpath, fn)
            if _is_reparse_or_link(ap):
                continue
            yield ap[len(root_prefix):].replace(os.sep, "/")


def classify_citations(text, root=None, exempt=()):
    """把文中反引号路径分四类：文件 / 目录 / glob 模式 / 测试节点。

    坑 72 要求"陈述指向在盘制品"，但把 glob 与节点 id 当单件文件断言会造成
    **构造性假失败**（本单 T6 首跑即触发）。分类后各自按各自的方式核。
    """
    import re
    root = root or ROOT
    toks = sorted(set(re.findall(r"`((?:docs|scripts|results|tasks|tests|src)/[^`\s]+)`", text)))
    out = {"files": [], "dirs": [], "patterns": [], "nodes": [], "exempted": [],
           "missing": []}
    for t in toks:
        if t in exempt or t.rstrip("/") in exempt:
            out["exempted"].append(t)
            continue
        if "*" in t or "?" in t:
            out["patterns"].append(t)
            continue
        if "::" in t:
            f = t.split("::")[0]
            out["nodes"].append(t)
            if not os.path.isfile(os.path.join(root, f.replace("/", os.sep))):
                out["missing"].append(t)
            continue
        ap = os.path.join(root, t.replace("/", os.sep))
        if t.endswith("/"):
            out["dirs"].append(t)
            if not os.path.isdir(ap):
                out["missing"].append(t)
        else:
            out["files"].append(t)
            if not os.path.isfile(ap):
                out["missing"].append(t)
    out["n_cited"] = len(toks)
    out["all_present"] = not out["missing"]
    return out


if __name__ == "__main__":
    print(json.dumps(assert_env_pinned(), ensure_ascii=False, allow_nan=False))
    print(json.dumps(df_c(), ensure_ascii=False, allow_nan=False))
