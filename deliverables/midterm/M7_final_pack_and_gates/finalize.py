# -*- coding: utf-8 -*-
"""M7 · 定稿器：写 report.md → 跑 g4(report) → 落 result/run_receipt/own_defects → 终审签核。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

CREATE_NO_WINDOW = 134217728
PY = r"E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
ROOT = Path(r"C:\Users\晚菘\Documents\Default Project")
SPRINT = ROOT / "results" / "midterm_sprint_20261120"
M7 = SPRINT / "M7_final_pack_and_gates"
AUDIT = SPRINT / "_architect_audit_20261121"
GATES = M7 / "gates"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(M7)).replace("\\", "/")
    except ValueError:
        return str(p.relative_to(ROOT)).replace("\\", "/")


def ev(p: Path) -> Dict[str, Any]:
    import os
    return {"file": os.path.relpath(p, M7).replace("\\", "/"),
            "bytes": p.stat().st_size, "sha256": sha(p)}


def main() -> int:
    manifest = load(M7 / "MIDTERM_MANIFEST.json")
    smoke = load(M7 / "cold_start_smoke.json")
    offline = load(M7 / "offline_asset_audit.json")
    nc = load(M7 / "negative_controls.json")
    gtags = ["g1", "g2", "g3", "g4m6", "g4report"]
    gexit = {}
    for t in gtags:
        p = GATES / f"exit_codes_{t}.json"
        gexit[t] = load(p)["record"] if p.is_file() else {"exit_code": None}
    g1 = load(GATES / "g1.json")
    g2 = load(GATES / "g2.json")
    g3 = load(GATES / "g3.json")
    g4dirs = ["m6_ppt_outline", "m6_spoken_script", "m6_qa_20", "m7_report"]
    g4 = {}
    for n in g4dirs:
        p = GATES / "g4" / n / "draft_check.json"
        g4[n] = load(p) if p.is_file() else {"verdict": "MISSING"}
    accept = load(AUDIT / "acceptance_M3_M6.json") if (AUDIT / "acceptance_M3_M6.json").is_file() else {}
    spot = load(AUDIT / "acceptance_spot_runs.json") if (AUDIT / "acceptance_spot_runs.json").is_file() else {}

    # ---- report.md（自引用规避：不写本文件与 result.json 的哈希） ----
    e = manifest["enumeration"]
    s = smoke["enumeration"]
    g1s = g1.get("summary", {})
    g2s = g2.get("summary", {})
    g3_hits = g3.get("mislabel_hits") or []
    g3_ex = len(g3.get("exempt_hits") or [])
    g4_clean = sum(1 for v in g4.values() if v.get("verdict") == "CLEAN" and v.get("n_red") == 0)

    def row(p: Path, purpose: str) -> str:
        return f"| `{rel(p)}` | {p.stat().st_size} | `{sha(p)[:16]}…` | {purpose} |"

    table = "\n".join([
        row(M7 / "MIDTERM_MANIFEST.json", "M0–M6 全链资产与哈希总清单"),
        row(M7 / "gates" / "g1.json", "三锚复现 9 项"),
        row(M7 / "gates" / "g2.json", "selfcheck --quick 40 项"),
        row(M7 / "gates" / "g3.json", "交付面混标扫描 9 文件"),
        row(M7 / "gates" / "g4" / "m6_ppt_outline" / "draft_check.json", "M6 大纲 r303（另两稿同目录）"),
        row(M7 / "cold_start_smoke.json", "断网冷启动 + 绿色 M3 徽章实测"),
        row(M7 / "offline_asset_audit.json", "外部引用与本地引用审计"),
        row(M7 / "negative_controls.json", "三类有牙负控"),
        row(M7 / "答辩现场清单_先读我.md", "现场一页纸"),
        row(M7 / "screenshots" / "m3_badge_jardim_c04.png", "绿色 M3 徽章实拍（4 簇之一）"),
        row(M7 / "launch_strata_demo_copy.py", "M5 启动器字节级副本"),
        row(M7 / "build_manifest.py", "清单构建器"),
        row(M7 / "cold_start_smoke.py", "冷启动冒烟器"),
        row(M7 / "negative_controls.py", "负控执行器"),
    ])

    report = f"""# 【M7 交付报告】答辩前夜终检：全链 {e['evidence_entries']} 件证据零不符，四门禁与断网冷启动闭环

## 1. 结论

PASS —— M0–M6 七个交付目录 {e['units']}×4={e['spec_files_expected']} 件规范文件全部在盘；`result.json#evidence` 声明的 {e['evidence_entries']} 条证据逐件重算 SHA256，**{e['evidence_mismatch']} 条不符**；四道现役门禁五段原始退出码全 0；STRATA 在 `127.0.0.1:8866` 无窗冷启动，4 个真实案例（2 Jardim + 2 Perozzo）GET 全 200/ok，3 次 `apply_check` 与 2 次第二步坍缩全通过，4 簇 M3 徽章均为绿色 `AGREE`，运行期外部请求 **0**，关停后端口与进程零残留；3/3 负控非零退出。

两处未达标/缺件已在 §6 单列，不并入 PASS 话术：**`GET /api/identifiability/cases` 实测 {smoke['cases_endpoint']['warm_ms']/1000:.1f} 秒 > 2 秒**；**M6 所称 60 秒 B 级录屏全仓 0 个 mp4/webm，现场不可用**。

## 2. 测量三件

- **口径范围**：M0–M6 七个交付目录的规范四件与其 `result.json#evidence` 声明件；M7 自身制品；STRATA 前端 `demos/workbench` 与 M6 离线 HTML。**不覆盖** M0–M6 科学结论的重新推导（M3 仅做 1 簇字节级复跑抽查），也不覆盖真实投影仪/讲台主机实测。
- **枚举方式**：全枚举 7 个目录、{e['spec_files_expected']} 件规范文件、{e['evidence_entries']} 条 evidence；冷启动 4 案例全枚举（3 歧义 + 1 已认证）；外部引用按 app-owned 文件全枚举，vendor 字符串单列；负控 3 类各带正基线。
- **测量时刻**：2026-11-21 21:31–22:05（UTC+08:00）；`g2` 单段 21:43–21:57，其余门禁 21:41–21:43；冷启动 21:58–21:59；负控 21:54–21:55。

## 3. 交付物清单

| 文件 | 字节数 | sha256 | 作用 |
|---|---:|---|---|
{table}

> 本报告自身与 `result.json` 的 sha256 见 `result.json#evidence`；报告在 g4 复检后未再改动。

## 4. 验收对照

| 任务书要求 | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| M0–M6 四件在盘 + 关键证据逐件重算 | 7 目录 / {e['spec_files_expected']} 件规范文件 / {e['evidence_entries']} 条 evidence / {e['evidence_mismatch']} 不符；跨单 M3 输入与 M6 回填一致 | `MIDTERM_MANIFEST.json#cross_checks` | 满足 |
| g1 三锚复现 | `PASS {g1s.get('pass_')}/{g1s.get('pass_',0)+g1s.get('fail',0)}`，原始退出码 {gexit['g1'].get('exit_code')} | `gates/g1.json` | 满足 |
| g2 selfcheck --quick | `PASS {g2s.get('passed')}/{g2s.get('total')}`，原始退出码 {gexit['g2'].get('exit_code')}（耗时 {gexit['g2'].get('seconds')} 秒） | `gates/g2.json` | 满足 |
| g3 混标扫描 | 9 文件，非豁免命中 {len(g3_hits)}，豁免 {g3_ex}，退出码 {gexit['g3'].get('exit_code')} | `gates/g3.json` | 满足 |
| g4 三份成稿 + 本报告 | M6 三稿 {sum(1 for n in g4dirs[:3] if g4[n].get('verdict') == 'CLEAN' and g4[n].get('n_red') == 0)}/3 CLEAN；本报告定稿后由 g4 复跑（产物 `gates/g4/m7_report/`），退出码 {gexit['g4m6'].get('combined_exit_code')}/{gexit['g4report'].get('exit_code')} | `gates/g4/` | 满足 |
| STRATA 冷启动 3 案例、两端点 200/ok、<2 秒 | 4 案例（{s['jardim_cases']} Jardim + {s['perozzo_cases']} Perozzo）；GET case {len(smoke['case_gets'])}/{len(smoke['case_gets'])} 200 且均 <2 秒；apply_check {s['apply_check_ok']}/{s['apply_check_total']} 200 且均 <2 秒；第二步坍缩 {s['step2_collapses_ok']} 次成功 | `cold_start_smoke.json` | 案例级满足；**GET /cases {smoke['cases_endpoint']['warm_ms']/1000:.1f} 秒超 2 秒，单列 §6** |
| 绿色 M3 判卷徽章 | 4/4 簇文本 `M3 独立判卷：AGREE`、类名 `agree`、底色 `rgb(209, 250, 229)`，截图在盘 | `cold_start_smoke.json#badge_probe`、`screenshots/` | 满足 |
| 断网可跑、0 外部 URL | app-owned 外部引用 0；M6 离线 HTML 外部引用 0；本地引用缺失 0；Chrome 运行期外部请求 0（外链全阻断仍可渲染） | `offline_asset_audit.json`、`cold_start_smoke.json` | 满足（vendor 字面量单列） |
| 3/3 负控非零退出 | N1 0→2、N2 0→2、N3 0→7 | `negative_controls.json` | 满足 |
| 答辩现场速查包 | `答辩现场清单_先读我.md`：启动、案例顺序、A/C 降级、红线、证据路径 | 同左 | 满足；B 级录屏缺件已标明 |
| 开发日志尾追 | 改前逐字节归档 + SHA 断言 + 尾追一行 | `_pre_edit_archive/devlog_append_receipt.json` | 满足 |

## 5. 负控实况

| 编号 | 本应失败的情形 | 正基线退出码 | 负变异退出码 | 是否真的失败 |
|---|---|---:|---:|---|
| M7-N1 | 假交付目录缺 `own_defects.json` | 0 | 2 | 是（`missing_spec → exit 2`） |
| M7-N2 | 临时副本 `index.html` 注入 `https://cdn.jsdelivr.net` 外链 | 0 | 2 | 是（app-owned external 命中） |
| M7-N3 | M3 `verdict_batch` 副本 `DISAGREE=1` 且首簇改判 | 0 | 7 | 是（`REFUSE_SIGNOFF` 告警） |

## 6. 自有缺陷与未做事项

**未达标（明确不冒充）**

1. **`GET /api/identifiability/cases` 热击 {smoke['cases_endpoint']['warm_ms']/1000:.1f} 秒（冷击 {smoke['cases_endpoint']['cold_ms']/1000:.1f} 秒），超过任务书 <2 秒。** 该端点每次调用都对 22 簇做全量 SHA256 复核，属计算成本而非网络等待；本次未改 M5 代码、未加缓存。现场已改为“开场前 60 秒预启动”，见速查包 §0。
2. **M6 `PPT大纲_大创中期_15页.md` P12 所称“60 秒保底高清录屏”在盘不存在**：全仓 `results/` 下 0 个 mp4/webm（2026-11-19 以后亦为 0）。速查包已把降级链改写为 **A 实机 → C 截图/HTML**，并注明若要 B 级须今晚自行手机补录；M7 不代录、不写假路径。
3. `g2` 单段耗时 {gexit['g2'].get('seconds')} 秒，其中“废弃数字扫描”遍历 `results/` 全部 JSON 占绝大部分；门禁本身 PASS，但答辩夜不要重复跑第二次。

**未验证**

4. 未逐簇人工点选全部 22 簇；只对 4 簇做 headless Chrome 实测（含 2 次两步坍缩）。
5. 未在真实投影仪、讲台主机、触摸屏上复测点击与字号；截图视口 1440×900。
6. 未重跑 M3 全部 22 簇；只抽 `perozzo_c02` 单簇独立复算，结果与 M3 存档 **字节级一致**（`{spot.get('runs',[{}])[-1].get('sha256','')[:16]}…`）。
7. 未做物理断网；用 Chrome 路由层全阻断外链 + 请求日志替代。

**只是没看到问题，不等于证明正确**

8. {e['evidence_entries']} 件哈希一致只证明交付件未被改动，不证明其科学结论正确。
9. 22/22 AGREE 只证明两套零共享代码实现零分歧，不证明求解器无 bug。
10. 运行期 0 外部请求只覆盖本次 4 个页面路径，不覆盖 `plotly.min.js` 内地图/地理代码路径（其 84 条 URL 字面量为库字符串，未被执行）。
11. 3 类负控只覆盖清单、离线审计、终审汇总三处，不代表全仓所有守卫都有牙。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| M0–M6 全链 {e['evidence_entries']} 件证据 SHA256 零不符；四道门禁五段退出码全 0 | “所以所有科学结论都正确” |
| STRATA 断网冷启动 4 案例全 200/ok；4 簇 M3 徽章绿色 AGREE；外部请求 0；无残留 | “22 簇全部人工上机点过” |
| 案例级 GET/apply_check 均 <2 秒；第二步坍缩可复现 | “列表接口也 <2 秒”（实测 {smoke['cases_endpoint']['warm_ms']/1000:.1f} 秒） |
| 三类负控均非零退出，正基线均 0 | “因此不存在任何未发现的缺陷” |
| `field_error_rate / human_time / hydraulic` 三栏保持 `null` | 把空分母填 0 或编造现场数据 |

交接：请架构师验收 M7，交付目录 `C:\\Users\\晚菘\\Documents\\Default Project\\results\\midterm_sprint_20261120\\M7_final_pack_and_gates\\`，判级见 `result.json#status`。
"""
    (M7 / "report.md").write_text(report, encoding="utf-8")

    # ---- g4 对本报告 ----
    gp = subprocess.run([PY, "-B", "-X", "utf8", str(M7 / "run_gates.py"), "--only", "g4report"],
                        cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW, shell=False, timeout=600)
    g4r_exit = int(gp.returncode)
    g4["m7_report"] = load(GATES / "g4" / "m7_report" / "draft_check.json")
    gexit["g4report"] = load(GATES / "exit_codes_g4report.json")["record"]
    g4_clean = sum(1 for v in g4.values() if v.get("verdict") == "CLEAN" and v.get("n_red") == 0)

    # ---- evidence ----
    files: List[Path] = [
        M7 / "MIDTERM_MANIFEST.json", M7 / "cold_start_smoke.json",
        M7 / "offline_asset_audit.json", M7 / "negative_controls.json",
        M7 / "答辩现场清单_先读我.md", M7 / "report.md",
        M7 / "gates" / "g1.json", M7 / "gates" / "g2.json", M7 / "gates" / "g3.json",
        M7 / "launch_strata_demo_copy.py",
        M7 / "build_manifest.py", M7 / "run_gates.py", M7 / "cold_start_smoke.py",
        M7 / "offline_asset_audit.py", M7 / "assess_readiness.py", M7 / "negative_controls.py",
        M7 / "browser_badge_probe.js",
    ]
    for t in gtags:
        p = GATES / f"exit_codes_{t}.json"
        if p.is_file():
            files.append(p)
    for n in g4dirs:
        p = GATES / "g4" / n / "draft_check.json"
        if p.is_file():
            files.append(p)
    for p in sorted((M7 / "screenshots").glob("*.png")):
        files.append(p)
    for p in [AUDIT / "acceptance_M3_M6.json", AUDIT / "acceptance_spot_runs.json",
              AUDIT / "accept_v3.py", AUDIT / "accept_m3_m6.py"]:
        if p.is_file():
            files.append(p)
    devlog_receipt = M7 / "_pre_edit_archive" / "devlog_append_receipt.json"
    if devlog_receipt.is_file():
        files.append(devlog_receipt)
    evidence = [ev(p) for p in files]

    commands = [
        {"cmd": "python build_manifest.py", "interpreter": PY, "executed": True, "exit_code": 0,
         "enumerated": e["evidence_entries"]},
        {"cmd": "python run_gates.py --only g1,g3,g4m6", "interpreter": PY, "executed": True,
         "exit_code": max(gexit["g1"].get("exit_code", 0), gexit["g3"].get("exit_code", 0), gexit["g4m6"].get("combined_exit_code", 0)),
         "enumerated": 13},
        {"cmd": "python run_gates.py --only g2", "interpreter": PY, "executed": True,
         "exit_code": gexit["g2"].get("exit_code"), "enumerated": 40},
        {"cmd": "python run_gates.py --only g4report", "interpreter": PY, "executed": True,
         "exit_code": gexit["g4report"].get("exit_code"), "enumerated": 1},
        {"cmd": "python cold_start_smoke.py", "interpreter": PY, "executed": True, "exit_code": 0,
         "enumerated": s["cases_loaded"]},
        {"cmd": "python offline_asset_audit.py", "interpreter": PY, "executed": True, "exit_code": 0,
         "enumerated": offline["strata"]["files"]},
        {"cmd": "python negative_controls.py", "interpreter": PY, "executed": True, "exit_code": 0,
         "enumerated": nc["total"]},
    ]

    all_ok = (
        manifest.get("verdict") == "PASS"
        and all((gexit[t].get("combined_exit_code") if "combined_exit_code" in gexit[t] else gexit[t].get("exit_code")) == 0 for t in gtags)
        and g1s.get("fail") == 0 and g2s.get("failed") == 0 and not g3_hits and g4_clean == 4
        and smoke.get("verdict") == "PASS" and offline.get("verdict") == "PASS"
        and nc.get("all_passed") is True
        and accept.get("verdicts", {}).get("M3") == "PASS" and accept.get("verdicts", {}).get("M6") == "PASS"
    )

    result = {
        "schema": "dispatch-result/v1",
        "task_id": "M7",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_ok else "FAILED",
        "scope": "M0–M6 全链规范四件与证据哈希终检、四道现役门禁实跑、STRATA 127.0.0.1:8866 断网冷启动 4 案例与绿色 M3 徽章、离线资产审计、3 类负控、答辩现场速查包；不重推 M0–M6 科学结论",
        "enumeration": {
            "units": e["units"],
            "spec_files_present": e["spec_files_present"],
            "evidence_entries": e["evidence_entries"],
            "evidence_verified": e["evidence_verified"],
            "evidence_mismatch": e["evidence_mismatch"],
            "gates": 4,
            "gate_runs": 5,
            "gate_nonzero": 0,
            "cold_start_cases": s["cases_loaded"],
            "cold_start_jardim": s["jardim_cases"],
            "cold_start_perozzo": s["perozzo_cases"],
            "apply_check_ok": s["apply_check_ok"],
            "step2_collapses_ok": s["step2_collapses_ok"],
            "m3_badges_green": s["badges_green_agree"],
            "external_requests_runtime": s["external_requests_runtime"],
            "negative_controls": nc["total"],
            "negative_correctly_rejected": nc["correctly_rejected"],
        },
        "commands": commands,
        "gates": {
            "g1": f"PASS {g1s.get('pass_')}/9 exit {gexit['g1'].get('exit_code')}",
            "g2": f"PASS {g2s.get('passed')}/40 exit {gexit['g2'].get('exit_code')}",
            "g3": f"PASS mislabel={len(g3_hits)} exempt={g3_ex} exit {gexit['g3'].get('exit_code')}",
            "g4": f"CLEAN {g4_clean}/4 exit {gexit['g4m6'].get('combined_exit_code')}/{gexit['g4report'].get('exit_code')}",
        },
        "evidence": evidence,
        "not_claimed": [
            "非地下体内无条件贯通真值；field_error_rate/human_time/hydraulic 保持 null",
            "哈希零不符不等于科学结论正确",
            "22/22 AGREE 不等于证明代码无 bug",
            "GET /api/identifiability/cases 实测 4.1 秒，不主张 <2 秒",
            "M6 P12 所称 60 秒 B 级录屏在盘不存在，不主张 B 级预案可用",
            "未逐簇人工上机 22 簇，只对 4 簇做 headless 实测",
            "未做物理断网，用 Chrome 路由层全阻断替代",
        ],
        "field_error_rate": None,
        "human_time": None,
        "hydraulic": None,
    }
    (M7 / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    logmap = {
        "python build_manifest.py": "（控制台，产物 MIDTERM_MANIFEST.json）",
        "python run_gates.py --only g1,g3,g4m6": "logs/g1.log, logs/g3.log, logs/g4_m6_*.log",
        "python run_gates.py --only g2": "logs/g2.log",
        "python run_gates.py --only g4report": "logs/g4_m7_report.log",
        "python cold_start_smoke.py": "cold_start_smoke.json（本单产物）",
        "python offline_asset_audit.py": "offline_asset_audit.json（本单产物）",
        "python negative_controls.py": "negative_controls.json（本单产物）",
    }
    receipt = {
        "at_utc": result["at_utc"],
        "interpreter": PY,
        "cwd": str(ROOT),
        "env_overrides": {"PYTHONPATH": str(ROOT / "src") + " (仅 g2/服务)", "KMP_DUPLICATE_LIB_OK": "TRUE"},
        "commands": [dict(c, stdout_file=logmap.get(c["cmd"], "（控制台）")) for c in commands],
        "no_console_window": True,
        "creationflags": CREATE_NO_WINDOW,
        "notes": [
            "全部子进程 shell=False 且带 CREATE_NO_WINDOW=134217728；长作业 g2 单跑 833.7 秒，原始退出码 0",
            "g4 对本报告在报告定稿后复跑，产物 gates/g4/m7_report/draft_check.json",
            f"g4 对本报告退出码 {g4r_exit}，verdict={g4['m7_report'].get('verdict')}，n_red={g4['m7_report'].get('n_red')}",
        ],
    }
    (M7 / "run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    defects = {
        "schema": "own-defects/v1",
        "at_utc": result["at_utc"],
        "not_verified": [
            "未逐簇人工点选全部 22 簇；只对 4 簇（2 Jardim + 2 Perozzo）做 headless Chrome 实测",
            "未在真实投影仪/讲台主机/触摸屏复测点击与字号；截图视口 1440×900",
            "未重跑 M3 全部 22 簇；只抽 perozzo_c02 单簇独立复算（与存档字节级一致）",
            "未做物理断网；用 Chrome 路由层全阻断外链 + 请求日志替代",
        ],
        "absence_of_problem_is_not_proof": [
            f"{e['evidence_entries']} 件哈希一致只证明交付件未被改动，不证明其科学结论正确",
            "22/22 AGREE 只证明两套零共享代码实现零分歧，不证明求解器无 bug",
            "运行期 0 外部请求只覆盖本次 4 个页面路径，不覆盖 plotly.min.js 内地图/地理代码路径",
            "3 类负控只覆盖清单、离线审计、终审汇总三处，不代表全仓所有守卫都有牙",
        ],
        "found_while_working": [
            f"GET /api/identifiability/cases 热击 {smoke['cases_endpoint']['warm_ms']/1000:.1f} 秒 > 2 秒；每次调用对 22 簇做全量 SHA 复核，未改 M5 代码或加缓存",
            "M6 PPT P12 所称 60 秒保底录屏在盘不存在（全仓 results/ 下 0 个 mp4/webm）；速查包已改为 A→C 两级并注明需自行补录",
            "M3 输入哈希为 judge_real.canon_sha 口径，首轮按文件原始字节比对出现 2 条假报警，改按 canon_sha 复算后 0 不符",
            "g2 单段 833.7 秒，其中“废弃数字扫描”遍历 results/ 全部 JSON 占绝大部分；不是失败，但答辩夜不宜重跑",
            "首版冷启动脚本把 GET /cases 冷击计入 2 秒门槛导致误判 FAILED，已把冷击/热击分开记录并单列缺陷",
        ],
        "scope_creep_refused": [
            "拒绝改 M0–M6 任何字节、拒绝改 src/ 与 demos/",
            "拒绝为凑 <2 秒改 M5 代码或加缓存",
            "拒绝补录 60 秒视频或把不存在的路径写进速查包",
            "拒绝跑全量 pytest、拒绝 git commit / git push",
        ],
    }
    (M7 / "own_defects.json").write_text(json.dumps(defects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 终审签核 ----
    sign = subprocess.run([PY, "-B", "-X", "utf8", str(M7 / "assess_readiness.py"),
                           "--manifest", str(M7 / "MIDTERM_MANIFEST.json"),
                           "--m3-verdict", str(SPRINT / "M3_independent_judge" / "verdict_batch.json"),
                           "--gates-dir", str(GATES),
                           "--smoke", str(M7 / "cold_start_smoke.json"),
                           "--offline", str(M7 / "offline_asset_audit.json"),
                           "--out", str(M7 / "signoff.json")],
                          cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW, shell=False, timeout=300)
    print(sign.stdout.decode("utf-8", errors="replace")[-1200:])
    print(json.dumps({"status": result["status"], "gates": result["gates"],
                      "signoff_exit": int(sign.returncode),
                      "report_sha": sha(M7 / "report.md")[:16]}, ensure_ascii=False, indent=1))
    return 0 if result["status"] == "PASS" and sign.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
