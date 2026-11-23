# 【M7 交付报告】答辩前夜终检：全链 216 件证据零不符，四门禁与断网冷启动闭环

## 1. 结论

PASS —— M0–M6 七个交付目录 7×4=28 件规范文件全部在盘；`result.json#evidence` 声明的 216 条证据逐件重算 SHA256，**0 条不符**；四道现役门禁五段原始退出码全 0；STRATA 在 `127.0.0.1:8866` 无窗冷启动，4 个真实案例（2 Jardim + 2 Perozzo）GET 全 200/ok，3 次 `apply_check` 与 2 次第二步坍缩全通过，4 簇 M3 徽章均为绿色 `AGREE`，运行期外部请求 **0**，关停后端口与进程零残留；3/3 负控非零退出。

两处未达标/缺件已在 §6 单列，不并入 PASS 话术：**`GET /api/identifiability/cases` 实测 4.2 秒 > 2 秒**；**M6 所称 60 秒 B 级录屏全仓 0 个 mp4/webm，现场不可用**。

## 2. 测量三件

- **口径范围**：M0–M6 七个交付目录的规范四件与其 `result.json#evidence` 声明件；M7 自身制品；STRATA 前端 `demos/workbench` 与 M6 离线 HTML。**不覆盖** M0–M6 科学结论的重新推导（M3 仅做 1 簇字节级复跑抽查），也不覆盖真实投影仪/讲台主机实测。
- **枚举方式**：全枚举 7 个目录、28 件规范文件、216 条 evidence；冷启动 4 案例全枚举（3 歧义 + 1 已认证）；外部引用按 app-owned 文件全枚举，vendor 字符串单列；负控 3 类各带正基线。
- **测量时刻**：2026-11-21 21:31–22:05（UTC+08:00）；`g2` 单段 21:43–21:57，其余门禁 21:41–21:43；冷启动 21:58–21:59；负控 21:54–21:55。

## 3. 交付物清单

| 文件 | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `MIDTERM_MANIFEST.json` | 14136 | `0dbb783ede45a03e…` | M0–M6 全链资产与哈希总清单 |
| `gates/g1.json` | 1904 | `37d7f80fbb999a7d…` | 三锚复现 9 项 |
| `gates/g2.json` | 10720 | `6c7c382ab4c77184…` | selfcheck --quick 40 项 |
| `gates/g3.json` | 20558 | `a353ae2f312ec410…` | 交付面混标扫描 9 文件 |
| `gates/g4/m6_ppt_outline/draft_check.json` | 2991 | `3a554438c93e20bd…` | M6 大纲 r303（另两稿同目录） |
| `cold_start_smoke.json` | 11744 | `bbbd843a3e75ca9e…` | 断网冷启动 + 绿色 M3 徽章实测 |
| `offline_asset_audit.json` | 15896 | `fdeb131249b10041…` | 外部引用与本地引用审计 |
| `negative_controls.json` | 6244 | `c9fcc49c92f08a53…` | 三类有牙负控 |
| `答辩现场清单_先读我.md` | 6731 | `345c613c315fafe6…` | 现场一页纸 |
| `screenshots/m3_badge_jardim_c04.png` | 266671 | `9ff9cbf17f23832c…` | 绿色 M3 徽章实拍（4 簇之一） |
| `launch_strata_demo_copy.py` | 8458 | `73553950cc31fd15…` | M5 启动器字节级副本 |
| `build_manifest.py` | 10282 | `015c930c5373ee15…` | 清单构建器 |
| `cold_start_smoke.py` | 21009 | `fb93b92cb1ea532b…` | 冷启动冒烟器 |
| `negative_controls.py` | 6839 | `80ba1b71642617d3…` | 负控执行器 |

> 本报告自身与 `result.json` 的 sha256 见 `result.json#evidence`；报告在 g4 复检后未再改动。

## 4. 验收对照

| 任务书要求 | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| M0–M6 四件在盘 + 关键证据逐件重算 | 7 目录 / 28 件规范文件 / 216 条 evidence / 0 不符；跨单 M3 输入与 M6 回填一致 | `MIDTERM_MANIFEST.json#cross_checks` | 满足 |
| g1 三锚复现 | `PASS 9/9`，原始退出码 0 | `gates/g1.json` | 满足 |
| g2 selfcheck --quick | `PASS 40/40`，原始退出码 0（耗时 833.66 秒） | `gates/g2.json` | 满足 |
| g3 混标扫描 | 9 文件，非豁免命中 0，豁免 8，退出码 0 | `gates/g3.json` | 满足 |
| g4 三份成稿 + 本报告 | M6 三稿 3/3 CLEAN；本报告定稿后由 g4 复跑（产物 `gates/g4/m7_report/`），退出码 0/0 | `gates/g4/` | 满足 |
| STRATA 冷启动 3 案例、两端点 200/ok、<2 秒 | 4 案例（2 Jardim + 2 Perozzo）；GET case 4/4 200 且均 <2 秒；apply_check 3/3 200 且均 <2 秒；第二步坍缩 2 次成功 | `cold_start_smoke.json` | 案例级满足；**GET /cases 4.2 秒超 2 秒，单列 §6** |
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

1. **`GET /api/identifiability/cases` 热击 4.2 秒（冷击 4.2 秒），超过任务书 <2 秒。** 该端点每次调用都对 22 簇做全量 SHA256 复核，属计算成本而非网络等待；本次未改 M5 代码、未加缓存。现场已改为“开场前 60 秒预启动”，见速查包 §0。
2. **M6 `PPT大纲_大创中期_15页.md` P12 所称“60 秒保底高清录屏”在盘不存在**：全仓 `results/` 下 0 个 mp4/webm（2026-11-19 以后亦为 0）。速查包已把降级链改写为 **A 实机 → C 截图/HTML**，并注明若要 B 级须今晚自行手机补录；M7 不代录、不写假路径。
3. `g2` 单段耗时 833.66 秒，其中“废弃数字扫描”遍历 `results/` 全部 JSON 占绝大部分；门禁本身 PASS，但答辩夜不要重复跑第二次。

**未验证**

4. 未逐簇人工点选全部 22 簇；只对 4 簇做 headless Chrome 实测（含 2 次两步坍缩）。
5. 未在真实投影仪、讲台主机、触摸屏上复测点击与字号；截图视口 1440×900。
6. 未重跑 M3 全部 22 簇；只抽 `perozzo_c02` 单簇独立复算，结果与 M3 存档 **字节级一致**（`a33fdf0cd4da1f84…`）。
7. 未做物理断网；用 Chrome 路由层全阻断外链 + 请求日志替代。

**只是没看到问题，不等于证明正确**

8. 216 件哈希一致只证明交付件未被改动，不证明其科学结论正确。
9. 22/22 AGREE 只证明两套零共享代码实现零分歧，不证明求解器无 bug。
10. 运行期 0 外部请求只覆盖本次 4 个页面路径，不覆盖 `plotly.min.js` 内地图/地理代码路径（其 84 条 URL 字面量为库字符串，未被执行）。
11. 3 类负控只覆盖清单、离线审计、终审汇总三处，不代表全仓所有守卫都有牙。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| M0–M6 全链 216 件证据 SHA256 零不符；四道门禁五段退出码全 0 | “所以所有科学结论都正确” |
| STRATA 断网冷启动 4 案例全 200/ok；4 簇 M3 徽章绿色 AGREE；外部请求 0；无残留 | “22 簇全部人工上机点过” |
| 案例级 GET/apply_check 均 <2 秒；第二步坍缩可复现 | “列表接口也 <2 秒”（实测 4.2 秒） |
| 三类负控均非零退出，正基线均 0 | “因此不存在任何未发现的缺陷” |
| `field_error_rate / human_time / hydraulic` 三栏保持 `null` | 把空分母填 0 或编造现场数据 |

交接：请架构师验收 M7，交付目录 `C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M7_final_pack_and_gates\`，判级见 `result.json#status`。
