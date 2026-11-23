# 【M5 交付报告】STRATA 三维工作台接入双见证切换、主动坍缩交互与 ADD_FRAGMENT 补齐

## 1. 结论

**PASS**（唯一判级权威见 `results/midterm_sprint_20261120/M5_strata_workbench_demo/result.json`）——在不触碰历史冻结文件及 `workbench_service.py`、`rule_solver.py`、`edit_engine.py` 前提下，已完成 6 个改前既有源码的逐字节归档与 SHA256 断言，实现真实派生域只读/校验后端 `identifiability_bridge.py` 与 `/api/identifiability/*` 路由、STRATA 三维工作台「结构面编辑台 / 贯通可辨识性与主动复核」双模式界面、绿红双见证一键切换与多步主动复核坍缩交互（兼容 M2 深度 2 `collapse_tree` / `next_step` 与实时精确引擎回退）、无窗隔离端口启动器 `launch_strata_demo.pyw`，并补齐 `#ed-action` 的 `ADD_FRAGMENT` 选项；在隔离端口 `8855` 实机运行全量 `tests/surface_product/` 回归（`290 passed, 1 skipped`）、`8` 项有牙后端 API 负控及 `0` 外部网络请求的 9 帧无头 Chrome 端到端冒烟，三项空分母指标 `field_error_rate`、`human_time`、`hydraulic` 严格保持 `null`。

## 2. 测量三件

- **口径范围**：本次交付仅覆盖 M1 冻结双场地真实露头条件有限多裂隙簇（`schema = "TC-network/real-conditioned/v1"`、`scope = "REAL_DERIVED_CONDITIONED_FINITE_UNION"`）在 STRATA 3D 工作台上的精确加载校验、双见证切换渲染、消费 M2 `runs/run_20261120_exact_v1/cases/*/active_check.json`（`collapse_tree` 深度 1/深度 2）的主动复核坍缩与重置展示，以及既有合成域编辑台 `ADD_FRAGMENT` 动作补齐。**不覆盖**地下体内无条件贯通真值、野外专家实测耗时、水力连通、或尚未落盘的 M3 真实 `AGREE/DISAGREE` 判卷内容（接口与界面均显式展示 `PENDING_M3`）；真实派生域与合成域 `SYNTHETIC_DECLARED_TRIANGLE_UNION` 永久分轨且硬性拦截跨目录混入。
- **枚举方式**：全量与典型场景分层结合——
  1. 后端接口与 SHA256/canonical_json 校验全量枚举 M1 `cluster_register.json` 全部 `22/22` 簇（Jardim Garcia `6` 簇 + Perozzo 2024 `16` 簇）及 M2 `run_20261120_exact_v1` 对齐证书（`1` 簇贯通、`14` 簇断割、`6` 簇双见证歧义、`1` 簇 `jardim_c01` 单列超预算拒收）；
  2. 既有回归全量枚举运行 `tests/surface_product`（`290 passed, 1 skipped`，跳过项为 Windows 环境 symlink 权限限制的 `tests/surface_product/test_w154_portable_project.py:447`）；
  3. 负控与端到端冒烟枚举 `5` 大类共 `10` 项实跑检查（后端 `M5-N1a/b`、`M5-N2a/b/c/d`、`M5-N3a/b` 共 `8` 项独立 HTTP 拒收，全量回归 `M5-N4` 1 项，无头 Chrome 端到端交互 `M5-N5` 1 项含坏 `ADD_FRAGMENT` HTTP 400 拒收与 9 张高分截图）。
- **测量时刻**：
  - 改前 6 份源码逐字节归档断言完成时刻：`2026-11-20T15:39:53.997276+00:00`（见 `archive_verification.json`）
  - 负控 `M5-N1..N5` 暨全量 `tests/surface_product/` 与无头 Chrome E2E 完整通过时刻：`2026-11-21T04:15:09.765347+00:00`（见 `negative_controls.json`）
  - 门禁 G1（`repro_certify`）与 G3（`r263_gain_mislabel_scan --scope delivery`）先期跑通时刻：`2026-11-21 12:16`（中国标准时间）；门禁 G4（`r303_draft_check --draft report.md --doc-class governance`）与末步错峰执行的门禁 G2（`fractureflow.selfcheck --quick`）实测时刻见 `run_receipt.json` 与 `result.json`。

## 3. 交付物清单

| 文件 | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `src/fractureflow/surface_product/identifiability_bridge.py` | 109729 | `e4349ad9083e1110dcc1bdfd4db1c00641e5302919d9c1a87b37387e0d918fac` | M1/M2/M3 真实露头桥接、SHA 核验、域隔离守卫、2 步 `collapse_tree` 与实时精确回退 |
| `src/fractureflow/surface_product/workbench_server.py` | 15436 | `f43bad9653bde3a585b665c06d3d7975a2286b217753cb0414f8f11a450a05db` | 新增 `/api/identifiability/{health,cases,case,apply_check}` 只读/校验路由（不改写既有事务） |
| `demos/workbench/index.html` | 3356 | `679cda033da5a07413734bf918f18e48f70a13efc107c4b5c655e531a14e7cb5` | `#ed-action` 补齐 `ADD_FRAGMENT`、新增 3D 双见证切换条 DOM 与零 CDN 离线静态壳 |
| `demos/workbench/core_api.js` | 51952 | `7a198eaab716647c96eeddf759931c3a351f6a781ce27ac8c2db8db150d54923` | 状态机扩展 `ident` 分支（双步 `identApplyCheck`/重置）及 `ADD_FRAGMENT` 双模态骨架 |
| `demos/workbench/plate.js` | 53952 | `c0b246cbf998fecc45c3f643b2ba913c1f37d8d11a860f3af55b975b7ccb18eb` | 零依赖 Canvas 3D 视口渲染 `endpoints A/B`、真孔洞、绿链、红白 `S/T` 断口、#1 琥珀金片 |
| `demos/workbench/app.js` | 141906 | `9f6076c725d9fb1b672a3f1ca9654a0d486105232b3bacb2c92400dc4d1f2541` | 双模式切换 Tab、22 簇清单、基线误报对照卡、1/2 步坍缩控制卡与底部学术边界条 |
| `demos/workbench/styles.css` | 67223 | `29cd35cfd975993f0a5443e4e25aeba172e4af9d7e82a8c96c2791c87b732a28` | 学术米白/墨绿配色系统、双见证按钮、防遮挡 `vp-witness-bar` 布局与单行顶部状态栏样式 |
| `M5_strata_workbench_demo/launch_strata_demo.py` | 8458 | `73553950cc31fd1590e215490cffc1d524d43993d783843499101971f819e290` | 隔离端口 `8855` 无窗一键启动与就绪自检 CLI（含禁用 `8791/8801/8810` 护栏） |
| `M5_strata_workbench_demo/launch_strata_demo.pyw` | 678 | `cdec01331672feccb207e1340be49b7502b91f5e517bff5c4ac030518614b401` | Windows `pythonw.exe` 零弹窗双击演示启动器入口 |
| `M5_strata_workbench_demo/smoke_verify_m5.py` | 33838 | `0df2c60c724b04174eff09b603234bbdf07fba52b134d219af5261e975aca5c7` | `M5-N1..N5` 自动化负控、全量 `tests/surface_product/` 回归与 Chrome E2E 验证脚本 |
| `M5_strata_workbench_demo/archive_verification.json` | 2613 | `b3e875d017b544982c5929169fc015cf0ee627473551a0a0cdf5949abf61d00f` | `_pre_edit_archive/` 下 6 份改前原文件逐字节与 SHA256 相等断言凭证 |
| `M5_strata_workbench_demo/negative_controls.json` | 8954 | `d010a0ad5c9911ad5d7285d80a906053293bf65401ed9a4e16c10ec3d40067f8` | 5 类 10 项负控与 9 张无头 Chrome 截图交互验证机器报告 |
| `M5_strata_workbench_demo/own_defects.json` | 5112 | `1570bd00efd3916e5db7faa9114f4a76d9c826c589fbefa7b023c58716a1b5c0` | 非空自有缺陷、过程中发现的问题与拒绝范围外改动记录 |
| `M5_strata_workbench_demo/screenshots/00_workbench_add_fragment.png` | 182646 | `c8de611ce10d19cd190de36440289cfe2771356e5f97693152af26b401e3d9ce` | 原结构面编辑台 `#ed-action=ADD_FRAGMENT` 提交 `v2` 实机截图 |
| `M5_strata_workbench_demo/screenshots/01_overview_jardim_c02_ambiguous.png` | 261751 | `c5b7e8ff55efae500ddba47402741a2046bd0c0e736137f21419cfc85fd60218` | `jardim_c02` 真实簇 3D 视口总览、基线误报对照与 Top-k 排序截图 |
| `M5_strata_workbench_demo/screenshots/02_witness_connected_emerald.png` | 265543 | `1f06a7f61d3bcbd1855fcbb699c5876b9fd1b0b3b178b6cfdc8a8959b830919b` | `jardim_c02` 切换翠绿连通见证 `(m+, t+)` 高亮 5 片 10 处真实接触 |
| `M5_strata_workbench_demo/screenshots/03_witness_disconnected_crimson.png` | 264867 | `7aa7b76239b6ef30e7f4ef10b3f5a61aead1b004c8051b40005d2555c9c39ee9` | `jardim_c02` 切换红橙断割见证 `(m-, t-)` 标注 `S/T` 割界与断口 |
| `M5_strata_workbench_demo/screenshots/04a_step1_narrowed_jardim_c02.png` | 271467 | `e9cdbc7919cf7e2dd6ddd241b88132a8fed668cc9ae3c3ddd71186ad990d6f1e` | `jardim_c02` Step 1 模态缩域并揭示 Step 2 `frag_306f0` 推荐面板 |
| `M5_strata_workbench_demo/screenshots/04_active_collapse_certified_disconnected.png` | 248710 | `94ba03eeb88bef2d6e185eaec60226ce6d4d4e843658d6f8cf9b40333e290334` | `jardim_c02` 两步主动复核后实时坍缩为 `CERTIFIED_DISCONNECTED` |
| `M5_strata_workbench_demo/screenshots/05_jardim_c04_holed_before_collapse.png` | 273557 | `617ccd9bc464939516b6720fc77a56fdaaba9fd3bed317cf0fc65d692e4f1dd4` | 含真实拓扑孔洞 `jardim_c04`（`保洞×1`）复核前琥珀金高亮状态 |
| `M5_strata_workbench_demo/screenshots/06_active_collapse_certified_connected.png` | 250848 | `0771372680171beab505cd26215d8d4a18dc6894e79b8da5836ee8239d7ab3d4` | `jardim_c04` 1 步复核实时坍缩为 `CERTIFIED_CONNECTED` 绿链状态 |
| `M5_strata_workbench_demo/screenshots/07_perozzo_c02_outcrop_overview.png` | 270244 | `d085f2ac20712e2133934b5880cc8cec2fb669bded883d0b2f470d535471fe07` | 切换 Perozzo 2024 场地清单并渲染 `perozzo_c02` 三维露头与推荐分枝 |
| `M5_strata_workbench_demo/gates/g1.json` | 1904 | `7d78a4ad96b28445f9d2d9c4b00e9f4f5ac6425d81e6e0fcb6eccc606499c30d` | 门禁 1（`scripts/repro_certify.py`，`9/9 PASS`，`exit_code=0`） |
| `M5_strata_workbench_demo/gates/g3.json` | 20648 | `e5c9276f8f25f0896fb0898eb673f3f7fc13ff8b3931d98b77c21e79d447d89b` | 门禁 3（`r263_gain_mislabel_scan.py --scope delivery`，`exit_code=0`） |
| `M5_strata_workbench_demo/gates/g2.json` / `gates/g4/` / `report.md` / `result.json` / `run_receipt.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 门禁 4（校验本件）、末步错峰门禁 2 及规范收口权威三件 |

## 4. 验收对照

| 任务书要求（`M5_STRATA三维工作台双见证与主动坍缩接入.md`） | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| §二 改前逐字节归档全部允许修改既有源码并当场断言 SHA256 相等；严禁触碰 `workbench_service.py` / `rule_solver.py` / `edit_engine.py` 与 `8791/8801/8810` | 在编辑前完成 6 个文件逐字节快照归档并验证 6/6 SHA256 与字节完全一致；禁止修改核心事务文件均保持零改动；全测试隔离于 `8855..8858` | `_pre_edit_archive/`、`archive_verification.json` | 满足 |
| §三.1 `GET /api/identifiability/cases`、`GET /case?id=`、`POST /apply_check` 实时核验 SHA256 一致性并返真实几何、基线对照与 1/2 步 `collapse_tree` | `identifiability_bridge.py` + `workbench_server.py` 实现全 22 簇真实 SHA256 重算比对、409/404/400 硬拒收、`PRECOMPUTED_VERIFIED_TREE` 与 `LIVE_EXACT_SOLVER` 显式来源及 2 步 `next_step` 坍缩回执 SHA | `identifiability_bridge.py`、`workbench_server.py`、`negative_controls.json` | 满足 |
| §三.2 STRATA 3D 工作台双模式 Tab、零外网离线运行、绿红双见证切换、#1 琥珀金临界片、主动坍缩、一键重置、底部固定学术边界条及 T-M `ADD_FRAGMENT` 修复 | 前端 5 件完成零 CDN 离线集成，实时切换名义态/绿链 `(m+, t+)` /红断 `(m-, t-)`，展示朴素并集/名义单一模型误报对比、支持 1 步/2 步坍缩与重置，固定学术边界条显式标 `null`，`#ed-action` 已支持合成/真点云双模态 `ADD_FRAGMENT` | `demos/workbench/*`、`screenshots/00..07*.png`、`negative_controls.json` | 满足 |
| §三.3 无窗一键启动器 `launch_strata_demo.pyw`（及 `.py` CLI），零 `.bat`/`.cmd`/`cmd.exe` 黑框 | `launch_strata_demo.pyw` / `launch_strata_demo.py` 全程采用 `CREATE_NO_WINDOW = 134217728` 与 `pythonw.exe`，实测 `--self-check --port 8855 --no-browser` 返回 `0` 且端口就绪 `readiness.ok=true` | `launch_strata_demo.pyw`、`launch_strata_demo.py`、`runtime/launcher_status.json` | 满足 |
| §三.4 >= 5 类有牙负控 + 既有 `tests/surface_product/` 零回归 + >= 4 张实机高清截图 + 0 外部网络请求 | 实现 `M5-N1`..`M5-N5` 共 10 项校验：8 项后端 HTTP 精准拒收、1 项全量回归（`290 passed, 1 skipped`）、1 项浏览器端到端冒烟（含坏 `ADD_FRAGMENT` HTTP 400 拒收、0 外部请求、0 控制台/页面报错、9 张截图） | `smoke_verify_m5.py`、`negative_controls.json`、`screenshots/*.png` | 满足 |
| §四 交付规范四件完整，四道门禁（含带 `--draft` 的 G4 与末步错峰 G2）原始退出码全 `0` | 落盘规范四件与非空 `own_defects.json`，逐一跑通 G1、G3、G4 与 G2，全部命令真实执行并登记原退出码与检索面计数 | `result.json`、`run_receipt.json`、`own_defects.json`、`gates/{g1.json,g2.json,g3.json,g4/}` | 满足 |

## 5. 负控实况

全部 8 项后端有牙负控均在临时复制的隔离目录 `_nc_scratch/` 与隔离端口 `8856..8858` 内实测，浏览器端坏 `ADD_FRAGMENT` 与 E2E 在隔离端口 `8855` 实测（详见 `negative_controls.json`）：

| 编号 | 本应失败的情形 | 实际退出码 | 是否真的失败 |
|---|---|---|---|
| `M5-N1a` | 篡改隔离副本 `jardim_c02/model.json` 1 个字节（保留原 `cluster_register.json` 与 M2 证书）后访问 `/case` | HTTP `409`（`ok=false`, `MODEL_SHA256_MISMATCH`） | 是 |
| `M5-N1b` | 篡改 `jardim_c02/model.json` 1 字节并同步改写 `cluster_register.json` 哈希，保留原 M2 `certification.json#input_sha256` | HTTP `409`（`ok=false`, `CERT_INPUT_SHA256_MISMATCH`） | 是 |
| `M5-N2a` | 请求不存在的簇 ID `nonexistent_cluster_999` | HTTP `404`（`ok=false`, `CLUSTER_NOT_FOUND`） | 是 |
| `M5-N2b` | 对 `jardim_c02` 提交非法分支 `branch_id=FAKE_ILLEGAL_BRANCH_999` 到 `/apply_check` | HTTP `400`（`ok=false`, `INVALID_BRANCH_ID`） | 是 |
| `M5-N2c` | 对已精确认证断割的 `perozzo_c01` 强行发起 `/apply_check` | HTTP `400`（`ok=false`, `INVALID_CHECK_TARGET`） | 是 |
| `M5-N2d` | 对两步坍缩分支提交非法 `step2_branch_id=FAKE_STEP2_BRANCH_999` | HTTP `400`（`ok=false`, `INVALID_BRANCH_ID`） | 是 |
| `M5-N3a` | 向真实 `clusters/` 目录投放 `scope=SYNTHETIC_DECLARED_TRIANGLE_UNION` 冒充文件后访问 `/cases` 与 `/case` | HTTP `409`（`ok=false`, `SCOPE_VIOLATION`） | 是 |
| `M5-N3b` | 将已登记真实簇 `jardim_c02` 的 `scope` 篡改为 `SYNTHETIC_DECLARED_TRIANGLE_UNION` | HTTP `409`（`ok=false`, `SCOPE_VIOLATION`） | 是 |
| `M5-N4` | 复跑全量既有 `tests/surface_product/` 验证未破坏 T-D 编辑/保存/导出契约 | Pytest 退出码 `0`（`290 passed, 1 skipped`） | 验证不破坏（正对照回归通过） |
| `M5-N5` | 在结构面编辑台对缺真实观测集 `synthetic_contract` 提交旧版 `{observation_id, indices}` `ADD_FRAGMENT` 载荷，并跑通双见证/2 步坍缩 E2E | 坏载荷 HTTP `400`（`BAD_TARGET: observation is missing: obs_synth`）；E2E `exit=0` | 坏编辑确实被拒，正常流程全通过 |

## 6. 自有缺陷与未做事项

- **没能验证的项**：
  1. M3 `verdict_batch.json` 在本单交付时刻尚未落盘（`m3_present=false`），页面当前实测了 `PENDING_M3` 显式占位展示，尚未联调真实 `AGREE/DISAGREE` 判卷分支；
  2. `tests/surface_product/` 在 Windows 下因系统符号链接权限限制存在 `1` 条环境跳过项 `test_w154_portable_project.py:447`（`290 passed, 1 skipped`）；
  3. 无头 Chrome 交互截图集中验证了典型代表簇 `jardim_c02`、`jardim_c04`（含保洞真拓扑）与 `perozzo_c02`，未对全量 `22` 簇逐一单独录屏截图；
  4. M2 包缺失时启用的 `LIVE_EXACT_SOLVER` 回退路径未在浏览器 UI 内人为清空 M2 目录压测端到端重算耗时（默认工作台联调消费 `PRECOMPUTED_VERIFIED_TREE`）。
  5. 三项空分母指标（`field_error_rate` 地下体内真值、`human_time` 真实耗时、`hydraulic` 水力连通）保持 `null`。
- **只是没看到问题、并非证明正确的项**：
  1. `M5-N1..N5` 与 `290 passed` 证明改动零破坏既有契约且 8 类负控全拒收，但不等价于极端退化共线百万面片输入的全边界完备性；
  2. M2 `collapse_tree` 仅预计算至深度 2（`next_step`），连续两步仍未决的次级分支将诚实停留在 `AMBIGUOUS` 态等待复位，未在第 3 步动态在线续搜；
  3. 离线零 CDN Canvas 2D 透视画家算法依赖三角质心排序，未采用 WebGL 深度缓冲；
  4. G3 与 G4 通过只表示登记扫描面与既定规则无违规命中。
- **明确没做的事**：未修改 `workbench_service.py`、`rule_solver.py`、`edit_engine.py`；未修改 M1/M2 或历史冻结文件；未越权代填 M3 判卷；未触碰已有运行端口（`8791/8801/8810`）；未运行全仓 `pytest`；未执行 `git commit`/`git push`；全程零弹窗。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| STRATA 3D 工作台已真实集成双场地 22 簇真实露头网络（SHA256 每次加载强核验、域隔离硬拒收），并在隔离端口实测完成绿/红双见证一键切换、#1 临界裂隙高亮、1/2 步主动坍缩与一键重置、学术边界条常驻及 `#ed-action=ADD_FRAGMENT` 修复 | 不可以说展示的几何认证等价于地下体内无条件贯通真值、实际工程水力连通或野外人工真实耗时（三者均为 `null`） |
| M5 全部代码改动均已在改前逐字节归档并校验哈希一致，既有 `tests/surface_product/` 复测 `290 passed, 1 skipped`（零回归），5 类共 10 项负控与 9 帧离线浏览器 E2E 全部有牙通过，且实现了 `CREATE_NO_WINDOW` 零黑框启停 | 不可以说 `verdict_batch.json` 尚未落盘时前端 `PENDING_M3` 占位已经完成真实 M3 独立判卷结果展示 |
| 全工作台在无外网环境下（0 外部 CDN 请求）可于答辩现场通过 `launch_strata_demo.pyw` 一键起用，四道现役门禁与规范交付件完整对应 | 不可以把真实派生域 `REAL_DERIVED_CONDITIONED_FINITE_UNION` 与合成域 `SYNTHETIC_DECLARED_TRIANGLE_UNION` 混在同一统计栏位 |

交接：请架构师验收 M5，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M5_strata_workbench_demo，判级见 result.json#status。
