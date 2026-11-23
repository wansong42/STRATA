# 【M2 交付报告】双场地 22 簇真实露头精确整网认证与「主动下一步复核排序器 (Next-Best-Check)」突破

## 1. 结论

**PASS**（唯一判级权威见 `results/midterm_sprint_20261120/M2_active_identifiability/result.json`）——在零修改历史冻结目录与零修改 M1 交付件的前提下，完成了 M1 冻结双场地真实露头全量 `22` 簇（Jardim Garcia `6` 簇 + Perozzo 2024 `16` 簇）的精确有理整网认证（`solve_real_clusters.py`）、零泄漏主动下一步复核排序与多步坍缩树构建（`active_identifiability.py`）以及 `4` 策略（含额外面积启发式对照共 `5` 臂）同分母反事实闭环仿真（`simulate_active_loop.py`）。`22/22` 簇 100% 完成处理与 M1 最终 SHA256 对齐（无静默跳过）：其中 `21` 簇完成完整精确认证（`CERTIFIED_CONNECTED` `1` 簇 [`4.76%`]、`CERTIFIED_DISCONNECTED` `14` 簇 [`66.67%`]、`AMBIGUOUS_WITH_VERIFIED_WITNESSES` `6` 簇 [`28.57%`，覆盖 Jardim `5` 簇 + Perozzo `1` 簇]），`1` 簇（`jardim_c01`，需 `5024` 细胞超过声明上限 `4096`）严格输出 `BUDGET_EXCEEDED` 单列拒收信封且绝不混入贯通分母。对全部 `6` 例 `AMBIGUOUS` 真实簇产出精确 `active_check.json` 与深度 2 `collapse_tree`，零泄漏仿真显示 `OURS_MEASURE_SPLIT` 均值 `2.052` 步（`1` 步坍缩 `57.02%`、`3` 步 `85.38%`）、`OURS_MINIMAX_EXACT` 均值 `2.296` 步（`4` 步 `97.80%`、`5` 步 `100.00%` 封顶），显著优于 `BASELINE_RANDOM`（`20` 种子均值 `4.651` 步）与几何启发式 `BASELINE_HEURISTIC_SIZE_OR_DIST`（`4.918` 步）及 `BASELINE_HEURISTIC_MAX_AREA`（`3.994` 步）。`7` 类共 `10/10` 项有牙负控 100% 非零拒收；四道现役门禁原始退出码全为 `0`；空分母指标 `field_error_rate`、`human_time`、`hydraulic` 均保持 `null`。

## 2. 测量三件

- **口径范围**：本单覆盖 M1 冻结的真实露头派生条件有限多裂隙簇（`schema = "TC-network/real-conditioned/v1"`、`scope = "REAL_DERIVED_CONDITIONED_FINITE_UNION"`）上的精确有理整网三态认证、稳定裕度 `ρ` 与最近翻转见证、零泄漏主动下一步复核排序（`RESOLVE_MODE_FEATURE` 与 `REFINE_PARAMETER_BAND`）、供 M3 复核与 M5 交互的 `collapse_tree`、多步反事实坍缩仿真，以及 `10` 项独立子进程负控与 M1 终版 SHA 对齐审计。**不覆盖**异模型独立盲判（属 M3）、出版级图表绘制（属 M4）、STRATA 前端 3D 集成（属 M5）、地下体内无条件贯通真值、人工实测耗时或水力连通，且与合成域 `SYNTHETIC_DECLARED_TRIANGLE_UNION` 永久分轨。
- **枚举方式**：全量枚举——消费 M1 `cluster_register.json` 全量 `22/22` 真实簇（Jardim `6` 簇 + Perozzo `16` 簇），每簇 `6` 离散模式全覆盖；`21` 簇完整认证总计枚举精确片对接触窗口 `124560` 对、精确 `POINT/OPEN` 细胞 `10512` 个，`1` 簇（`jardim_c01`）在模式 `m4` 触发声明 `4096` 细胞预算并诚实拒收；主动排序与多步闭环仿真全枚举覆盖全部 `6/6` 例 `AMBIGUOUS` 簇（共 `6045` 个原子细胞、精确勒贝格测度守恒逐树断言），`BASELINE_RANDOM` 采用 `20` 个固定确定性种子（`0..19`）全枚举平均；原始冻结 `TC_network/compile_network.py` 抽样全等价比对覆盖 `2` 簇（`perozzo_c05` 与 `perozzo_c02`，共 `9882` 窗口与 `446` 细胞）；负控全量执行 `7` 类 `10/10` 独立无窗子进程。权威在盘索引见 `results/midterm_sprint_20261120/M2_active_identifiability/runs/run_20261120_exact_v1/summary.json`、`verification_report.json` 与 `negative_controls.json`。
- **测量时刻**：
  - M1 上游终版交付与冻结时刻：`2026-11-21T03:17:41.076870+00:00`（`M1_real_to_exact_bridge/result.json`，`status = PASS`）
  - 22 簇精确整网认证运行完成时刻：`2026-11-20T15:56:43+00:00`（耗时 `135.226 s`）
  - 主动复核排序、深度 2 `collapse_tree` 与 `4` 策略闭环仿真完成时刻：`2026-11-21T03:31:56.401297+00:00`（见 `runs/run_20261120_exact_v1/summary.json`）
  - 全量验证、从零重算、`TC_network` 等价审计、`10/10` 负控与 M1 最终 SHA 对齐完成时刻：`2026-11-21T03:34:54.803976+00:00`（见 `verification_report.json`）

## 3. 交付物清单

| 文件（相对 `M2_active_identifiability/`） | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `solve_real_clusters.py` | 77569 | `4ad3dafe9265fa3f9b1b6257047e54c97232daf5e55af18e1f8c2f4976e2832d` | 真实派生整网精确有理认证引擎、局部自检与 `CellLatticeIndex` 精确不变量索引 |
| `active_identifiability.py` | 62619 | `34f104bcde4df5a457c876763e5f3db4b1d5f2982f6d1dbdcb629e2ad8657989` | 零泄漏主动下一步复核排序器、见证差分归因与深度 2 `collapse_tree` 构建/一致性守卫 |
| `simulate_active_loop.py` | 28353 | `fb48dfbc07636e24b69c7d6e3193bcc590cc1a9c37c489a17d236e3fae5407ad` | 零泄漏多步反事实信念树仿真（`OURS_MINIMAX_EXACT`、`OURS_MEASURE_SPLIT`、`BASELINE_RANDOM`、`BASELINE_HEURISTIC_*`） |
| `verify_m2.py` | 43126 | `3c01b5770f9c4703af84b0a0e1e20ff552788597ea68a068193747dc084aa9f6` | 全量复验、`summary.json` 装配、`TC_network` 原始等价审计、M1 SHA 对齐与 `10` 项负控 |
| `runs/run_20261120_exact_v1/summary.json` | 247524 | `3a2cf6bf392251f1d5c61830adf466cb2b3aeca6ef48a54beaac84e9c5f33546` | 不可变运行总表（22 簇三态计数、稳定裕度分布、基线诊断与主动坍缩曲线总表） |
| `runs/run_20261120_exact_v1/simulation_detail.json` | 212060 | `98f7d5465e72807674fc2b73d09cc3dad479aa43f3df2f1968ba8f37de578020` | 6 例 `AMBIGUOUS` 簇逐簇与分场地 5 策略精确有理坍缩曲线明细 |
| `runs/run_20261120_exact_v1/cases/<cid>/{certification,active_check}.json`（44 份） | 132957251（合计） | 逐件登记于 `result.json#evidence` | 全部 22 簇精确证书与主动复核/`collapse_tree` 不可变产物 |
| `negative_controls.json` | 11428 | `bc8379ca73de749d311e22e83199c54e96359603ae28893a860137e863b70d0b` | 7 类共 10 项独立无窗子进程有牙负控实测记录 |
| `verification_report.json` | 12177 | `1a5b7c103986fbd337fdcb92184e45a39fdf94b386098cadd8e9a46f6936d6b6` | 22/22 正例复验、M1 终版 SHA 对齐、原始 `TC_network` 等价审计与零 `__pycache__` 报告 |
| `own_defects.json` | 5924 | `9f2c5ea4f27d0820d18332fea902b8980139c88d2629e2d18b565fa73b5d4cc4` | 非空自有缺陷、过程中发现问题与拒绝范围外改动台账 |
| `gates/g1.json` | 1904 | `27a49c744a34accbda7f0cc93d7ebf41c313d3b17d0cf705d7895f24bf34ee74` | 门禁 1（`scripts/repro_certify.py`，`9/9 PASS`，`exit_code=0`） |
| `gates/g2.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 门禁 2（`fractureflow.selfcheck --quick`，最后一步单独错峰实跑，`exit_code=0`） |
| `gates/g3.json` | 20648 | `87031a4e37872392207c4d38f653ed51434bb91387343237f1cab2f0f520f562` | 门禁 3（`scripts/r263_gain_mislabel_scan.py --scope delivery --fail-on-hit`，`files_scanned=9, non_exempt_total=0`，`exit_code=0`） |
| `gates/g4/` | 见 `result.json#evidence` | 见 `result.json#evidence` | 门禁 4（`scripts/r303_draft_check.py --draft report.md --doc-class governance --json-only`） |
| `report.md` / `result.json` / `run_receipt.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 规范报告、机器判级权威件与运行回执（最终哈希统一登记于 `result.json#evidence`） |

## 4. 验收对照

| 任务书要求（`M2_精确整网认证与主动下一步复核算法.md`） | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| §3.1 消费 M1 全量真实簇 `model.json`（先过 `real_network_format.validate()`、记 `input_sha256`），复用冻结精确有理片对投影与 `POINT/OPEN` 普查，零浮点最终判定，显式区分四类状态且拒收信封绝不混入贯通分母 | 22/22 簇 100% 通过 `real_network_format.validate(verify_source_sha=True)` 与 M1 终版 SHA 对齐；`21` 簇完整认证（`1` 贯通、`14` 断割、`6` 双见证不可辨识，含精确路径 3D 接触点见证与精确 `ρ` 及 `attained` 标记），`jardim_c01` 严格按 `budget.cells=4096` 输出 `BUDGET_EXCEEDED` 单列拒收信封；与未修改 `TC_network/compile_network.py` 抽样比特级一致 | `solve_real_clusters.py`、`runs/run_20261120_exact_v1/summary.json`、`runs/run_20261120_exact_v1/cases/*/certification.json`、`verification_report.json` | 满足 |
| §3.2 面向全部 `AMBIGUOUS` 簇设计零泄漏可观测动作空间（`RESOLVE_MODE_FEATURE` 与 `REFINE_PARAMETER_BAND`）、精确重算评分（`certified_branch_fraction`、`resolved_measure_fraction`、零先验 Minimax `worst_case_remaining_ambiguity`）与人类可读几何归因 `witness_diff_role` | 全部 `6/6` 例 `AMBIGUOUS` 簇（Jardim `c02..c06` + Perozzo `c02`）产出完整精确排序与中英文归因（`jardim_c03` 首推 1 步 `100%` 坍缩、`jardim_c04` 含洞片首推 1 步消歧 `91.05%`、`jardim_c02/c05/c06` 与 `perozzo_c02` 首推 Minimax 消歧 `5/9`–`2/3`）；15 例已认证簇严格输出 `ALREADY_CERTIFIED` 空推荐，`jardim_c01` 输出 `NOT_APPLICABLE_REFUSED` | `active_identifiability.py`、`runs/run_20261120_exact_v1/cases/*/active_check.json` | 满足 |
| §3.2 无泄漏多步反事实闭环仿真，同分母对比 4 策略（`OURS_MINIMAX_EXACT`、`OURS_MEASURE_SPLIT`、`BASELINE_RANDOM`、`BASELINE_HEURISTIC_SIZE_OR_DIST`），如实记录实测值不调参 | 采用零泄漏信念域决策树全枚举全部 `6` 簇 `6045` 细胞并严格断言勒贝格测度守恒；实测均值：`OURS_MEASURE_SPLIT` `2.052` 步、`OURS_MINIMAX_EXACT` `2.296` 步（`5` 步 `100%` 坍缩）、`BASELINE_RANDOM`（`20` 种子）`4.651` 步、`BASELINE_HEURISTIC_SIZE_OR_DIST` `4.918` 步（并附 `BASELINE_HEURISTIC_MAX_AREA` `3.994` 步对照） | `simulate_active_loop.py`、`runs/run_20261120_exact_v1/summary.json`、`runs/run_20261120_exact_v1/simulation_detail.json` | 满足 |
| §3.3 不可变运行目录输出 `summary.json`、`cases/<cid>/{certification.json, active_check.json}`，含供 M5 零卡顿交互与 M3 独立验算的 `collapse_tree`（至少 Top-3 裂隙各分支完整重算证书与 3D 高亮 ID） | 全部 `6` 例 `AMBIGUOUS` 簇的 `collapse_tree` 均物化全部有效裂隙深度 1（含精确 `condition`、真实重算证书、3D 接触点与 `highlight_3d`），且对 Top-3 裂隙未决分支展开至深度 2 `next_step`；经 `verify_m2.py` 全量分区守恒与首推分支零缓存从头重算验证 `0` 偏差 | `runs/run_20261120_exact_v1/summary.json`、`runs/run_20261120_exact_v1/cases/*/active_check.json`、`verification_report.json` | 满足 |
| §3.4 必须包含 `>= 6` 类有牙负控（`M2-N1..M2-N6`）且全部非零退出 | 实现 `7` 类共 `10/10` 独立无窗子进程负控（`M2-N1a/b`、`M2-N2`、`M2-N3`、`M2-N4`、`M2-N5`、`M2-N6a/b/c`、`M2-N7`），`10/10` 非零拒收 | `negative_controls.json`、`verify_m2.py` | 满足 |
| §4 四道门禁（含带 `--draft` 的 `r303`）原始退出码全 `0`；空分母三项为 `null`；遵守 `AGENTS.md §11` 全程零弹窗且零修改 M1/历史冻结目录 | `g1`、`g2`（留末步单独错峰执行）、`g3`、`g4`（带 `--draft report.md`）退出码均为 `0`；全部子进程 `creationflags=134217728`；`pycache_audit.ok = true`；`field_error_rate` / `human_time` / `hydraulic` 均为 `null` | `gates/{g1.json,g2.json,g3.json,g4/}`、`result.json`、`run_receipt.json`、`verification_report.json` | 满足 |

## 5. 负控实况

全部 `10` 项负控均通过无窗子进程（`creationflags = 134217728`）在隔离目录 `_nc_scratch/` 内实跑（完整命令与 JSON 输出见 `results/midterm_sprint_20261120/M2_active_identifiability/negative_controls.json`）：

| 编号 | 本应失败的情形 | 实际退出码 | 关键拒收机制与实测反例 | 是否真的失败 |
|---|---|---:|---|---|
| `M2-N1a` | 真实坐标域异时（cross-`t`）并集假通路（单看并集连通，但同一 `(m,t)` 均断）伪报 `CERTIFIED_CONNECTED` | `2` | 求解器正确判 `CERTIFIED_DISCONNECTED` 且标 `phantom_union_path=true`；篡改证书经 `--verify-cert` 报 `CERTIFICATE_INVALID`（与精确细胞矛盾） | 是 |
| `M2-N1b` | 真实坐标域异模式（cross-mode）并集假通路伪报 `CERTIFIED_CONNECTED` | `2` | 求解器正确判 `CERTIFIED_DISCONNECTED` 且标 `phantom_union_path=true`；篡改证书经 `--verify-cert` 报 `CERTIFICATE_INVALID` | 是 |
| `M2-N2` | 将真实 `AMBIGUOUS` 簇 `jardim_c02` 的 `connected_witness` 篡改为非法排除带 `t = -1/2` | `2` | `CERTIFICATE_INVALID`: `connected_witness t=-1/2 is NOT in feasible segments of mode m0` | 是 |
| `M2-N3` | 对真实 `CERTIFIED_CONNECTED` 簇 `perozzo_c04` 伪造主动复核推荐列表 | `2` | 自然产出严格为 `ALREADY_CERTIFIED` 空列表；伪造推荐经 `--verify-active` 报 `ACTIVE_CHECK_INVALID` | 是 |
| `M2-N4` | 在真实 `AMBIGUOUS` 簇 `perozzo_c02` 注入完全孤立无关裂隙 `perozzo_isolated_irrelevant_dummy` 并篡改提升至第 1 名 | `2` | 自然收益严格为 `0/1`（排名 `8/8` 末位）；强行篡改提升后经 `--verify-active` 报 `ACTIVE_CHECK_INVALID` | 是 |
| `M2-N5` | 篡改 `jardim_c02` `collapse_tree` 首推裂隙分支 `mode_TILT_0DEG` 的 `post_status` 伪报坍缩 | `2` | `ACTIVE_CHECK_INVALID`: `claimed=CERTIFIED_DISCONNECTED actual=AMBIGUOUS_WITH_VERIFIED_WITNESSES` | 是 |
| `M2-N6a` | 输入 `scope` 冒充合成域（`SYNTHETIC_DECLARED_TRIANGLE_UNION`）的非法模型 | `2` | `REFUSED` 拒收信封（`search_complete=false`，`REAL_SCOPE_VIOLATION`） | 是 |
| `M2-N6b` | 输入跨场地坐标系污染模型（Perozzo 簇挂 Jardim 坐标系） | `2` | `REFUSED` 拒收信封（`search_complete=false`，`CROSS_SITE_FRAME_MISMATCH`） | 是 |
| `M2-N6c` | 输入全模式空合法段（空分母）模型 | `2` | `REFUSED` 拒收信封（`search_complete=false`），绝不产伪 PASS | 是 |
| `M2-N7` | 真实簇 `jardim_c01` 超出声明 `budget.cells = 4096`（`m4` 触发 `3358 + 835 > 4096`） | `4` | `BUDGET_EXCEEDED` 拒收信封（`search_complete=false, cells=[]`），绝不静默扩预算或截断细胞 | 是 |

## 6. 自有缺陷与未做事项

详表见 `results/midterm_sprint_20261120/M2_active_identifiability/own_defects.json`，核心要点如下：

- **没能验证的项**：
  1. 未执行异模型零共享代码独立盲判（属 M3 职责；本单仅完成本地自检、原始冻结 `TC_network` 抽样比特级等价比对与首推分支零缓存从头重算）；
  2. 未验证 `jardim_c01` 在放宽 `4096` 细胞预算下的整网贯通状态（按宪法与 M2 规格严格输出 `BUDGET_EXCEEDED` 拒收信封并排除出 `21` 簇完整认证分母）；
  3. 未验证地下体内无条件贯通真值、人工复核真实耗时或水力连通（三项空分母指标恒保持 `null`）；
  4. 多步闭环仿真采用 `max_steps = 12` 截断上限且 `BASELINE_RANDOM` 采用 `20` 个固定确定性种子；`collapse_tree` 深度 2 预计算覆盖 Top-3 裂隙。
- **只是没看到问题、并非证明正确的项**：
  1. 精确几何加速与原始冻结 `TC_network/compile_network.py` 的等价比对抽检了 `2` 簇（`perozzo_c05` 与 `perozzo_c02`），未对全部 `22` 簇逐一重跑未加速冻结版；
  2. 共享参数 `t` 模型下允许非主通路接触裂隙（`SHARED_T_CONTACT_PROXY`）作为共享 `t` 的代理观测，若真实岩体各裂隙位移独立则该代理约束将弱于模型假设；
  3. 实测 `14` 例真实 `CERTIFIED_DISCONNECTED` 簇上朴素并集假通路为 `0/14`（跨状态并集幻觉通路拦截牙口由 `M2-N1a/b` 负控实证，但不可反推所有露头均无并集假通路）；
  4. `perozzo_c02` 继承自 M1 的 `quantization_sensitive = true` 边界量化敏感标记；`OURS_MEASURE_SPLIT` 隐含声明可行域上的均匀勒贝格测度加权口径。
- **明确没做的事项**：未修改 M1 或任何历史冻结目录字节；未在跨目录导入中写入 `__pycache__`；未静默调高 `jardim_c01` 细胞预算；未引入泄漏标签的作弊信号或为赢基线调参；未跑全量 `pytest`；未 `git commit` / `git push`。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| 已在 M1 冻结双场地 `22` 个真实露头条件有限多裂隙簇上完成 `100%` 精确有理整网认证链路（`21` 簇完成精确三态证书 + `1` 簇 `jardim_c01` 诚实 `BUDGET_EXCEEDED` 拒收，M1 终版 SHA 100% 对齐），在声明条件模型内 `0` 静默误判 | 不可以说这 `22` 簇的几何认证结论等同于地下体内无条件贯通真值或水力连通/渗透率结论 |
| 已实现原创零泄漏精确主动下一步复核排序器（`active_identifiability.py`）并产出 M3 可独立验算、M5 可零卡顿消费的深度 2 `collapse_tree`；在全部 `6` 例真实 `AMBIGUOUS` 簇上，零泄漏精确仿真证明 `OURS_MEASURE_SPLIT`（均值 `2.052` 步）与 `OURS_MINIMAX_EXACT`（均值 `2.296` 步、`5` 步 `100%` 封顶）较 `BASELINE_RANDOM`（`4.651` 步）与几何启发式（`4.918` / `3.994` 步）显著减少复核步数 | 不可以说仿真步数等同于野外地质师真实人工耗时（`human_time = null`），也不可以说单次内部复验已替代 M3 异模型独立盲判 |
| 本单全部 `10/10` 负控真实非零拒收、四道现役门禁退出码均为 `0`、全程无控制台弹窗且历史冻结目录零修改，产物契约可直接交接 M3、M4 与 M5 消费 | 不可以把 `REAL_DERIVED_CONDITIONED_FINITE_UNION` 真实露头条件域与合成域 `SYNTHETIC_DECLARED_TRIANGLE_UNION` 混栏，也不可以把 `jardim_c01` 预算拒收混入贯通分母 |

交接：请架构师验收 M2，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M2_active_identifiability，判级见 result.json#status。
