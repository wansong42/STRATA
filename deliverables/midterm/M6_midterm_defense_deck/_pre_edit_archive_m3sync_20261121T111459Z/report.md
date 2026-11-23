# 【M6 交付报告】大创中期答辩四幕 15 页 PPT 大纲、8 分钟实战讲稿、评委尖锐 20 问与 100% 数字溯源闭环

## 1. 结论

**PASS**（唯一机器判级权威见 `results/midterm_sprint_20261120/M6_midterm_defense_deck/result.json`）——在零修改任何历史冻结目录与上游 M0/M1/M2/M4/M5 交付件的前提下，严格基于已验收冻结的权威输入（`results/midterm_sprint_20261120/M1_real_to_exact_bridge/cluster_register.json`、`results/midterm_sprint_20261120/M2_active_identifiability/runs/run_20261120_exact_v1/summary.json`、`results/midterm_sprint_20261120/M4_benchmark_figures/benchmark_summary.json`、`results/midterm_sprint_20261120/M4_benchmark_figures/data_ladder_anchors.json`、`results/midterm_sprint_20261120/M4_benchmark_figures/perozzo_dse_vs_manual_summary.json`、M4 5 张 300 DPI 高清大图 `fig1..fig5.png` 及 M5 9 张实机截图 `00..07*.png`），完成了大创中期答辩全新四幕硬核方案：`PPT大纲_大创中期_15页.md`（15 页，每页同屏核心指标 ≤ 2 组）、`口头讲稿_8分钟实战版.md`（分幕带分秒时间轴、手势/切屏/点击 STRATA 工作台提示，正文零内部代号）、`评委最尖锐20问_标准应答卡.md`（A/B/C 三档 20 问分级防御）、`number_traceability_map.json`（登记 87 项核心指标共 212 种字面形式，覆盖 3 份核心文稿全部 603 处数字出现且 100.00% 从源 JSON 动态重算零偏差）以及自包含离线 HTML5 演示包 `midterm_deck.html`（由 `build_midterm_deck.py` 一键生成并内置 `--sync-m3` 自动回填机制）。严格落实两条诚实口径：① 朴素并集图（Baseline 2）在 14 例已完备真实断割簇上跨状态幻觉通路为 `0/14`（`0.00%`），其主要危害是在全部 `6/6`（`100.00%`）多解歧义簇上单边盲报连通从而掩盖断割反例（`MASKS_DISCONNECTED_WITNESS`），跨状态假通路拦截牙口由真实坐标域 2 项定向负控实证；② `jardim_c01` 按预注册声明的 `4,096` 细胞算力上限诚实触发 `BUDGET_EXCEEDED` 拒答并单列于 `21` 簇完备贯通分母之外；因 M3 `verdict_batch.json` 当前尚在并行会话运行中，真实派生域独立判卷槽位按指令保留 `[待M3回填]` 占位。3 大类共 5 项独立无窗子进程有牙负控（`M6-N1a/b/c`、`M6-N2`、`M6-N3`）`5/5` 非零退出拦截；全部 3 份核心答辩文稿及本报告实跑带 `--draft` 的 `r303_draft_check.py` 均为 `CLEAN`（`n_red = 0`）；四道现役门禁原始退出码全为 `0`；三项空分母指标 `field_error_rate`、`human_time`、`hydraulic` 严格保持 `null`。

## 2. 测量三件

- **口径范围**：本单覆盖大创中期答辩四幕 15 页 PPT 大纲、8 分钟实战口头讲稿、评委最尖锐 20 问标准应答卡、全片数字溯源映射表（`number_traceability_map.json`）、离线 HTML5 幻灯片（`midterm_deck.html`）及 3 类 5 项有牙负控与四道现役门禁。严格保持四条数据轨道永久分栏隔离：① 历史非理想上界三阶段数据阶梯与信息墙参考轨（`HISTORICAL_DATA_LADDER_NON_ORACLE`：无联结钻孔 `36.69°`、信息墙锚点 `30.50°` 及其成套 95% 区间 `[27.01°, 34.25°]`、井壁成像 `12.87°`、合成壁面稠密点云基准 `0.37°`）；② Perozzo 2024 自动分割（`202,371` 面）与专家人工解译（`1,405` 面，过分割 `144.04` 倍）独立对照轨（`PEROZZO_DSE_VS_MANUAL_SEPARATE_COLUMN`）；③ 双真实露头派生条件有限多裂隙并集模型轨（`REAL_DERIVED_CONDITIONED_FINITE_UNION`：巴西 `6` 簇 + 意大利 `16` 簇 = `22` 簇，完备认证 `21` 簇 + 超预算诚实拒答 `1` 簇）；④ 合成声明有理三角形域历史独立判卷轨（`SYNTHETIC_INDEPENDENT_JUDGE_ARCHIVE`：`784` 条基础接触与证书回环 + `668` 条整网与跨模式断言）。**不覆盖**地下体内无条件贯通真值、水力导通或野外人工实测耗时（三项恒为 `null`）。
- **枚举方式**：全量枚举——通过 `verify_m6.py` 对 3 份核心答辩文稿全量提取全部百分比、角度、小数、分数比值与千分位计数共 `603` 处数字（`PPT大纲_大创中期_15页.md` `265` 处、`口头讲稿_8分钟实战版.md` `148` 处、`评委最尖锐20问_标准应答卡.md` `190` 处），逐一映射至 `number_traceability_map.json` 的 `87` 项在册条目（覆盖 `212` 种字面形式），并打开对应上游 JSON 全量动态重算核验（`87/87` 零偏差）；对 3 份核心文稿与本报告共 `4` 份 Markdown 文件逐一传入 `--draft` 实跑 `r303_draft_check.py`；负控覆盖 3 大类共 `5/5` 项独立无窗子进程实测。
- **测量时刻**：
  - 上游 M1/M2/M4/M5 权威输入冻结时刻：`2026-11-20T14:52:28Z` 至 `2026-11-21T04:32:59Z`
  - 本单 `build_midterm_deck.py` 构建数字溯源表与离线 HTML 演示包时刻：`2026-11-21T06:14:45+00:00`
  - 本单 `verify_m6.py --run-negative-controls` 完成 603 处数字溯源核验、3 份核心文稿 `r303` 实跑与 5/5 负控时刻：`2026-11-21T06:15:10.445357+00:00`
  - 四道现役门禁（G1、G3、G4 与末步错峰 G2）实测时刻见 `run_receipt.json` 与 `result.json#at_utc`。

## 3. 交付物清单

| 文件（相对 `M6_midterm_defense_deck/`） | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `PPT大纲_大创中期_15页.md` | 43440 | `ecbbf6a90221c636123be35b45f86691e6b9ff9751a6e88038781bf514165fa2` | 四幕 15 页答辩 PPT 大纲（含每页标题、一句话结论、≤2 组指标、左右布局、M4/M5 图表绝对路径与页脚口径） |
| `口头讲稿_8分钟实战版.md` | 21115 | `daf64b0b9b649a686a99ddf07ca4587d0490c4db6cadb40faff56378e59296ea` | 8 分钟（00:00–08:00）逐页口头实战讲稿（含手势、切屏、STRATA 工作台点击分镜与防卡顿预案，零内部代号） |
| `评委最尖锐20问_标准应答卡.md` | 41061 | `90639cc981fd10c37ba68c11d3fd4976ef9d8f127953da0ad810a123a2e4005b` | A/B/C 三级难度共 20 个高危评委问题标准应答卡（含一句话破题、30 秒应答、图表调取指令与禁踩红线） |
| `number_traceability_map.json` | 204727 | `e8291596829e0df55617994ec42c2c4dd4b33fa801c5c13b720cd38fdf5a0b64` | 全片 87 项核心数字（212 种字面形式、603 处出现）到上游 `results/*.json#json_path` 的可重算溯源映射表 |
| `build_midterm_deck.py` | 84786 | `d522c8252cc4b27fc47ce292044cac2ebffbba34c3a336af79bcde168eb2da47` | 数字溯源映射表构建器、`--sync-m3` 自动回填器与离线 HTML5 演示包生成脚本 |
| `midterm_deck.html` | 138437 | `481c3e76d9ee54e70dd7c8a5e0298c459515e421a5538be65e5e77ad8badf71c` | 自包含零 CDN 离线 HTML5 15 页交互式幻灯片（内联 M4 5 张大图与 M5 9 张截图、当前页讲稿抽屉与 20 问速查抽屉） |
| `verify_m6.py` | 23895 | `ac3074af7847a15a632960ae51202476eef5bc77c5a9abc5c6026f109a52e058` | 100% 数字溯源与动态重算校验、内部代号/oracle/混栏守卫、3 份核心文稿 `r303` 实跑及 5 项负控执行引擎 |
| `negative_controls.json` | 5893 | `538c341d685aad1a99423d2f5492e8aba40d0af0aa1976cf4b82ffde5c26053e` | 3 大类共 5 项独立无窗子进程有牙负控（`M6-N1a/b/c`、`M6-N2`、`M6-N3`）实测退出码与拦截记录 |
| `own_defects.json` | 3968 | `c1e22ceee982fef11a7dcad1d0ee844c47fae6741083a87ebf48c585f971ae0b` | 非空自有缺陷、过程中发现并处置的撞值/题注问题及拒绝范围外改动台账 |
| `gates/g1.json` | 1904 | `593791153707e1e3f5aeb10d7f5973bcd562e58d559eca6b485a28fadf4b0f73` | 门禁 1（`scripts/repro_certify.py`，`9/9 PASS`，`exit_code=0`） |
| `gates/g3.json` | 20558 | `b22af223687815313285b027fa3933d830f51613bbefaca684b8b72d17e4db82` | 门禁 3（`scripts/r263_gain_mislabel_scan.py --scope delivery --fail-on-hit`，`scanned=9, red=0`，`exit_code=0`） |
| `gates/g4_ppt_outline/draft_check.json` | 2990 | `36649132cd93105dcfb5118ac5ccdf8d6396750abd73e22f563691e83cea05e0` | 核心文稿 1（`PPT大纲_大创中期_15页.md`）带 `--draft` 实跑 `r303`（`doc_class=paper`，`CLEAN, red=0`） |
| `gates/g4_spoken_script/draft_check.json` | 2877 | `c4bc26f6064317fbc2ca5b65598b1c6bb294c223919eab0e0266168ab6bbeeca` | 核心文稿 2（`口头讲稿_8分钟实战版.md`）带 `--draft` 实跑 `r303`（`doc_class=paper`，`CLEAN, red=0`） |
| `gates/g4_qa_20/draft_check.json` | 2884 | `a769bf584f2c95a34426202d557d4f9cd4a478effeb78211e0e21fbb069b39f5` | 核心文稿 3（`评委最尖锐20问_标准应答卡.md`）带 `--draft` 实跑 `r303`（`doc_class=paper`，`CLEAN, red=0`） |
| `gates/g2.json` / `gates/g4/` / `report.md` / `result.json` / `run_receipt.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 门禁 4（校验本报告）、末步错峰门禁 2（`fractureflow.selfcheck --quick`）及规范四件（哈希统一登记于 `result.json#evidence`） |

## 4. 验收对照

| 任务书要求（`M6_大创中期答辩PPT大纲讲稿与评委20问.md`） | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| §一 & §三.1 构建四幕 `PPT大纲_大创中期_15页.md`（控制 14–16 页，每页同屏核心指标 ≤ 2 组，含标题、一句话核心结论、左右版面布局、M4/M5 图表绝对路径、页脚口径说明） | 已产出 15 页（`P01`–`P15`）四幕完整大纲：第一幕（`P01`–`P04`）用三阶段数据阶梯（`36.69°` / 墙锚 `30.50°` `[27.01°, 34.25°]` → `12.87°` → 合成点云基准 `0.37°`）、`68.78%` 非正交率与 DSE `144.04` 倍过分割破局；第二幕（`P05`–`P09`）展示双场地 `22` 簇、三方基线对照、双反例见证与 $\rho$、主动复核坍缩曲线（`2.05` 步 vs `4.65` 步）及独立判卷/37 项负控；第三幕（`P10`–`P12`）给出现场演示分镜与三级防卡顿预案；第四幕（`P13`–`P15`）明列三项 `null` 诚实边界与下半程规划 | `PPT大纲_大创中期_15页.md` | 满足 |
| §三.2 产出 `口头讲稿_8分钟实战版.md`（分幕带时间节点、手势/切屏/点击 STRATA 工作台提示、自然口语化，禁念内部代号） | 已产出 `00:00`–`08:00`（`480` 秒）分秒级实战讲稿，含 `【姿态与手势】`、`【点击工作台与讲词】`、`【切屏 / 防卡顿预案触发点】`，正文零内部代号（经 `verify_m6.py` 与 `r303 --doc-class paper` 双重核验 0 命中） | `口头讲稿_8分钟实战版.md`、`gates/g4_spoken_script/draft_check.json` | 满足 |
| §三.3 产出 `评委最尖锐20问_标准应答卡.md`（按难度分级覆盖专业评委与通识评委必问的 20 个高危问题，含任务书点名的全部 6 大问题） | 已按 A 档（★★★ 极高危 `Q01`–`Q08`）、B 档（★★☆ 高危 `Q09`–`Q15`）、C 档（★☆☆ 常考 `Q16`–`Q20`）构建完整 20 问应答卡，逐题给出杀伤点、一句话破题金句、30 秒带数标准应答、现场调取指令与绝对禁踩红线 | `评委最尖锐20问_标准应答卡.md`、`gates/g4_qa_20/draft_check.json` | 满足 |
| 用户指令特别强调的两条诚实口径与 M3 占位规则：① 朴素并集图在 14 例真实断割簇上幻觉通路为 `0/14`（主要危害在 6 例歧义簇上掩盖断割反例 `MASKS_DISCONNECTED_WITNESS`，跨状态假通路牙口由定向负控实证）；② `jardim_c01` 按声明 `4,096` 细胞预算诚实拒答；③ M3 未落盘前留 `[待M3回填]` 占位 | 在 P06、P13、讲稿第 6 页及应答卡 Q04/Q05/Q07 中均严格按此两条诚实口径表述，绝不夸大；对尚未完结的 M3 保留 `[待M3回填]` 占位并在 `build_midterm_deck.py --sync-m3` 中实现一键自动回填与重验 | `PPT大纲_大创中期_15页.md`、`口头讲稿_8分钟实战版.md`、`评委最尖锐20问_标准应答卡.md`、`build_midterm_deck.py` | 满足 |
| §三.4 产出 `number_traceability_map.json`（把 PPT 与讲稿及 20 问里出现的每一个数字映射到具体 `results/*.json#json_path`、口径范围、测量时刻，0 无源数字、0 判废数字、0 oracle 混入、0 混栏） | 已建立 87 项指标（212 种字面形式）到上游 JSON 的完整映射，经 `verify_m6.py` 对 3 份文稿提取的 603 处数字 100.00% 匹配并从源 JSON 动态重算（0 偏差、0 无源数字、0 oracle、0 判废数字、0 混栏） | `number_traceability_map.json`、`verify_m6.py` | 满足 |
| §三.5（加分项）提供可运行的 `build_midterm_deck.py` 生成自包含离线 HTML 幻灯片 `midterm_deck.html`，直接嵌好 M4 5 张高清图与 M5 实机截图 | 已实现 `build_midterm_deck.py` 并生成 `midterm_deck.html`（138,437 字节，零外部 CDN，支持键盘左右翻页、`S` 键展开当前页讲稿抽屉、`Q` 键展开 20 问应答抽屉） | `build_midterm_deck.py`、`midterm_deck.html` | 满足 |
| §三 必须包含 ≥3 类有牙负控（`M6-N1`、`M6-N2`、`M6-N3`）写入 `negative_controls.json` 且全部非零退出 | 已实现 3 大类共 5 项独立无窗子进程有牙负控（`M6-N1a` 内部代号注入、`M6-N1b` oracle 数值注入、`M6-N1c` R263 混标注入、`M6-N2` 无源百分比 `73.49%` 注入、`M6-N3` 3 份核心文稿 `--draft` 实跑 CLEAN 暨缺 `--draft` 参数 exit 2 拒收），`5/5` 验证通过 | `negative_controls.json`、`verify_m6.py` | 满足 |
| §四 & §五 规范四件齐全、空分母三项为 `null`、四道现役门禁（含对全部成稿实跑 `--draft` 的 r303）退出码全 `0`、全程严禁弹出控制台窗口 | 规范四件齐全；`field_error_rate` / `human_time` / `hydraulic` 全为 `null`；G1、G2、G3、G4（含 3 份核心文稿与本报告共 4 次 `--draft` 实跑）退出码全为 `0`；全部子进程带 `creationflags=134217728` | `result.json`、`run_receipt.json`、`own_defects.json`、`gates/*` | 满足 |

## 5. 负控实况

全部 3 大类共 5 项有牙负控均由 `verify_m6.py --run-negative-controls` 启动独立无窗子进程（`creationflags = 134217728`，`shell = False`）在临时隔离目录 `_nc_scratch/` 内实跑（完整记录见 `results/midterm_sprint_20261120/M6_midterm_defense_deck/negative_controls.json`）：

| 编号 | 本应失败的情形 | 实际退出码 | 触发拦截机制与错误输出摘要 | 是否真的失败 |
|---|---|---:|---|---|
| `M6-N1a` | 在讲稿副本中故意塞入内部代号（任务号与台账编号） | `3` | `verify_m6.py` 报 `INTERNAL_CODE_LEAK`（`exit_code=3`），同时 `r303 --doc-class paper` 判 `HIT`（`R5` `n_red=1`） | 是 |
| `M6-N1b` | 在 PPT 大纲副本中故意塞入 `oracle` 禁引数值 `` `12.37` `` | `4` | `verify_m6.py` 报 `ORACLE_VALUE_FORBIDDEN`（`exit_code=4`），同时 `r303` 判 `HIT`（`R3b` `n_red=1`） | 是 |
| `M6-N1c` | 在 20 问应答卡副本中故意混入 R263 绝对误差冒充配对增益的病灶句 | `5` | `verify_m6.py` 调用 `r263_gain_mislabel_scan` 报 `R263_GAIN_MISLABEL_HIT: rule=R-A`（`exit_code=5`） | 是 |
| `M6-N2` | 在 PPT 大纲副本中故意写一个不在 `number_traceability_map.json` 里的无源百分比 `73.49%` | `2` | `verify_m6.py` 报 `UNTRACED_NUMBER_TOKEN: token='73.49%'`（`exit_code=2`），同时 `r303` 报 `R1a` `n_red=1` | 是 |
| `M6-N3` | 对全部 3 份核心文稿真实传入 `--draft` 跑通 `r303`（`3/3 CLEAN, n_red=0`），并测试不传 `--draft` 的空跑 | `2`（空跑） / `0`（3 份实跑） | 缺 `--draft` 时 `r303` 原始退出码为 `2` 被拒收；传入 `--draft` 时 3 份核心文稿 100.00% `verdict == CLEAN` 且 `n_red == 0` | 是 |

## 6. 自有缺陷与未做事项

详表见 `results/midterm_sprint_20261120/M6_midterm_defense_deck/own_defects.json`，核心要点如下：

- **没能验证的项**：
  1. `M3_independent_judge/verdict_batch.json` 在本单首次交付落盘时刻仍由并行异模型会话计算中，故真实派生域异模型独立判卷结果按指令保留 `[待M3回填]` 占位（已在 `build_midterm_deck.py --sync-m3` 中写好一键自动回填逻辑，M3 落盘后单条无窗命令即可刷齐终稿）；
  2. 未在答辩现场物理投影仪与讲台音频设备上实测 8 分钟脱稿语速，时间轴（`00:00`–`08:00`）按每分钟 220–240 字学术口语语速与 100 秒工作台操作分镜测算；
  3. 地下体内无条件贯通真值（`field_error_rate = null`）、野外地质师人工复核实测耗时（`human_time = null`）与水力导通（`hydraulic = null`）无实测分母，恒保持 `null`。
- **只是没看到问题、并非证明正确的项**：
  1. 全部 3 份核心文稿与本报告以 `0 RED, 0 EXEMPT, 0 NOTE` 通过 `r303_draft_check.py` 与 `verify_m6.py` 100% 数字溯源核验，证明文稿内零无源数字、零内部代号泄漏、零混栏，但不代表现场评委不会从工程造价或区域地质构造演化角度提出 20 问以外的开放性问题；
  2. 朴素并集图在 14 例真实断割簇上幻觉通路为 `0/14`（`0.00%`），已如实陈述并指出其主要危害是在 6 例歧义簇上掩盖断割反例（`MASKS_DISCONNECTED_WITNESS`），不代表朴素并集图在一般三维网络中无跨状态假通路风险；
  3. 离线演示包 `midterm_deck.html` 通过相对路径引用 M4 图表与 M5 截图，拷贝演示时需保持 `results/midterm_sprint_20261120/` 目录相对结构。
- **过程中自己发现并处置的问题**：
  1. 发现 Perozzo 2024 专家与自动分割中位半径比值（`1.315676 / 0.106392 = 12.3663`）若按两位小数舍入，会在字面上与北山 K=12 oracle 禁引值 `` `12.37` `` 发生同值不同物撞值并触发 `R303 R3b` 红线，因此在文稿中直接汇报中位半径原值（`0.11` 米对比 `1.32` 米）与数量过分割倍比（`144.04` 倍），彻底消除了字面撞值；
  2. `verify_m6.py` 初扫时抓出初稿版面说明中的分栏宽度百分比（如 55%、45%）和应答卡中的假设举例百分比，随即清理了全部装饰性/假设性百分比，确保全稿每一个 `%` 均为在册实测比例；
  3. 针对 `r303` 的 `R8` 图表题注双向配对规则，在 `PPT大纲_大创中期_15页.md` 附录补齐了 5 张核心附图的标准题注使 `refs == defs == 5`，并在讲稿与应答卡中采用 `附图①..⑤` 表述，实现了零孤儿引用。
- **明确拒绝做的范围外事项**：未越权写入或伪造 `M3_independent_judge/` 文件；未改动 M0/M1/M2/M4/M5 或任何历史冻结目录字节；未夸大朴素并集图 `0/14` 读数或隐瞒 `jardim_c01` 预算拒答；未跑全量 `pytest`；未执行 `git commit` / `git push`；全程零控制台弹窗。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| 已完成面向大创中期答辩的四幕 15 页 PPT 大纲、8 分钟分秒级实战讲稿、评委最尖锐 20 问分级应答卡与自包含离线 HTML5 演示包（`midterm_deck.html`），并嵌好 M4 5 张高清大图与 M5 9 张实机截图 | 不可以说答辩材料中展示的真实露头条件有限几何认证结果等同于地下岩体深部无条件贯通真值、水力渗流导通或野外地质师实测工时（三项均严格为 `null`） |
| 3 份核心答辩文稿中的全部 603 处数字（87 项指标、212 种字面形式）已 100.00% 映射至在盘 `results/*.json#json_path` 并经 `verify_m6.py` 动态重算零偏差；3 份核心文稿在最严 `--doc-class paper` 下实跑 `r303_draft_check.py --draft` 均达 `verdict == CLEAN` 且 `n_red == 0, n_exempt == 0, n_note == 0` | 不可以夸大朴素并集图在 14 例真实断割簇上的幻觉通路率（必须如实报告为 `0/14`，将主要危害准确定位于 6/6 歧义簇上掩盖断割反例 `MASKS_DISCONNECTED_WITNESS`），也不可以把 `jardim_c01` 超 `4,096` 细胞算力上限拒答混入 `21` 簇完备贯通分母 |
| 3 大类共 5 项独立无窗子进程有牙负控（`M6-N1a/b/c`、`M6-N2`、`M6-N3`）100.00% 非零退出拦截，四道现役门禁原始退出码全为 `0`，且内置了 `build_midterm_deck.py --sync-m3` 可在 M3 落盘后一键刷齐真实派生域独立判卷读数 | 在 `M3_independent_judge/verdict_batch.json` 正式落盘前，不可以用虚构数字替代 `[待M3回填]` 占位 |

交接：请架构师验收 M6，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M6_midterm_defense_deck，判级见 result.json#status。
