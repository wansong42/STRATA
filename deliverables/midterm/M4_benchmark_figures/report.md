# 【M4 交付报告】三方基线定量对照与答辩级 5 张核心附图

## 1. 结论

**PASS**（判级权威见 `results/midterm_sprint_20261120/M4_benchmark_figures/result.json`）——在严格保持 **数据阶梯历史参考锚点**（`data_ladder_anchors.json`，不含任何 oracle）、**Perozzo 2024 自动 DSE vs 专家人工解译对照**（`perozzo_dse_vs_manual_summary.json`）与 **双场地 22 簇真实条件有限并集基线基准**（`benchmark_summary.json`，`scope = REAL_DERIVED_CONDITIONED_FINITE_UNION`）三栏永久分轨的前提下，完成基于 `M1` 22 个真实三维多裂隙簇（Jardim Garcia 6 簇、Perozzo 2024 16 簇）与 `M2` 精确有理整网认证产出（`runs/run_20261120_exact_v1`）的三方同台定量对照与 5 张答辩级高清大图（300 DPI PNG + 矢量 SVG）。在已完备精确认证的 21 簇中（另 `jardim_c01` 触发 4096 细胞算力上限进入 `BUDGET_EXCEEDED` 诚实拒答包络），**Baseline 1（名义单态 DFN）发生 6/21（28.6%，含 Jardim 5/5 即 100.0% 与 Perozzo 1/16 即 6.2%，各 3 例假安全连通与 3 例假安全断割）静默翻转**；**Baseline 2（朴素并集图）在已完备认证中幻觉通路率为 0/21（断割条件误报 0/14）**；**本文方法（Ours）在声明条件模型内实现 0/21（0.0%）静默误判**，对 6 簇可辨识歧义（`AMBIGUOUS_WITH_VERIFIED_WITNESSES`）给出有限精确稳定裕度 `rho`（0.087305–1.644842）与主动下一步复核排序——在 6 例歧义簇同分母无泄漏反事实闭环中，`OURS_MINIMAX_EXACT` 均值仅需 `2.29552` 步（于 `k=5` 达 100.0% 测度坍缩），`OURS_MEASURE_SPLIT` 均值仅需 `2.052422` 步（首步坍缩 57.0%），显著优于几何启发式（均值 `3.993562` 与 `4.918113` 步）与 20-seed 随机策略（均值 `4.6505` 步，至第 12 步仍仅 97.2%）。全量重算核验与 4 类 5 项有牙负控 100% 非零拒收，空分母三项（`field_error_rate`、`human_time`、`hydraulic`）及 Jardim 断割空分母率严格保留 `null`。

## 2. 测量三件

- **口径范围**：本单覆盖（A）历史非 oracle 三阶段数据阶梯参考提取（`NNS-001`、`NNS-046`、`NNS-021`，严格锁定 `contains_oracle = false`）；（B）Perozzo 2024 `Automatic-Discontinuities-Geometrical-Info.xlsx`（202,371 面，`Kcolinearity` 含 2 处空缺）与 `Discontinuities-Geometrical-Info.xlsx`（1,405 面）全量对比；（C）同源 M1 `22` 簇 × M2 `run_20261120_exact_v1` 之三方同台定量对照（Baseline 1 名义单态精确有理 BFS、Baseline 2 跨模式/跨 `t` 朴素并集图 BFS、Ours 精确三态认证 + 超预算诚实拒答 + 5 策略主动坍缩曲线，数据权威见 `results/midterm_sprint_20261120/M4_benchmark_figures/benchmark_summary.json`）；（D）5 张答辩图与 4 类 5 项负控及四道门禁。**不覆盖**地下体内无条件贯通真值、现场实测误判率、人工实测耗时及水力渗透连通，**严禁**将 `data_ladder_anchors.json` 或 `perozzo_dse_vs_manual_summary.json` 混入真实 22 簇基准分母。
- **枚举方式**：全量枚举——Perozzo 自动 DSE 表全部 202,371 行与专家表全部 1,405 行通过标准库 `zipfile + ElementTree` 全量解析；22 个真实三维裂隙簇（Jardim `6` 簇 15×6=90 对、Perozzo `16` 簇合计 400 对，共 490 对裂隙法向夹角）与各自 6 个离散模式 × 全部片对区间全量枚举；M2 主动坍缩曲线按同分母分组（全量 `n=6`、Jardim `n=5`、Perozzo `n=1`）全量读取 `summary.json#active_collapse_simulation` 全部 13 个步点（`k=0..12`）；负控覆盖 4 类 5 项独立无窗子进程（`M4-N1a`、`M4-N1b`、`M4-N2`、`M4-N3`、`M4-N4`）。权威索引见 `results/midterm_sprint_20261120/M4_benchmark_figures/benchmark_summary.json`、`per_cluster_table.csv`、`negative_controls.json`。
- **测量时刻**：
  - Phase-1 脚手架（Perozzo 全量解析 + 数据阶梯提取 + M1-only 基线）完成时刻：`2026-11-20` 晚间轮（见 `results/midterm_sprint_20261120/M4_benchmark_figures/perozzo_dse_vs_manual_summary.json`）
  - 接通 M2 `run_20261120_exact_v1` 全量 22 簇三方基准重算时刻：`2026-11-21T04:25:15+00:00`（`run_baselines.py`，历时 `99.45 s`，见 `benchmark_summary.json#at_utc`）
  - 5 张图渲染、全量重算核验与 5/5 负控完成时刻：`2026-11-21T04:32:59+00:00`（见 `negative_controls.json#at_utc`）

## 3. 交付物清单

| 文件（相对 `M4_benchmark_figures/`） | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `run_baselines.py` | 67632 | `72a46d9d928ffba1cf9f7db986fa606097df40fbc70f9dbe5b7dd26501510705` | Perozzo DSE 解析、Nominal DFN / Naive Union 精确计算与 M2 三方汇总脚本 |
| `plot_figures.py` | 50833 | `186dfaec87741d5924dc8c9eff2083adafde13fdb59baec5c918aec0b0a4cc1e` | Agg 后端无弹窗出图脚本（300 DPI PNG + 矢量 SVG） |
| `verify_m4.py` | 20073 | `99c493dde01c46618701dbe178be2e6f011c04bbefdfcfcc2a5ed459278f69fb` | 重算一致性核验与 4 类 5 项有牙负控脚本 |
| `perozzo_dse_vs_manual_summary.json` | 14984 | `eb7884c0afe11637c6abb51c8ddb520d5270453b1609cf4c946938d0fec89942` | 独立分栏：Perozzo 自动 DSE (202,371) vs 专家解译 (1,405) 全量统计 |
| `data_ladder_anchors.json` | 1887 | `2b507fbbfd975c7c8b1e6e98c6df412b6ae7a103de3c3a42e693b22456a3b7a1` | 独立分栏：历史三阶段非 oracle 数据阶梯锚点（含 `contains_oracle=false`） |
| `baseline_scaffold_m1_only.json` | 455825 | `3dd4047223b14c8b3d952e63567b1968fbeaa8479d0cc2ca4d0c36c4045fc2cf` | Phase-1 M1-only 基线预跑产物（保留供溯源） |
| `per_cluster_table_scaffold.csv` | 5950 | `68452bf43688ce78586a00b3a23a0224f52d89a0a92a5659411a91994e9d1072` | Phase-1 22 簇脚手架表 |
| `benchmark_summary.json` | 146596 | `2783744eacf3f43054aa196b88cac0ff9952f7fc66101d2bc494fb35fe90370f` | M4 核心机读三方对比、几何非正交性及五策略主动坍缩汇总权威件 |
| `per_cluster_table.csv` | 7706 | `b076e1ac4d87d0435c43a4466143172a38c3370c23d2a4640d6565ac742cb6c1` | 22 簇逐簇可重算明细 CSV |
| `negative_controls.json` | 4752 | `a5a4019449ea40077f495b35933c11cb6e26bd0f096ccbb13b14e4a25b4ba745` | 4 类 5 项负控实跑原始退出码与反例记录 |
| `figures/fig1_data_ladder_and_architecture.png` | 952399 | `568d6bf43dac98bea50db69afb2bf66135fd9f27f4945e09f236d62907e51f63` | `fig1`（数据阶梯与五重闭环架构图）300 DPI PNG |
| `figures/fig1_data_ladder_and_architecture.svg` | 432322 | `092971a255060aa9a192b5b065c93f1067da32d0a3c97c41a53dfd5a77884da1` | `fig1` 矢量 SVG |
| `figures/fig2_real_outcrop_clusters_overview.png` | 1540153 | `16650e79c9e40fc9755a35d0deaa06ba1d66e8c403414e0ba37c5667c0b62920` | `fig2`（双场地 22 簇总览、非正交性及 DSE 过分割对照）PNG |
| `figures/fig2_real_outcrop_clusters_overview.svg` | 459933 | `026c59914d73326ba2386bd41896164ebc6e7c0cef7c493ae520c395cbad27b0` | `fig2` 矢量 SVG |
| `figures/fig3_baseline_failure_modes.png` | 553285 | `bdc29d1a65db72a0142d399d2c7794a78a3b3bff022d401fe0ad752b82fa694d` | `fig3`（三方基线静默失真对比与四态证书分布）PNG |
| `figures/fig3_baseline_failure_modes.svg` | 354157 | `76e79a709125d3be2c8212a8e7fa11686a39eda94a53718e8cad057662e0390a` | `fig3` 矢量 SVG |
| `figures/fig4_stability_rho_and_witness_anatomy.png` | 1093808 | `ebb79ee1cc993632623b80b88e6f7ebca44f07adb412d37682da70fbc47442f7` | `fig4`（22 簇稳定裕度分布与 `jardim_c04` 双见证 3D 剖面）PNG |
| `figures/fig4_stability_rho_and_witness_anatomy.svg` | 357272 | `0e6f9c2382ea6ca647bf55d32cda5d248e8a15d8eef29eb599601cf1c914df1d` | `fig4` 矢量 SVG |
| `figures/fig5_active_next_check_collapse.png` | 1240136 | `4c36e7c86efd1db354e7985c383a0df9beedbcb0866b969eaff86031832d03be` | `fig5`（主动下一步复核 5 策略同分母坍缩曲线）PNG |
| `figures/fig5_active_next_check_collapse.svg` | 322653 | `22651319677ba8673191d5287163931b10d1229d8a8ca7bd56361d9bd9cd710b` | `fig5` 矢量 SVG |
| `gates/{g1.json,g2.json,g3.json,g4/}` | 见 `result.json#evidence` | 见 `result.json#evidence` | 四道现役门禁落盘回执（G2 最后错峰运行） |
| `report.md` / `result.json` / `own_defects.json` / `run_receipt.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 规范报告、机读权威结果、自有缺陷与运行回执 |

## 4. 验收对照

| 任务书要求（`M4_三方基线定量对照与答辩出图.md`） | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| §3.1 独立解析 Perozzo `Automatic-Discontinuities-Geometrical-Info.xlsx` (202,371 面) vs `Discontinuities-Geometrical-Info.xlsx` (1,405 面)，用标准库 `zipfile + ElementTree` 解析并严格不混为真值 | 已在 `run_baselines.py` 实现共享字符串/流式 XML 解析器；实测 202,371 自动面 vs 1,405 专家面，数量过分割倍比 `144.0363`（半径 p50：`0.106392 m` vs `1.315676 m`，即约 `12.4` 倍差距；R<0.5 m 碎片占比 `91.16%` vs `7.69%`），并诚实登记 `Kcolinearity` 第 109210 与 109821 行缺值；严格设置 `never_used_as_ground_truth = true` | `perozzo_dse_vs_manual_summary.json`、`run_baselines.py` | 满足 |
| §3.2 实现 Baseline 1（名义单态 DFN：`reference.mode` + `reference.t`）与 Baseline 2（朴素并集图：全模式全 `t` 区间存在接触即连边），严格与 M2 `run_20261120_exact_v1` 同源同输入定量对照 | 22/22 簇精确有理几何接触与 BFS 全算；对 21 个已完备簇实现 `nom_cell` 与 `union` 交叉一致性校验（0 偏差），并诚实纳入 `jardim_c01` 为 `BUDGET_EXCEEDED`；产出 `benchmark_summary.json`（Combined `n=22` 完备 `21`：Baseline 1 静默翻转 `6/21 = 28.6%`；Baseline 2 幻觉通路 `0/21`；Ours 模型内静默误判 `0/21`） | `benchmark_summary.json`、`per_cluster_table.csv`、`run_baselines.py` | 满足 |
| §3.3 产出 5 张答辩核心图（300 DPI PNG + SVG），强制 `matplotlib.use("Agg")`、禁止弹窗并目视核查无遮挡/缺字 | 已渲染全部 5 张（PNG+SVG 共 10 文件），`Noto Sans SC` + LaTeX mathtext 零缺失字形警告；对 `fig3`（偏移柱距防数字碰边）、`fig4`（改 `t_minus` 简短小数防长有理串出界 + 抬高图例框）与 `fig5`（白色衬底标注防压线）完成两轮目视打磨 | `plot_figures.py`、`figures/fig1..fig5.{png,svg}` | 满足 |
| §3.4 4 类有牙负控（`M4-N1..M4-N4`）全部真实非零退出并留证据 | 实现并实跑 4 类共 5 项独立无窗子进程负控（`M4-N1a`、`M4-N1b`、`M4-N2`、`M4-N3`、`M4-N4`），5/5 真实非零退出并记录首个反例 | `negative_controls.json`、`verify_m4.py` | 满足 |
| §4 数据纪律与四道门禁（`g1..g4`，G2 留到最后错峰跑）+ 空分母全 `null` | `field_error_rate`/`human_time`/`hydraulic` 及 Jardim `phantom_fp_given_disconnected_rate` 全为 `null`；三轨道严格隔离零 oracle；四门禁全跑留原退出码且无 `| tail` | `gates/{g1.json,g2.json,g3.json,g4/}`、`result.json`、`run_receipt.json` | 满足 |

## 5. 负控实况

全部 4 类共 5 项负控通过 `verify_m4.py` 启动无窗子进程（`creationflags=134217728`，`shell=False`）实跑（详录于 `results/midterm_sprint_20261120/M4_benchmark_figures/negative_controls.json`）：

| 编号 | 本应失败的情形 | 实际退出码 | 触发错误类与关键拒收原因 | 是否真的失败 |
|---|---|---:|---|---|
| `M4-N1a` | 向 `real_outcrop_benchmarks` 注入合成域标号 `SYNTHETIC_DECLARED_TRIANGLE_UNION` | `1` | `CONTAMINATION_REJECTED (M4-N1): SYNTHETIC_DECLARED_TRIANGLE_UNION found in real_outcrop_benchmarks` | 是 |
| `M4-N1b` | 向真实露头 `jardim_garcia_uh` 统计块注入 oracle 禁值 | `1` | `CONTAMINATION_REJECTED (M4-N1): forbidden oracle numeric value ...` | 是 |
| `M4-N2` | 将 `combined_real_conditioned.nominal_dfn.silent_flip_rate` 篡改超过 1% 后跑 `--recompute-check` | `1` | `RECOMPUTE_MISMATCH (M4-N2): expected=0.285714 got=0.298571` | 是 |
| `M4-N3` | 在 `jardim_garcia_uh.disconnected_count=0` 下强填 `0.0` 替代 `null` | `1` | `EMPTY_DENOMINATOR_VIOLATION (M4-N3): must be null, never 0.0` | 是 |
| `M4-N4` | 复制 M2 目录并篡改 `jardim_c01/certification.json` 的 `input_sha256` 首字符后重跑 `run_baselines.py` | `1` | `M2_INPUT_SHA256_MISMATCH (M4-N4)` | 是 |

## 6. 自有缺陷与未做事项

详表见 `results/midterm_sprint_20261120/M4_benchmark_figures/own_defects.json`：

- **没能验证的项**：
  1. M4 本身未从头重演多步主动决策树（第 5 张图及主动曲线指标直接消费并交叉校验 M2 `summary.json#active_collapse_simulation` 落盘结果），故若要审计多步决策树内部构造需联审 M2 源码与 `active_check.json`；
  2. `jardim_c01` 在 M2 超出 4096 细胞算力上限进入 `BUDGET_EXCEEDED`（未产出精确三态认证与 `cells`），因此全量 M1 的 22 簇中仅 21 簇参与静默翻转与幻觉通路率分母；在 M4 不代表对该单簇给出贯通或断割结论；
  3. 现场实测误判率、人工真实耗时与水力渗透连通均无实测样本，恒保持 `null`。
- **只是没看到问题、并非证明正确的项**：
  1. Baseline 2（朴素并集图）在 `14` 例已完备可证断割簇上幻觉通路误报率为 `0/14`，仅说明在这 14 个来自 Perozzo 2024 的簇中，跨模式/跨 `t` 接触片段恰好未能拼接出 A-to-B 通路，不证明朴素并集图在一般三维簇上无假阳性幻觉风险；
  2. `fig1` 展示的数据阶梯历史非 oracle 读数（`NNS-001` / `NNS-046` / `NNS-021`）源自历史交付件 `M0` 与既有 JSON，本单未从原始 2026-08 合成扫描脚本全量重跑合成实验；
  3. 本报告中 `fig3`/`fig4`/`fig5` 的视觉零遮挡由静态 PNG 检查与 `plot_figures.py` 脚本重算保障，SVG 在不同浏览器的中文字体回退取决于查看端系统字体库。
- **过程中自查发现并处置的问题**：
  1. 接入 M2 真实产出后第一时间发现 `jardim_c01` 为 `BUDGET_EXCEEDED` 诚实拒答态、原脚手架 `_extract_m2_status` 会直接中止且原 `fig3`/`fig4` 会将其误标为已认证样本，已及时修正为四态完备分母分离呈现并在 `fig4` 明确标 `REF (超限)`；
  2. 目视核查首批渲染 PNG 时发现 `fig3`（A 栏标签压线）、`fig4`（C 栏标题超长精确有理 `t` 串向右溢出画幅 + 图例略擦最高柱）与 `fig5`（局部百分点标签压曲线），全部已修补并复验；
  3. 修复了 `verify_m4.py` 原字符串子串检查误将 `plot_figures.py` 文档注释中的 `plt.show()` 误判为调用的缺陷，改为 AST `ast.Call` 实检；同时发现并修复 `verify_m4.py` 同时传 `--audit-summary` 与 `--run-negative-controls` 时提前返回的问题。
- **明确拒绝做的范围外事项**：未改动 `M0`/`M1`/`M2`、`TC_network/`、`TB_feasible/`、`TE_judge/`、`data/`、`src/` 任何既有文件字节；未运行已退役的全量 `pytest`；未执行 `git commit` 或 `git push`；未混用历史 `NNS`/`DEP` 或 oracle 数字进入真实露头统计口径。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| 已在同一批 22 个真实露头条件有限多裂隙簇（Jardim 6 + Perozzo 16，与 M1/M2 同 SHA256 绑定）上完成 Baseline 1（名义单态 DFN）、Baseline 2（朴素并集图）与 Ours（精确有理三态认证 + 超预算诚实拒答）三方定量对照，并生成 5 张带 300 DPI PNG 与矢量 SVG 的答辩级图表 | 不可以说 22 个真实簇上的三态结果或主动复核曲线等同于地下体内无条件贯通真值、水力连通或人工现场实测耗时 |
| 在 21 个已完备精确认证簇中，名义单态 DFN 发生 6/21（28.6%，其中 Jardim 5/5 全部翻转）静默误判，而本文方法在声明条件模型内静默误判为 0/21（含对算力超限簇 `jardim_c01` 诚实拒答）；在 6 例歧义簇同分母主动复核中，`OURS_MINIMAX_EXACT` 与 `OURS_MEASURE_SPLIT` 平均步数分别为 `2.29552` 与 `2.052422`，显著低于几何启发式与随机基线 | 不可以说“朴素并集图误报率为 0/14”意味着朴素并集图在普遍三维问题中可靠，也不可以把 Jardim Garcia 与 Perozzo 2024 混成单一无分层场地平均数或省略 `BUDGET_EXCEEDED` 分母口径 |
| 已独立完成 Perozzo 自动 DSE（202,371 面）vs 专家人工解译（1,405 面）全量对比与历史非 oracle 数据阶梯对照（严格分栏隔离），并经全量 CSV 重算一致性检验与 4 类 5 项负控实测 5/5 拒绝 | 不可以说自动 DSE 过分割分布或历史合成域读数可以直接充当真实 22 簇贯通基准的真值或外推系数 |

交接：请架构师验收 M4，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M4_benchmark_figures，判级见 result.json#status。
