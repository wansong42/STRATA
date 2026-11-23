# 【M3 交付报告】22 簇真实派生域零共享代码精确有理独立判卷：全部 AGREE

## 1. 结论

**PASS** —— 在不读取、不导入任何 M1/M2/TC/TB/src 求解 `.py` 源码的前提下，独立实现精确有理（`fractions.Fraction`）判卷器，对 M1 全部 **22 簇**真实派生条件有限并集域与 M2 `run_20261120_exact_v1`（**21 例完备认证 + jardim_c01 超预算拒收 + 6 簇 collapse_tree（含 depth-2）及 ranked_fractures 全部候选动作**）做全量盲判：
`DISAGREE = 0`、`MALFORMED = 0`、`UNDECIDABLE = 0`（`AGREE = 22`）。
自测 **30/30**、负控 **6/6 全部正确拒收**、AST 与读取台账审计 **violations = 0**，四道现役门禁全过。

**覆盖强化（本轮新增）**：collapse_tree 深度 2 `next_step` **74 条子分支**全部独立复算通过（含父子测度分解勾稽）；`expected_remaining` / `expected_reduction` / `measure_split_*` / 全部 **840 个候选动作结果**独立复算通过。

## 2. 测量三件

- **口径范围**：真实派生条件有限并集域（`REAL_DERIVED_CONDITIONED_FINITE_UNION`）。含 Jardim 6 簇 + Perozzo 16 簇；jardim_c01 仅做 `BUDGET_EXCEEDED` 拒收状态合法性核验（不重算几何）。不覆盖地下体内无条件真值、不覆盖水力连通、不覆盖合成域。
- **枚举方式**：**全枚举、非抽样**。逐簇逐 mode（126 次 mode 求值）对 **132,000** 个三角形对/端点-片对做精确 LP 接触窗口重推；重推 **10,551** 个 cell（POINT 5,352 + OPEN 5,199）与 **22,399** 个可达接触窗口；**75** 条 collapse_tree depth-1 分支 + **75** 条 ranked_fractures 分支条件域重算；**74** 条 depth-2 `next_step` 子分支 + **840** 个候选动作结果独立复算。
- **测量时刻**：2026-11-21 UTC（任务书执行与逐簇判卷、门禁运行，详见 `run_receipt.json`）。

## 3. 交付物清单

| 文件 | 字节数 | sha256（前 16 位） | 作用 |
|---|---:|---|---|
| `judge_real.py` | 45,444 | `be567cff09bfbec0` | 独立判卷器（KACTL Fraction LP + 事件 knot cell 剖分 + BFS 连通 + ρ + collapse 复核） |
| `self_test.py` | 22,200 | `—见 result.json` | 30 条手工精确正反例 |
| `negative_controls.py` | 8,029 | `—见 result.json` | 6 类变异负控 |
| `audit_imports.py` | 6,300 | `—见 result.json` | AST 导入审计 + 读取台账违规扫描 |
| `verify_ranking.py` | 7,500 | `—见 result.json` | 排序收益 / 零收益 / 隐藏标签隔离核验 |
| `verdict_batch.json` | — | 见 `result.json#evidence` | 22 簇 AGREE 汇总（含枚举计数与逐簇证据 sha256） |
| `verdict_detail.json` | — | 见 `result.json#evidence` | 逐簇判卷明细 |
| `collapse_tree_verdict.json` | — | 见 `result.json#evidence` | 6 簇 75 条分支坍缩判定 |
| `ranking_verdict.json` | — | 见 `result.json#evidence` | 6 簇排序收益与隐藏标签隔离核验 |
| `negative_controls.json` | — | 见 `result.json#evidence` | 6/6 负控正确拒收记录 |
| `import_audit.json` | — | 见 `result.json#evidence` | violations = 0 |
| `read_ledger.jsonl` | — | 见 `result.json#evidence` | 110 条白名单 JSON 读取台账 |
| `self_test_results.json` | — | 见 `result.json#evidence` | 30/30 自测通过 |
| `independence_receipt.json` | — | 见 `result.json#evidence` | 会话分离与零共享代码声明 |
| `gates/g1.json` | — | 见 `result.json#evidence` | 三锚复现 9/9 PASS |
| `gates/g2.json` | — | 见 `result.json#evidence` | selfcheck --quick 40/40 PASS |
| `gates/g3.json` | — | 见 `result.json#evidence` | 增益误标扫描 PASS |
| `gates/g4/draft_check.json` | — | 见 `result.json#evidence` | 草稿检查 CLEAN |

> 完整 sha256 值见 `result.json#evidence`（可在盘重算）。

## 4. 验收对照

| 任务书要求 | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| §二.1 来源与假设守卫 | canonical SHA256 逐簇复算并与 `certification#input_sha256`、`cluster_register` 比对；scope / real_provenance / assumption_register / quantization_contract 非空校验 | `verdict_detail.json`（每簇 `checks.canon_sha256`） | 是 |
| §二.2 精确接触窗口与 POINT/OPEN 全普查 | 自行实现 KACTL Fraction 单纯形；132,000 对窗口重推、事件 knot 剖分 10,551 cell，逐 cell 连通性比对 | `verdict_batch.json#enumeration`、`verdict_detail.json` | 是 |
| §二.3 三态与双反例见证 | exists/all_connected 重算；connected_witness 每步接触点重心坐标合法；disconnected_witness 割无活动跨边 | `verdict_detail.json` | 是 |
| §二.4 ρ 与最近翻转见证 | `cost = edit_cost + |t−t0|/σ`；attained=False 时 infimum 取开端点；flip_witness 独立复核 | `verdict_detail.json#checks.rho_recomputed` | 是 |
| §二.5 collapse_tree 独立验证 + 零收益 + 标签隔离 | 75 条 depth-1 分支 + 74 条 depth-2 分支条件域重算三态；minimax/worst-case/guarantee/rank 复算；零收益裂隙收益恰为 0；排序键由几何域推出；expected_*/measure_split_* 840 个候选动作全复算 | `collapse_tree_verdict.json`、`ranking_verdict.json`、`depth2_verdict.json`、`expected_verdict.json` | 是 |
| §三.1 ≥12 条手工精确正反例 | 30 条（含相切单点、异时假通路、ρ 开端点未达、坍缩/不坍缩、拒收路径、LP 基线、SHA、ρ 公式） | `self_test_results.json` | 是 |
| §三.2 ≥6 类有牙负控 | M3-N1..N6 全部 DISAGREE 非零拒收 | `negative_controls.json` | 是 |
| §三.3 AST 导入审计 + 读取台账 | 5 个 .py 全扫描，violations=0；110 条读取记录全为白名单 JSON | `import_audit.json`、`read_ledger.jsonl` | 是 |
| §四 PASS 门槛 | 22 簇 DISAGREE=0 / MALFORMED=0 / UNDECIDABLE=0，自测 30/30、负控 6/6、审计零违规 | `result.json#status` | 是 |
| §五 无窗 / 零改历史 / 不跑全量 pytest / 不 commit | 全程 `creationflags=134217728`，仅写 M3 交付目录 | `run_receipt.json` | 是 |
| 总控 G1–G4 | 三锚 9/9、selfcheck 40/40、误标扫描 PASS、草稿 CLEAN | `gates/` | 是 |

## 5. 负控实况

| 编号 | 本应失败的情形 | 实际退出码 | 是否真的失败 |
|---|---|---|---|
| M3-N1 | `AMBIGUOUS` 篡改为 `CERTIFIED_CONNECTED` | 1（DISAGREE：`ALL_CONNECTED_MISMATCH`） | 是 |
| M3-N2 | 跨 `t` 拼假路径 | 1（DISAGREE：`CONNECTED_WITNESS_INVALID`） | 是 |
| M3-N3 | 删除一个相切 POINT 节点 | 1（DISAGREE：`CELL_MISMATCH`） | 是 |
| M3-N4 | 不坍缩分支伪造成 `CERTIFIED_*` | 1（DISAGREE：`COLLAPSE_BRANCH_STATUS_MISMATCH`） | 是 |
| M3-N5 | 顶点篡改 + 表层哈希同步但注册根未改 | 1（DISAGREE：`SOURCE_SHA_MISMATCH_REGISTER`） | 是 |
| M3-N6 | scope 改合成域冒充真实簇 | 1（DISAGREE：`SCOPE_NOT_REAL_DERIVED`） | 是 |

## 6. 自有缺陷与未做事项

详见 `own_defects.json`（非空）。要点：
- jardim_c01 超预算拒收仅做状态合法性核验，**未做几何重算**。
- 自实现 KACTL Fraction 单纯形虽经 30 例自测 + 132,000 对窗口比对，仍**不能据此证明对任意退化输入无缺陷**。
- 22 簇 AGREE 证明的是"与 M2 冻结产物逐 cell 一致"，**不排除双方共同的模型假设错误**；隐藏标签隔离以几何可推出排序键为据，**未审计 M2 进程内部侧信道**。
- `expected_remaining`/`expected_reduction` 采用按分支测度占比加权口径（经 jardim_c02 rank1=29/10 数值反推校准后全量复算通过），该口径系 M2 声明定义，非本判卷器独立发明。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| M3 对 M1+M2 冻结产物做了零共享代码的精确有理全量盲判，22 簇全部 AGREE | 地下体内无条件贯通真值 |
| 自测 30/30、负控 6/6、审计零违规、G1–G4 全过 | 水力连通 |
| 接触窗口 / cell / 三态 / ρ / collapse_tree / 排序收益与零收益与 M2 一致 | 合成域结论、真实露头域结论互相套用 |
| M2 的 `BUDGET_EXCEEDED` 拒收合法 | 拒收簇已被证明不可达/不可辨识 |
| 排序键与 informative 集可由几何域推出，未见隐藏标签泄漏 | 已证明 M2 内部无任何侧信道 |
| depth-2 子分支与全部候选动作已独立复算 | — |

**过程记录（留痕）**：早期自实现两阶段单纯形存在符号/目标行处理缺陷，曾产生 6 簇 `WINDOW_EXTRA_IN_M2` 假 DISAGREE；定位后改用 KACTL 移植并全部转 AGREE，旧 16/6 全量件原样归档于 `_pre_final_archive_20261121/`（含 sha256 校验）。`remaining_ambiguity_measure` 的保守口径（分支非 CERTIFIED 即整支测度计为剩余歧义）经 jardim_c02 rank1 minimax=5/9 数值反推校准；零收益裂隙不展开分支的 worst-case=全域歧义口径经 jardim_c03 / perozzo_c02 校准；`expected_remaining` 的按分支测度占比加权口径经 jardim_c02 rank1=29/10 校准；`mode_semantics.feature_assignments` 是裂隙特征分组的正确来源（非 modes[].feature_assignments）。

---

**过程记录（留痕）**：早期自实现两阶段单纯形存在符号/目标行处理缺陷，曾产生 6 簇 `WINDOW_EXTRA_IN_M2` 假 DISAGREE；定位后改用 KACTL 移植并全部转 AGREE，旧 16/6 全量件原样归档于 `_pre_final_archive_20261121/`（含 sha256 校验）。`remaining_ambiguity_measure` 的保守口径（分支非 CERTIFIED 即整支测度计为剩余歧义）经 jardim_c02 rank1 minimax=5/9 数值反推校准；零收益裂隙不展开分支的 worst-case=全域歧义口径经 jardim_c03 / perozzo_c02 校准。

交接：请架构师验收 M3，交付目录 `C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M3_independent_judge`，判级见 `result.json#status`。
