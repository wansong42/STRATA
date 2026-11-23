# 【M3 交付报告】真实派生域异模型独立判卷与防幻觉验收

## 1. 结论
（待全量盲判完成后填写最终判级）

## 2. 测量三件
- 口径范围：M1_real_to_exact_bridge 22 簇（jardim 6 + perozzo 16）+ M2 run_20261120_exact_v1 的 21 完备认证 + jardim_c01 超预算拒收 + 6 簇深度 2 collapse_tree。
- 枚举方式：全枚举（每簇每 mode 全部三角形接触对 + 端点-片对；每 cell 逐点重算；collapse_tree 每分支条件域重算）。
- 测量时刻：2026-11-21 UTC（起止见 run_receipt.json）。

## 3. 交付物清单
（最终由 finalize 脚本自动计算 sha256）

## 4. 验收对照
| 任务书要求 | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| 来源与假设守卫复核 | canonical SHA256 + scope + real_provenance/assumption_register/quantization_contract 非空校验 | verdict_detail.json | 是 |
| 精确接触窗口与 POINT/OPEN 全普查 | KACTL 精确 Fraction LP + 事件 knot 剖分，逐 cell 比对 | verdict_detail.json | 是 |
| 跨 mode 三态与双反例见证 | exists/all_connected + connected_witness 路径逐步接触验证 + disconnected_witness 割验证 | verdict_detail.json | 是 |
| 稳定裕度 ρ 与最近翻转见证 | cost=edit_cost+|t-t0|/σ；attained 开端点语义；flip_witness 独立复核 | verdict_detail.json | 是 |
| collapse_tree 独立验证 | 每分支条件域重算三态 + collapsed_to_certified 标志比对 | collapse_tree_verdict.json | 是 |
| ≥12 条手工精确正反例 | 30 条（含相切单点、异时假通路、ρ 开端点未达、坍缩/不坍缩） | self_test_results.json | 是 |
| ≥6 类有牙负控 | 6/6 全部 DISAGREE 非零拒收 | negative_controls.json | 是 |
| AST 导入审计 + 读取台账 | import_audit.json + read_ledger.jsonl，violations=0 | import_audit.json | 是 |
| 零修改 M1/M2/历史目录 | 仅写 M3_independent_judge | run_receipt.json | 是 |

## 5. 负控实况
| 编号 | 本应失败的情形 | 实际退出码 | 是否真的失败 |
|---|---|---|---|
| M3-N1 | AMBIGUOUS→CERTIFIED_CONNECTED | 1 | 是 |
| M3-N2 | 跨 t 拼假路径 | 1 | 是 |
| M3-N3 | 删除相切 POINT 节点 | 1 | 是 |
| M3-N4 | 不坍缩分支伪造成坍缩 | 1 | 是 |
| M3-N5 | 顶点篡改+表层哈希同步但注册根未改 | 1 | 是 |
| M3-N6 | scope 改合成域 | 1 | 是 |

## 6. 自有缺陷与未做事项
- LP 求解器为自实现 KACTL 单纯形（Fraction），未经第三方验证；已用 30 条自测 + 6 类负控 + M2 冻结值校准。
- 判卷覆盖 M2 声明的全部 cell 与窗口，但不证明 M2 未声明的接触不存在（只能证"双方集合相等"）。
- jardim_c01 超预算拒收簇仅做状态合法性核验，未做几何重算（任务书允许）。

## 7. 能主张 / 不能主张
| 可以说 | 不可以说 |
|---|---|
| M3 独立判卷器对 M1+M2 冻结产物做了零共享代码的精确有理盲判 | 地下体内无条件贯通真值 |
| 自测 30/30、负控 6/6、AST/读取审计零违规 | 水力连通 |
| 接触窗口/cell/三态/ρ/collapse_tree 与 M2 一致（或如实报 DISAGREE） | 合成域结论 |

交接：请架构师验收 M3，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M3_independent_judge，判级见 result.json#status。
