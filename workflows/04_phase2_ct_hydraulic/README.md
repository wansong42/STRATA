# Stage 04 · 下半程四大攻坚槽位（Phase 2 Roadmap）

新立项任务单统一放 `tasks/active/YYYYMMDD_<topic>.md`，实验产物落 `results/active_runs/<YYYYMMDD>_<topic>/`；验收定型后如要纳入最终结题包，输出副本至 `results/active_runs/release_<topic>/` 即可被 `05_export_release` 自动收录：

1. **P0 · 精确有理分层包围盒与区间剪枝**：将单簇预算由 `4,096` 格扩容至 `65,536` 格，攻克第 22 簇 `jardim_c01`（当前需 `5,024` 格）。
2. **P0 · 微 CT / 透明岩体体内真值基准**：补齐 1–2 组受控三维体内真值，填补当前 `field_error_rate = null` 缺口。
3. **P1 · 开度分布与独立渗流求解器耦合**：打通几何筛查 → 水力连通两级链路，填补 `hydraulic = null` 缺口。
4. **P1 · 4,320 案例-格多机制对照 + 人工计时可用性 + 软著/论文**：补齐申报书滞后项与 `human_time = null` 缺口。
