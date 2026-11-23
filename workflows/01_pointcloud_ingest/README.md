# Stage 01 · 点云切窗与有限面建簇（Pointcloud Ingest）

- **唯一输入**：通过根级 `data_registry.json` 只读引用原始点云（`jardim_garcia_pointcloud` / `perozzo_2024_finale_ligure` / 未来 `phase2_micro_ct`），严禁复制 >50 MB 原始点云。
- **中期参考实现**：`results/midterm_sprint_20261120/M1_real_to_exact_bridge/build_real_clusters.py`
- **三条守恒原则**：真实孔洞不填、自动碎片不直接吃、非正交裂隙不压成正交矩形。
- **新运行落盘**：`results/active_runs/<YYYYMMDD>_ingest/`
