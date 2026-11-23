# Stage 03 · 零泄漏主动下一步复核（Active Next-Check）

- **中期参考实现**：`results/midterm_sprint_20261120/M2_active_identifiability/active_identifiability.py`
- **零泄漏铁律**：策略只能读取当前公开合法域与静态几何，严禁读取隐藏真值标签或连通结果；所有策略同分母对比。
- **口径边界**：输出为反事实信念树仿真步数，不直接宣称现场人工工时。
- **新运行落盘**：`results/active_runs/<YYYYMMDD>_active/`
