# NAVIGATION · 现役总地图（2026-11-23 瘦身重构后唯一入口）

> 本仓已于 2026-11-23 经甲方授权完成全仓激进瘦身与工作流重构：根目录由 228 项压缩至 ~31 项，释放 C 盘 33.9+ GB 冗余副本与历史垃圾，四道门禁与中期成果 100% 绿灯留存。

## 1. 一分钟找东西（现役唯一权威）

| 你要找什么 | 直接去这里 |
|---|---|
| **中期最终成果（27 页 PPT + 3 份 Word + M0–M7）** | `results/midterm_sprint_20261120/` |
| **中期收官与下半程接手交接文档** | `results/midterm_sprint_20261120/架构师交接文档_20261122_中期收官与下半程接手.md` |
| **公网三维展示台（在线访问 / 源码）** | `https://49.233.186.35:29446/strata/` · `deploy/strata_showcase/` |
| **本机工作台（受保护端口 8810 / 离线演示 8855）** | `http://127.0.0.1:8810/` · `results/midterm_sprint_20261120/M5_strata_workbench_demo/launch_strata_demo.pyw` |
| **核心算法库** | `src/fractureflow/`（精确三角求交内核：`results/w180_technical_route_20261117/execution_next/triangle_contact/`） |
| **下半程标准工作流与一键导出交付包** | `workflows/README.md` · `workflows/run_gates.py` · `workflows/05_export_release/export_clean_release.py` |
| **在飞任务单 vs 历史任务单** | 在飞：`tasks/active/` · 已闭环历史单：`tasks/archive/` |
| **数据集唯一真源清单（禁复制大点云）** | `data_registry.json`（原始大文件池在 `E:\fracture_data\`） |
| **门禁与资产保护白名单（清理脚本必绕开）** | `contracts/protected_paths.json` |
| **开发日志（append-only，负面清单权威）** | `results/devlog/开发日志.md` |
| **项目红线与卫生条款** | `AGENTS.md`（§3 数字纪律 · §11 禁弹控制台 · §14 仓库卫生条款） |

## 2. 两条日常一键命令（全无窗）

```powershell
# 1) 一键跑全套四道门禁 + 中期 48 项交付验收
E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/run_gates.py

# 2) 预览 / 导出一份 <1 GB、零历史垃圾的干净结题交付包
E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/05_export_release/export_clean_release.py --dry-run
E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/05_export_release/export_clean_release.py --out E:/fracture_release_v1
```

## 3. 目录分层一览

- `src/fractureflow/`：唯一核心算法实现。
- `workflows/`：下半程五阶段流水线（`01_pointcloud_ingest` → `02_exact_certify` → `03_active_check` → `04_phase2_ct_hydraulic` → `05_export_release`）。
- `scripts/`：根目录仅保留 ~20 个门禁与核心工具脚本；5,279 个历史一次性脚本已收纳至 `scripts/_archive_legacy/`。
- `tasks/`：`active/` 放当前在飞任务单，`archive/` 收纳 403 张历史闭环单。
- `results/`：仅留中期冻结件 `midterm_sprint_20261120/`、门禁白名单、8810 运行实例、`devlog/` 与下半程新产物目录 `active_runs/`。
- `data/` & `E:\fracture_data\`：所有数据集按 `data_registry.json` 单源只读引用。
- `archive/`：历史 `AGENTS.md` 字节镜像与冷留档。
- `docs/`：历史文档（按 R322 口径禁新建 `docs/*.md`）。

## 4. 防再变胖铁律

1. 严禁往项目根目录或 `results/` 根目录直接写入临时文件、截图或日志；
2. 严禁把 >50 MB 的原始点云/网格复制到 `results/` 或运行时打包目录，一律只读引用 `data_registry.json`；
3. 新任务单只进 `tasks/active/`，新实验产物只进 `results/active_runs/<YYYYMMDD>_<name>/`；
4. 任何脚本/子进程严禁弹出 `cmd.exe` 或控制台窗口（`AGENTS.md` §11）。
