# workflows/ · 项目标准工作流与一键交付体系

> 本目录于 2026-11-23 仓库瘦身重构时建立，取代旧版在 `scripts/` 与 `results/` 平铺上千零散文件的做法。
> **任何新研发任务与下半程工作流，一律按本目录 5 个阶段组织，严禁再向根目录或 `results/` 根目录裸写文件。**

## 1. 五阶段流水线结构

| 阶段目录 | 职责 | 输入权威 | 输出落盘位置 |
|---|---|---|---|
| `01_pointcloud_ingest/` | 露头/微 CT 点云切窗、有限面构建、保洞聚簇与预注册校验 | `data_registry.json`（只读引用，禁复制大点云） | `results/active_runs/<YYYYMMDD>_ingest/` |
| `02_exact_certify/` | 有理量化、三维面片精确求交、接触图 BFS 四态认证与独立盲算复核 | 阶段 01 簇包 | `results/active_runs/<YYYYMMDD>_certify/` |
| `03_active_check/` | 零泄漏主动下一步复核排序、反事实信念树仿真与多策略同分母对照 | 阶段 02 认证证书 | `results/active_runs/<YYYYMMDD>_active/` |
| `04_phase2_ct_hydraulic/` | 下半程 4 条攻坚线（区间剪枝攻克 `jardim_c01` / 微 CT 体内真值 / 水力耦合 / 4,320 多机制实验） | 阶段 01–03 + 新增受控数据 | `results/active_runs/<YYYYMMDD>_phase2_<topic>/` |
| `05_export_release/` | **一键导出干净可交付精简包（<1 GB，零历史垃圾，带 SHA256 清单）** | `src/` + `workflows/` + `deploy/` + 冻结交付件 | 指定导出目录（不污染研发主仓） |

## 2. 两条常用一键命令（全无窗，自动配好环境变量）

### ① 一键跑全套四道门禁 + 交付核验
```powershell
E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/run_gates.py
```

### ② 预览 / 导出干净结题交付包（不含任何历史垃圾与 GB 级重复点云）
```powershell
# 先预览将打包哪些文件与预估体积（不写磁盘）
E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/05_export_release/export_clean_release.py --dry-run

# 正式导出干净交付包到指定目录（例如 E:\fracture_release_v1）
E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe workflows/05_export_release/export_clean_release.py --out E:/fracture_release_v1
```

## 3. 协作与防再变胖铁律

1. **任务单统一进 `tasks/active/`**：新单命名 `YYYYMMDD_<模块>_<主题>.md`，闭环登记进 `results/devlog/开发日志.md` 后即移入 `tasks/archive/`。
2. **临时文件严禁进根目录**：一次性探针、调试截图一律走系统临时目录或 `results/active_runs/_scratch/`（完事即删）。
3. **大文件（>50 MB）禁复制**：所有原始点云/网格/体数据只在 `data_registry.json` 登记唯一真源路径，代码只读引用。
4. **任何时候禁弹控制台窗口**：子进程必须带 `creationflags=subprocess.CREATE_NO_WINDOW`，长作业走无窗计划任务或 `pythonw.exe`。
