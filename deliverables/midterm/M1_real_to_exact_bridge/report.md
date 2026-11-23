# 【M1 交付报告】打通真实露头到精确有理整网引擎的第一公里（双场地 22 个真实三维多裂隙簇）

## 1. 结论

**PASS**（判级权威见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/result.json`）——依据架构师正式签署的 B1 / B2 修正案（`results/midterm_sprint_20261120/00_总控单与架构师裁定_20261120.md`），在零修改历史冻结目录（`TC_network/`、`TB_feasible/`、`TE_judge/`、`data/`）的前提下，新建独立真实派生准入校验器 `real_network_format.py`（`schema = "TC-network/real-conditioned/v1"`、`scope = "REAL_DERIVED_CONDITIONED_FINITE_UNION"`），并严格遵循先于任何选簇与连通计算冻结的预注册规则（`prereg.json`），从 **Jardim Garcia**（6 簇，其中 3 簇保拓扑保留真实孔洞）与 **Perozzo 2024**（16 簇，全量几何绑定 DXF 迹线并覆盖四个空间象限）双真实露头场地构建 **22 个真实三维多裂隙相互作用簇**。22/22 簇全部通过独立 CLI 源文件 SHA 校核、全模式精确零距离端点附着检验及内存无损投影至冻结 `TC_network/network_format.py#validate()`；7 类共 12 项有牙负控 100% 非零退出拒收；空分母指标 `field_error_rate`、`human_time`、`hydraulic` 均保持 `null`。

## 2. 测量三件

- **口径范围**：本单覆盖真实露头派生条件有限多裂隙簇的数据准入契约（`TC-network/real-conditioned/v1`）、预注册纯几何选簇构建、保拓扑精确有理三角化、残差驱动连续位移与离散不确定性模式建模，以及 7 类 12 项负控与历史目录零漂移审计。**不覆盖**整网精确连通性求解（属 M2）、异模型独立判卷（属 M3）、地下体内无条件贯通真值或水力连通，且与合成域 `SYNTHETIC_DECLARED_TRIANGLE_UNION` 永久分轨。
- **枚举方式**：全量枚举——Jardim Garcia 全部 `46` 个真实 fragment（主区域面积过滤后）按预注册最远点空间聚类全枚举生成 `6` 簇；Perozzo 2024 全部 `1405` 个专家拟合面与 `1405` 条 DXF 迹线按预注册质量/半径过滤与 `2×2` 空间象限分层全枚举生成 `16` 簇；正例校验覆盖 `22/22` 簇 × `6` 模式（共 `132` 模式），负控覆盖 `7` 类 `12/12` 独立变异子例；历史目录审计逐件核验 `115` 项基线历史文件。权威在盘索引见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/cluster_register.json`、`results/midterm_sprint_20261120/M1_real_to_exact_bridge/verification_summary.json`、`results/midterm_sprint_20261120/M1_real_to_exact_bridge/negative_controls.json` 与 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/frozen_dirs_audit.json`。
- **测量时刻**：
  - 预注册冻结时刻（早于任何建簇代码与选簇）：`2026-11-20T14:52:28.720200+00:00`（见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/prereg_freeze_receipt.json`）
  - 双场地 22 簇构建完成时刻：`2026-11-20T15:34:31.475564+00:00`（见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/cluster_register.json`）
  - 独立复验、负控与历史目录审计完成时刻：`2026-11-20T15:43:16.550678+00:00`（见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/verification_summary.json`）

## 3. 交付物清单

| 文件（相对 `M1_real_to_exact_bridge/`） | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `prereg.json` | 9470 | `f12186af21f896038ad249c54c4a8f1e54b8aac8cda0c6c5aa4b42d52e10593d` | 先于选簇冻结的纯几何预注册规则 |
| `prereg_freeze_receipt.json` | 33311 | `ce6f48cd0690c5c7426154b34b510969386621ff4565d8f7843beb43aa68eb62` | 冻结时刻、双场地源 SHA（含 2.5 GB LAS）与历史冻结目录基线 |
| `real_network_format.py` | 33226 | `c9a8386501873fc552b19707249ce49393733ec50c7269bdd38fdfad89ca47d4` | 独立真实派生准入校验器与内存无损内核适配器（不动 `TC_network/`） |
| `build_real_clusters.py` | 82042 | `4576efc98de4cc59d0adbc4c5f5a7e1382cd7413b69d8da31b67c3962aa18f3c` | 机械执行 `prereg.json` 的双场地 22 簇确定性构建器 |
| `verify_m1.py` | 24794 | `fe3dcc2f15af283a8121637485ceecd93d371a7ac88d6d89ff18426199a31ada` | 全量正例校验、端点附着核验、12 项负控与历史零漂移审计脚本 |
| `cluster_register.json` | 35380 | `2778ba4a096f431235273222bcd5b74b27f9419d33cc34606790bc758bba6bcf` | 22 簇总登记台账（场地分轨、非正交性、孔洞、量化敏感、模型 SHA） |
| `clusters/<cluster_id>/model.json`（22 份） | 4569941（合计） | 逐件见 `cluster_register.json#clusters[].model_file_sha256` | Jardim 6 簇 + Perozzo 16 簇不可变精确有理模型 |
| `verification_summary.json` | 7853 | `ff7a29d6e45a32c659bff0836f77f5c3a6c5ee3ee30a600692047c182dc6d973` | 22/22 正例校验、132 模式端点附着与内存 `TC_network` 投影回执 |
| `negative_controls.json` | 14158 | `32dfb75586be62bb16d23c4dc6ed03539adc7d45da847b6db428a15cf87e7879` | 7 类 12 项有牙负控独立子进程实测记录与首个反例 |
| `frozen_dirs_audit.json` | 1463 | `2ec1a8c7f9c6c098e052689ecae659039eb305d50f040f548eb71acee6a55136` | `TC_network/`、`TB_feasible/`、`TE_judge/`、`data/` 115 项历史文件零修改证明 |
| `own_defects.json` | 6437 | `fb6f52629d0620aa54f307275753c8b301d6014a403bcde475ec0a3c3f5d7eed` | 非空自有缺陷与过程中发现问题的诚实自查台账 |
| `gates/g1.json` | 1904 | `609e9eb6dfbc3459b8cf39e583ece1d7fbda2e647928ffd4efe9a9b550d5fc3a` | 门禁 1（`scripts/repro_certify.py`，`9/9 PASS`，`exit_code=0`） |
| `gates/g2.json` | 10720 | `ba6de785f766b518c85c4ae934dd118203c6e9994bd69ad6052ce227c8d6c931` | 门禁 2（`fractureflow.selfcheck --quick`，`40` 项 `0` 失败，`exit_code=0`） |
| `gates/g3.json` | 20558 | `0c52262e769a3786c9abad03edad4a05d40874f12673928fa60a007b4f5afd5f` | 门禁 3（`scripts/r263_gain_mislabel_scan.py --scope delivery --fail-on-hit`，`scanned=9, red=0`，`exit_code=0`） |
| `gates/g4/` | 见 `result.json#evidence` | 见 `result.json#evidence` | 门禁 4（`scripts/r303_draft_check.py --draft report.md --doc-class governance --json-only`） |
| `report.md` / `result.json` / `run_receipt.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 规范报告、机器判级权威件与运行回执（自引用哈希统一登记于 `result.json#evidence`） |

## 4. 验收对照

| 任务书要求（`M1_真实露头多裂隙簇破壁桥.md`） | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| §3.1 新建 `real_network_format.py`（`TC-network/real-conditioned/v1`，`REAL_DERIVED_CONDITIONED_FINITE_UNION`，含 `real_provenance` / `assumption_register` / `quantization_contract` 及内存无损投影 `to_core_kernel_payload`），零修改 `TC_network/` | 已独立实现标准库精确有理校验器、6 大硬拒守卫与内存投影函数，22/22 簇内存投影通过冻结 `TC_network/network_format.py#validate()`，零写回合成文件 | `real_network_format.py`、`verification_summary.json`、`frozen_dirs_audit.json` | 满足 |
| §3.2 先冻结 `prereg.json` SHA 再筛簇；双场地总数 `>= 15`（Jardim `>= 5`、Perozzo `>= 10`）；每簇 `<= 64` 三角、`<= 16` mode、位长 `<= 8192`；Jardim 保拓扑不凸包填洞；Perozzo 登记 DXF 迹线偏差 | `prereg.json` 冻结于 `14:52:28Z`、建簇于 `15:34:31Z`；共构建 `22` 簇（Jardim `6` 簇含 `3` 簇保洞覆盖全部 `4` 个真实含洞片段、Perozzo `16` 簇覆盖 `4` 个空间象限），三角数 `36–56 <= 64`，模式数 `6 <= 16`，总非正交对 `337/490`（`68.78%`），`perozzo_c02` 如实标为 `quantization_sensitive=true` | `prereg.json`、`prereg_freeze_receipt.json`、`cluster_register.json`、`clusters/*/model.json` | 满足 |
| §3.3 物理驱动非静态不确定性 `t` 与 `modes`（含实测残差法向微平移、边界延展、离散倾角微扰及多模式互斥归属/多段可行域簇） | 每簇锚点面固定（保证全模式端点精确附着），内部裂隙由实测残差 `sigma_i` 驱动法向平移（设计系数 `k=2`），覆盖 `STANDARD_EXT_TILT`（`13` 簇：径向 `1.00x/1.15x` × 精确毕达哥拉斯旋转 `0/±3.95°`）与 `MUTUAL_EXCL_ATTRIB_MULTISEG`（`9` 簇：双闭段 `[-1,-3/5]∪[-2/5,1]` + 互斥归属段 `[-1,-1/10]` vs `[1/10,1]`） | `cluster_register.json`、`clusters/*/model.json`（含 `fracture_catalog` 与 `mode_semantics`） | 满足 |
| §3.4 `>= 6` 类有牙负控（`M1-N1..M1-N6`）100% 非零退出并给首个反例 | 实现 `7` 类共 `12` 项独立子进程负控（`M1-N1a/b`、`M1-N2`、`M1-N3a/b`、`M1-N4a/b`、`M1-N5a/b`、`M1-N6a/b`、`M1-N7`），`12/12` 真实非零退出并记录 JSON 反例 | `negative_controls.json`、`verify_m1.py` | 满足 |
| §4 零修改 `TC_network/`、`TB_feasible/`、`TE_judge/`、`data/` 任何历史字节 | 逐件比对冻结前基线共 `115` 项历史文件，漂移数 `0`（同时透明记录外部并行会话向 `TE_judge/runs/` 的追加记录） | `frozen_dirs_audit.json`、`prereg_freeze_receipt.json` | 满足 |
| §4 四道现役门禁（含带 `--draft` 的 `r303`）原始退出码全 `0`；空分母三项为 `null` | `g1`（`exit 0`, `9` 项）、`g2`（`exit 0`, `40` 项）、`g3`（`exit 0`, `9` 项）、`g4`（带 `--draft` 检验本件终稿，`exit 0`）全部实跑保留原始退出码；`field_error_rate` / `human_time` / `hydraulic` 全部 `null` | `gates/{g1.json,g2.json,g3.json,g4/}`、`result.json`、`run_receipt.json` | 满足 |
| §5 AGENTS.md §11 禁弹控制台窗口纪律 | 全部子进程使用 `creationflags=134217728`（`CREATE_NO_WINDOW`）与 `shell=False`，零 `.bat`/`.cmd`/`cmd.exe` 弹窗 | `verify_m1.py`、`run_receipt.json` | 满足 |

## 5. 负控实况

全部 12 项负控均通过独立无窗子进程调用 `real_network_format.py --validate` 在隔离目录 `_nc_scratch/` 内实跑（完整命令与 JSON 反例见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/negative_controls.json`）：

| 编号 | 本应失败的情形 | 实际退出码 | 触发错误类与关键拒收原因 | 是否真的失败 |
|---|---|---:|---|---|
| `M1-N1a` | 篡改 Perozzo `xlsx` 源 SHA256 的 1 位十六进制字符 | `2` | `Invalid`: `SOURCE_SHA256_MISMATCH` | 是 |
| `M1-N1b` | 将 Jardim `project.json` 指向篡改副本但保留原 SHA 与字节数 | `2` | `Invalid`: `SOURCE_BYTES_MISMATCH` | 是 |
| `M1-N2` | 将真实派生簇 `scope` 偷改为 `SYNTHETIC_DECLARED_TRIANGLE_UNION` | `3` | `Outside`: `REAL_SCOPE_VIOLATION` | 是 |
| `M1-N3a` | 清空 `assumption_register.assumed_edges_tag` 与 `not_claimed` | `2` | `Invalid`: 缺少精确 `ASSUMED_NOT_OBSERVED` 标记 | 是 |
| `M1-N3b` | 保留标签但删去 `not_claimed` 中的地下真值与水力免责声明 | `2` | `Invalid`: 缺少地下无条件真值免责声明 | 是 |
| `M1-N4a` | 将 Jardim 世界坐标注入 Perozzo 簇顶点（跨场地混帧） | `3` | `Outside`: `CROSS_SITE_COORDINATE_OUT_OF_SITE_BBOX` | 是 |
| `M1-N4b` | 将 Jardim `coordinate_frame` 写入 Perozzo 簇元数据 | `3` | `Outside`: `CROSS_SITE_FRAME_MISMATCH` | 是 |
| `M1-N5a` | 用凸包三角化替换 Jardim 含洞片段 `frag_c8f4b1139db379e22a182d023125163a`（填洞） | `2` | `Invalid`: `HOLE_AREA_MISMATCH` | 是 |
| `M1-N5b` | 隐蔽填洞攻击：同步伪造声明面积并只用单个三角形覆盖洞内探测点 | `2` | `Invalid`: `HOLE_FILLED_PROBE_COVERED`（精确有理探测点拦截） | 是 |
| `M1-N6a` | 在 `m0` 注入 `65` 个三角形（超过 `64` 三角预算，禁静默截断） | `4` | `Exhausted`: `triangle budget exceeded in mode m0: 65 > 64` | 是 |
| `M1-N6b` | 注入 `17` 个离散模式（超过 `16` mode 预算，禁静默截断） | `4` | `Exhausted`: `mode budget exceeded: 17 > 16` | 是 |
| `M1-N7` | 注入 `8196` 位有理坐标（字符长 `<= 4096`，直接击穿 `8192` 位长预算） | `4` | `Exhausted`: `rational bit budget exceeded (>8192)` | 是 |

## 6. 自有缺陷与未做事项

详表见 `results/midterm_sprint_20261120/M1_real_to_exact_bridge/own_defects.json`，核心要点如下：

- **没能验证的项**：
  1. 未在 M1 运行全量 22 簇的完整片对接触窗口投影与整网连通认证（属 M2 任务范畴，M1 仅验证了准入、精确端点附着及内存无损投影通过冻结 `TC_network/network_format.py#validate()`）；
  2. 未验证地下体内无条件贯通真值、人工耗时或水力连通（三项空分母指标恒保持 `null`）；
  3. 未从 Perozzo 8,320 万点原始 LAS 点云重新分割拟合平面；默认单簇 CLI 校验对 2.5 GB LAS 仅校验字节数（完整 2.5 GB SHA256 已在预注册冻结时一次性实测核对，并支持 `--hash-large-files` 强检）。
- **只是没看到问题、并非证明正确的项**：
  1. Perozzo 正八边形有限盘模型存在最高约 `7.6%` 的径向几何近似误差且全部外边为 `ASSUMED_NOT_OBSERVED`，准入通过不代表真实裂隙端缘恰好止于该八边形；
  2. Jardim 在 `64` 三角预算下的保拓扑折线简化在极狭长凹片段上引入最高 `1.2487 m` 的二维豪斯多夫边界偏差（`jardim_c03`），多微孔再拟合片段仅保留最大主孔洞；
  3. 连续平移设计系数 `k=2`、交替符号规则与第二分层的双段/互斥区间属于锚定实测拟合残差的条件假设，并非 T-B 多观测交集自动反演结果；
  4. `perozzo_c02` 标称最小非零间距 `0.000433 m` 落入 `2x` Tier-2 量化误差包络，已诚实打标 `quantization_sensitive = true`；
  5. 内存投影通过静态 schema 校验不保证 M2 运行期全量片对投影均不触发 `4096` 细胞 / `65536` 片对算力预算。
- **过程中自查发现并处置的问题**：
  1. 实测 raw `points.ply` 发现上游 `project.json` 中 `frag_ecd9cedbc96720d94aa06f3a592b532f` 的 `surface_geometry.frame_local` 残留拆分前陈旧基向量（若盲用会错位 `2.867 m`），已改按 `support_refit` 规则重建并验证 `46/46` 片段 100% 落位原始点云；
  2. 优化了 329 顶点多边形 VW 简化内层复杂度并补齐了初版漏掉的 Jardim `min_non_orthogonal_pairs >= 2` 过滤（未改动已冻结的 `prereg.json`）；
  3. 识别外部并行会话向 `TE_judge/runs/` 追加自测目录的并发行为，按 `AGENTS.md §4` 改为历史基线文件逐件 SHA 审计并如实登记并发追加记录。
- **明确拒绝做的范围外事项**：未触碰 `TC_network/`、`TB_feasible/`、`TE_judge/`、`data/`、`src/` 任何既有文件字节；未跑全量 `pytest`；未执行 `git commit` / `git push`；未按连通结果挑例；未将合成投影落盘。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| 已在零修改历史冻结目录的前提下建立 `TC-network/real-conditioned/v1` 真实派生准入校验器与内存无损内核适配器，并完成 Jardim Garcia（6 簇）+ Perozzo 2024（16 簇）共 22 个真实三维非正交多裂隙簇的预注册构建与校验 | 不可以说这 22 个真实簇的贯通性结论等同于地下体内无条件贯通真值或水力连通 |
| 22/22 真实簇具备完整源文件 SHA 溯源、精确有理坐标、显式条件假设与量化契约（含 `ASSUMED_NOT_OBSERVED` 与 `quantization_sensitive` 诚实打标）、Jardim 3 簇保洞且不被凸包填洞、132 模式精确零距离端点附着，以及 12/12 有牙负控拦截 | 不可以说真实派生条件模型与合成域 `SYNTHETIC_DECLARED_TRIANGLE_UNION` 同口径，也不可以把条件八边形边界或残差设计系数 `k=2` 混说成直接观测到的野外物理边界 |
| M1 已冻结 `cluster_register.json` 与 `clusters/<cluster_id>/model.json` 共享数据契约，可供 M2 精确整网认证与主动复核排序、M3 异模型独立判卷、M4 三方基线对照及 M5 STRATA 三维工作台直接消费 | 不可以说 M1 已证明这些簇在 M2 精确整网求解中的连通/断割/不可辨识三态比例或主动复核步数优势（均待 M2/M3/M4 实算） |

交接：请架构师验收 M1，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M1_real_to_exact_bridge，判级见 result.json#status。
