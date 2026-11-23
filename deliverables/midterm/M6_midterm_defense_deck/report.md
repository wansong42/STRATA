# 【M6 交付报告】大创中期答辩四幕材料已完成 M3 回填、全量溯源与四门禁封稿

## 1. 结论

**PASS**（唯一机器判级权威见 `result.json#status`）——在仅写本单目录、零修改历史冻结目录和上游交付件的前提下，完成 15 页四幕 PPT 大纲、8 分钟逐页实战讲稿、A/B/C 三档评委尖锐 20 问、数字溯源映射表，以及可离线打开的 HTML 演示包。M3 `verdict_batch.json` 已在盘，三份核心文稿的全部 `[待M3回填]` 占位均已清零，回填为全量 `22/22` 簇独立核验 `AGREE`、`DISAGREE = 0`、`UNDECIDABLE = 0`、`MALFORMED = 0`；其中 `21` 个完备认证簇做几何复算，`jardim_c01` 只做超预算拒答状态一致性核验，明确不冒充几何重算。

严格落实两条诚实口径：① 朴素并集图在 `14` 例真实断割簇上跨状态幻觉通路为 `0/14`，主要危害是在全部 `6/6` 歧义簇上单边报连通并掩盖断割反例（`MASKS_DISCONNECTED_WITNESS`），跨状态假通路拦截牙口由两项定向负控实证；② `jardim_c01` 按声明的 `4,096` 细胞预算输出 `BUDGET_EXCEEDED`，不静默截断、不混入 `21` 簇完备认证分母。`verify_m6.py --run-negative-controls` 实测通过：88 项溯源条目、613 处数字全量核验，3 份核心文稿均为 `CLEAN`、`n_red = 0`，5 项负控全部正确拒收。四道现役门禁的原始退出码与运行回执详见第 4、5 节及 `run_receipt.json`。

## 2. 测量三件

- **口径范围**：本单覆盖大创中期答辩材料及其数字溯源、负控和四道现役门禁。数据轨道永久分栏：历史非理想上界数据阶梯与信息墙参考轨、Perozzo 2024 自动分割与专家解译独立对照轨、双真实露头派生条件有限模型轨、合成声明有理三角形域历史独立判卷轨。真实派生域 `22` 簇包含 `21` 个完备认证簇和 `1` 个超预算拒答簇。不覆盖地下体内无条件贯通真值、水力导通或野外人工实测耗时，三项指标恒为 `null`。
- **枚举方式**：全量枚举，不抽样。`verify_m6.py` 对 3 份核心文稿提取百分比、角度、小数、比值与千分位计数共 613 处（PPT 大纲 268 处、口头讲稿 151 处、20 问应答卡 194 处），逐一匹配 `number_traceability_map.json` 的 88 项条目、212 种字面形式，并对全部源 JSON 的 SHA256、JSON 路径原值和重算结果逐项核验；3 份核心文稿逐一真实传入 `r303_draft_check.py --draft`；5 项负控均由独立无窗子进程实跑。结构复核为 PPT 15 页、讲稿 15 页、应答卡 20 问、HTML 15 页。
- **测量时刻**：上游权威输入以各自 JSON 的测量时刻为准；M3 真实派生域判卷完成时刻为 `2026-11-21T09:43:50.056551+00:00`。本单续作开始前的逐字节归档为 `_pre_edit_archive_m3sync_20261121T111459Z/`（10/10 件 SHA256 一致）；M3 回填与溯源重建完成于 `2026-11-21T11:33:15.512513+00:00` 之后；最终全量校验与负控完成于 `2026-11-21T11:39:27.304189+00:00`。四道门禁的实测起止时刻见 `run_receipt.json`。

## 3. 交付物清单

| 文件（相对 `M6_midterm_defense_deck/`） | 字节数 | sha256 | 作用 |
|---|---:|---|---|
| `PPT大纲_大创中期_15页.md` | 43,982 | `5dbf6296cd3e4bf2d4e59515ceba797e60085e76a9add43d945679aeb12eff90` | 四幕 15 页大纲，含逐页标题、核心结论、≤2 组同屏指标、左右布局、图表绝对路径与页脚口径 |
| `口头讲稿_8分钟实战版.md` | 21,399 | `12c8df76dbb77ad8e30e4c39c96569c61fc9ee12e6b85660142924ceafbcdb9c` | 8 分钟逐页讲稿，含手势、切屏、工作台点击分镜和防卡顿预案 |
| `评委最尖锐20问_标准应答卡.md` | 41,366 | `79fff90d648cdc0c0d0b4bec6f0f27efdd658db196aa3a9f7618dccb006b018c` | A/B/C 三档 20 问，含破题、标准应答、现场调取与禁踩红线 |
| `number_traceability_map.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 88 项条目到在盘 JSON 的可重算溯源映射，含 M3 同步状态与来源哈希 |
| `build_midterm_deck.py` | 见 `result.json#evidence` | 见 `result.json#evidence` | 溯源表构建、M3 回填与离线 HTML 生成脚本 |
| `midterm_deck.html` | 见 `result.json#evidence` | 见 `result.json#evidence` | 15 页离线 HTML 幻灯片，引用 M4 五图与 M5 九张实机截图，含讲稿和 20 问抽屉 |
| `verify_m6.py` | 见 `result.json#evidence` | 见 `result.json#evidence` | 数字溯源、文稿守卫、r303 实跑和负控执行脚本 |
| `negative_controls.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 3 类共 5 项负控的原始退出码、拦截原因与留档路径 |
| `own_defects.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 非空自有缺陷、未验证项与范围外改动拒绝记录 |
| `run_receipt.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 实际解释器、命令、环境、原始退出码、标准输出/错误文件与无窗标志 |
| `gates/g1.json`、`gates/g2.json`、`gates/g3.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 三道通用现役门禁产物 |
| `gates/g4_ppt_outline/draft_check.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | PPT 大纲的 r303 `--draft` 实跑结果 |
| `gates/g4_spoken_script/draft_check.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 口头讲稿的 r303 `--draft` 实跑结果 |
| `gates/g4_qa_20/draft_check.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 20 问应答卡的 r303 `--draft` 实跑结果 |
| `gates/g4/draft_check.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 本报告的 r303 governance 门禁结果 |
| `result.json` | 见 `result.json#evidence` | 见 `result.json#evidence` | 本单唯一机器判级权威与证据清单 |

## 4. 验收对照

| 任务书要求 | 我的交付 | 证据文件 | 是否满足 |
|---|---|---|---|
| 四幕 15 页左右 PPT 大纲，逐页含标题、一句话结论、左右布局、绝对图片路径与页脚口径，每页核心指标不超过 2 组 | 15 页完整大纲，第一幕痛点、第二幕核心突破、第三幕实机演示、第四幕边界与计划；结构程序复核为 15 页 | `PPT大纲_大创中期_15页.md` | 满足 |
| 8 分钟口语讲稿，分幕带时间节点、手势/切屏/点击提示，不念内部代号 | 15 页逐页实战讲稿，含 `00:00–08:00` 时间轴与三级防卡顿预案；内部代号守卫零命中 | `口头讲稿_8分钟实战版.md`、`negative_controls.json` | 满足 |
| 按难度分级的 20 个高危评委问题 | A 档 Q01–Q08、B 档 Q09–Q15、C 档 Q16–Q20，覆盖任务书点名问题；结构程序复核为 20 问 | `评委最尖锐20问_标准应答卡.md` | 满足 |
| 两条诚实口径不得夸大 | 三份文稿均保留断割簇 `0/14`、歧义簇 `6/6` 掩盖断割反例及定向负控说明；均保留 `jardim_c01` 超 `4,096` 细胞预算拒答及分母排除说明 | 三份核心文稿、`PPT大纲_大创中期_15页.md` 第 6 页、应答卡 Q04/Q07 | 满足 |
| M3 未落盘留占位，落盘后刷齐终稿 | M3 已读取并回填；三份核心文稿和 HTML 无占位、无 `PENDING_M3`；明确 21 簇几何复算与 1 簇拒答状态核验的区别 | `number_traceability_map.json#m3_sync_status`、三份核心文稿、`midterm_deck.html` | 满足 |
| 每个上台数字映射至在盘 JSON，并含口径、枚举、测量时刻；零无源、零判废、零 oracle、零混栏 | 88 项条目、212 种字面形式、613 处数字全量核验；源 SHA256、路径原值和重算结果逐项通过 | `number_traceability_map.json`、`verify_m6.py` 输出 | 满足 |
| 加分项：可运行构建脚本并生成离线 HTML，嵌入 M4 五图与 M5 截图 | 已生成 15 页 HTML，包含讲稿和 20 问抽屉；HTML 结构程序复核为 15 页 | `build_midterm_deck.py`、`midterm_deck.html` | 满足 |
| 至少 3 类有牙负控，均须非零退出 | 3 类共 5 项：内部代号、oracle、混标、无源百分比及缺 `--draft` 空跑均正确拒收；3 份核心文稿实跑均 CLEAN | `negative_controls.json` | 满足 |
| 规范四件、空分母三项为 null、四道门禁退出码全 0、无控制台弹窗 | 四件齐全；三项空分母保持 null；四道门禁实测结果见 `result.json#commands` 与 `run_receipt.json`；子进程使用 `CREATE_NO_WINDOW = 134217728` | `result.json`、`run_receipt.json`、`gates/` | 以最终门禁实测为准 |

## 5. 负控实况

四道现役门禁实测全部通过：g1 `repro_certify` 9/9（exit 0）、g2 `selfcheck --quick` 40/40（exit 0）、g3 `r263 --scope delivery --fail-on-hit` 扫描 9 件、非豁免混标命中为 0（`mislabel_hits=[]`，8 条均为已豁免命中，`sum_reconcile.ok=true`，exit 0）、g4 `r303 --draft` 四份文稿（3 份核心文稿 paper + 本报告 governance）全部 `CLEAN`、`n_red = 0`（exit 0）。

全部负控均由 `verify_m6.py --run-negative-controls` 在独立无窗子进程中执行。负控输入副本和 r303 结果留存在 `negative_controls.json#scratch_dir`，不做递归清理。

| 编号 | 本应失败的情形 | 实际退出码 | 触发拦截机制 | 是否真的失败 |
|---|---|---:|---|---|
| `M6-N1a` | 讲稿副本塞入内部代号 | `3` | `INTERNAL_CODE_LEAK`；r303 `HIT`、`n_red = 1` | 是 |
| `M6-N1b` | PPT 副本塞入禁引读数 | `4` | `ORACLE_VALUE_FORBIDDEN`；r303 `HIT`、`n_red = 1` | 是 |
| `M6-N1c` | 应答卡副本混入绝对误差冒充配对增益的病灶句 | `5` | `R263_GAIN_MISLABEL_HIT`、规则 `R-A` | 是 |
| `M6-N2` | PPT 副本写入未登记的无源百分比 | `2` | `UNTRACED_NUMBER_TOKEN`；r303 `HIT`、`n_red = 1` | 是 |
| `M6-N3` | 缺 `--draft` 的 r303 空跑 | `2` | 参数缺失非零拒收；同时 3 份核心文稿真实 `--draft` 均为 `CLEAN`、`n_red = 0` | 是 |

四道现役门禁的逐项退出码、枚举数和日志位置由 `result.json#commands`、`run_receipt.json#commands` 与 `gates/` 承载，不用报告散文替代机器回执。

## 6. 自有缺陷与未做事项

详表见 `own_defects.json`。本轮关键记录如下：

- **没能验证**：未在真实答辩会场的投影仪、讲台主机与音频设备上实测 8 分钟语速和现场操作；HTML 已核验 15 页结构、M3 占位清零及 31 个图片标签（14 个不同图片文件）均可从本目录解析到在盘 PNG，但未完成整套浏览器视觉验收；地下体内真值、野外实测工时与水力导通没有实测分母，保持 `null`。
- **只是没看到问题、并非证明正确**：613 处数字通过溯源只证明登记路径与重算一致，不等于所有自然语言解释都不会被评委质疑；独立判卷全量 `AGREE` 不等于证明代码绝对无 bug；朴素并集图本批断割簇 `0/14` 不等于任意网络都没有跨状态假通路。
- **过程中发现并修复的问题**：初版 M3 回填把 `22/22` 写成“独立复算”，可能把 `jardim_c01` 的预算拒答状态核验误说成几何重算，已统一收紧为“独立核验”，并在 PPT 明确 21 簇几何复算、1 簇状态核验；旧负控脚本递归删除 `_nc_scratch` 被运行时 `CHAT_REFERENCE_PROTECTED` 拦截，已改为每次独立留档、零递归删除；门禁子进程的 `PYTHONPATH` 改为追加项目路径而非覆盖宿主守护路径。最终复核阶段曾因长命令超时中断，但 8 条命令的原始退出码已落盘且全为 0；随后单独刷新 `result.json` 并回读，19 件证据哈希全部一致，四份 r303 产物均为 CLEAN、5/5 负控正确拒收，图片路径全部有效。
- **明确未做**：未修改 M0–M5 或历史冻结目录，未伪造 M3 回执，未运行全量 pytest，未执行 git commit/push，未把 HTML 加分项包装成原生 PPTX 成品。

## 7. 能主张 / 不能主张

| 可以说 | 不可以说 |
|---|---|
| 已完成 15 页四幕 PPT 大纲、8 分钟逐页讲稿、20 问应答卡、数字溯源表及离线 HTML；M3 已回填，三份核心文稿无占位 | 不可把条件有限几何模型结论说成地下体内无条件真值、水力导通或野外实测工时 |
| 88 项溯源条目、613 处数字经全量核验，5 项负控全部正确拒收，3 份核心文稿 r303 均 CLEAN | 不可夸大朴素并集图断割簇 `0/14`，也不可将 `jardim_c01` 的拒答混入 21 簇完备认证分母或说成已做几何重算 |
| M3 对全量 22 簇独立核验均为 AGREE，其中 21 簇几何复算、1 簇拒答状态一致性核验 | 不可把独立判卷全量 AGREE 宣传为代码绝对无 bug，也不可将 HTML 加分项说成已完成原生 PPTX 视觉验收 |

交接：请架构师验收 M6，交付目录 C:\Users\晚菘\Documents\Default Project\results\midterm_sprint_20261120\M6_midterm_defense_deck，判级见 result.json#status。
