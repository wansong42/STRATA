"""R322 · 预注册（一切结果件之前落盘）。

落盘顺序即 G1 时序判据：本件 mtime 必须早于 results/r322_final_settlement/ 下其他任何文件。
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r322_common as C  # noqa: E402

TB = ("tasks/R322_任务书_★★_盘面终局清算与治理退役执行_红账登记不重锚_负空间守卫_编号终局_v0.md")
PLAN = "docs/技术路线_终局_结题锚定与治理退役_v2.0.md"
HANDOVER_LAST = "docs/架构师交接文档_20260923_终局轮_长线方案改判与三张终局任务书签发_治理退役落成机制_D1至D10逐条答复.md"
KANBAN = "tasks/R7R8_线状态看板.md"
AGENTS = "AGENTS.md"
INTRO = "docs/项目入门_从这里开始.md"
LEDGER = "results/digital_ledger.json"

DISCIPLINES = ("PYTHONDONTWRITEBYTECODE", "禁全 workspace 清缓存", "robocopy",
               "镜像不得排除", "git 禁全量还原")

DELIVERY_FACE = [
    "docs/论文_墙_发现过程技术报告.md",
    "docs/墙论文_体例与禁止项_作者取料件_v1.md",
    "docs/墙论文_证明状态_作者取料件_v1.md",
]

# 本单红面复扫范围（= 终局方案 §6.2 逐条点名族 ∪ 输入件 4 文件）。
# 全量 pytest 被任务书 §0-3 禁 ⇒ 红面全集不可得；本件只主张**下界**。
RESCAN_SCOPE = [
    "tests/test_v_r304_.py", "tests/test_v_r307_.py", "tests/test_v_r263_.py",
    "tests/test_v_r314_.py", "tests/test_v_r308_.py", "tests/test_v_r315_.py",
    "tests/test_v_r218_.py",
]


def _lines(p):
    return C.read_text(p).count("\n") + 1


def main():
    C.assert_env_pinned()
    os.makedirs(C.OUT, exist_ok=True)

    def face(p):
        ap = os.path.join(C.ROOT, p.replace("/", os.sep))
        return {"path": p, "exists": os.path.isfile(ap),
                "bytes": os.path.getsize(ap) if os.path.isfile(ap) else None,
                "lines": _lines(p) if os.path.isfile(ap) else None,
                "sha256": C.sha256_file(ap) if os.path.isfile(ap) else None}

    tkb = glob.glob(os.path.join(C.ROOT, "tasks", "R*.md"))
    hov = glob.glob(os.path.join(C.ROOT, "docs", u"架构师交接文档_2026*.md"))
    # 任务书 T3-4 字面口径 *取料件*.md 与"精确作者取料件口径"两个数都记，理由见 rulings D-2
    rose_loose = sorted(C._rel(p) for p in glob.glob(os.path.join(C.ROOT, "docs", u"*取料件*.md")))
    rose_precise = sorted(x for x in rose_loose if u"交接文档" not in x)

    prereg = {
        "id": "R322-PREREG-V1",
        "unit": "R322",
        "title": "盘面终局清算与治理退役执行",
        "star": "★★☆",
        "batch": "终局批次 3/3",
        "stamped_at": C.now_iso(),
        "epoch": C.time.time(),
        "taskbook": {"path": TB, "sha256": C.sha256_file(os.path.join(C.ROOT, TB.replace("/", os.sep))),
                     "bytes": os.path.getsize(os.path.join(C.ROOT, TB.replace("/", os.sep))),
                     "taskbook_version_read": "v0"},
        "authority": [PLAN, HANDOVER_LAST],
        "environment": {"python": C.PINNED, "env_guard": "scripts/r206_env_guard.py (坑 43)",
                        "env_guard_verdict": "PASS numpy 2.2.5 scipy 1.15.3（本会话 22:5x 实测）",
                        "env_vars": ["PYTHONUTF8=1", "KMP_DUPLICATE_LIB_OK=TRUE",
                                     "PYTHONDONTWRITEBYTECODE=1"]},
        "resources": "CPU only · 零训练 · 零 GPU · 单人单会话",

        # ───────────────────────── 前置实测（禁凭记忆，坑 72）
        "preconditions_measured": {
            "R321_phase1_handback": {
                "required": "results/r321_author_drill/handback.json（任务书 §0-1）",
                "on_disk": os.path.isfile(os.path.join(C.ROOT, "results/r321_author_drill/handback.json")),
                "results_r321_dir_exists": os.path.isdir(os.path.join(C.ROOT, "results/r321_author_drill")),
            },
            "R320_verdict": {
                "on_disk": bool(glob.glob(os.path.join(C.ROOT, "results/r320_*/verdict.json"))),
                "results_r320_dir_exists": os.path.isdir(os.path.join(C.ROOT, "results/r320_claim_hardening"))
                or bool(glob.glob(os.path.join(C.ROOT, "results/r320_*"))),
                "third_extraction_doc_on_disk": os.path.isfile(
                    os.path.join(C.ROOT, "docs/墙主张_范围与机制_作者取料件_v1.md")),
            },
            "R319_state": {
                "verdict_on_disk": os.path.isfile(os.path.join(
                    C.ROOT, "results/r319_travel_exemption_split/verdict.json")),
                "last_write_at_prereg": max(
                    (os.path.getmtime(p) for p in glob.glob(
                        os.path.join(C.ROOT, "results/r319_travel_exemption_split/**"), recursive=True)
                     if os.path.isfile(p)), default=None),
                "writes_AGENTS_or_kanban": False,
                "basis": "其任务书 §3-1 明列禁写面含 看板 / 台账 / AGENTS.md / docs/** ⇒ "
                         "本单 T3-2 看板冻结与 T5 AGENTS.md 追加与其写面无交集（实测依据）",
            },
            "concurrency_cap": "D4 裁定 = 终局期并发顶 1；本单开工时 python 进程实测 0 个",
        },

        # ───────────────────────── 判据（阈值全部预写死，后跑不改）
        "criteria": {
            "J1_红账登记": {
                "must": "每节点 file::test_name 粒度；owner/nature/would_red/evidence(到行) 齐备",
                "nature_enum": ["point_in_time_as_frozen_surface", "self_delegated_invariant",
                                "real_defect", "suspended_account"],
                "disposition_enum": ["PERMANENT_RED_REGISTERED", "REAL_DEFECT_FOR_R265"],
                "core_test": "would_red_if_work_done_right：属主单把工作做对了它还会红吗",
                "mapping": {"true": "PERMANENT_RED_REGISTERED（不修、不删牙、不代改）",
                            "false": "REAL_DEFECT_FOR_R265（须附'为什么做对了不会红'论证，否则归类无效退回永久红）"},
                "owner_correction": "红账误写'归 R281'处一律改判属主 = R265（R281 已于 09-21 收口）",
            },
            "J2_索引": {"lines_hard_max": 60, "every_active_item_must_be_on_disk": True,
                        "closed_five_verbatim": ["专利", "署名", "单位", "大创号", "空私有库"],
                        "suspended_four_verbatim": ["英文稿", "投稿包", "复现包", "导师包"]},
            "J3_守卫有牙": {"guards": 6, "cases_min": 12, "poison_noop_max": 0},
            "J4_磁盘": {"action_none_threshold_GiB": 15.0, "stop_and_report_below_GiB": 10.0,
                        "migrate_sha_mismatch_max": 0},
            "体积上限_MB": 5.0,
            "AGENTS_UFFFD_上限": 2094,
            "索引落点判据_任务书T2步骤2条件": "三条守卫须同时满足：不断言字节 sha ∧ 不断言行数常量 ∧ 不断言'某串只出现一次'",
        },

        # ───────────────────────── 六条负空间守卫规格
        "guard_spec": {
            "file": "tests/test_governance_retired.py",
            "root_override": "环境变量 R322_GUARD_ROOT 可把守卫指向临时底本（毒丸双测用）；缺省 = 真仓",
            "guards": [
                {"id": "G-1", "name": "编号终局", "assert": "tasks/R32[3-9]_ / R3[3-9][0-9]_ / R[4-9][0-9][0-9]_ 三类任务书零存在"},
                {"id": "G-2", "name": "看板封存", "assert": "看板 sha256 == results/r322_final_settlement/kanban_retired.sha256",
                 "not_r257_pathology": "退役后看板本就不该再长；'不得变化'是它应有的判据（须在注释里写明与 R257 病根的区分）"},
                {"id": "G-3", "name": "AGENTS.md 退役条款在场 + U+FFFD ≤ 2094"},
                {"id": "G-4", "name": "交付面未被新增文档稀释",
                 "assert": "作者取料件集合 == 退役冻结集合（新增即红）；技术路线_终局_*.md 恰 1 件"},
                {"id": "G-5", "name": "禁新建解释文档",
                 "assert": "docs/ 下相对 T2 基线清单的新增 md ⊆ 白名单 {治理退役声明.md, 终局轮交接文档}；"
                           "且任何新增 docs/架构师交接文档_2026* 一律 RED（交接家族冻结）"},
                {"id": "G-6", "name": "量纲字面量闸（D10 并入既有族）",
                 "assert": "scripts/r3[12]*_*.py 的渲染函数内 st_size / getsize / len(splitlines()) 结果禁进 f-string/format 正文"},
            ],
            "teeth": "每条 1 枚负例（构造违例必 RED）+ 1 枚正例（构造合规必 GREEN）；"
                     "注入后与被检对象字节相同 ⇒ 该毒丸无效并判 RED（禁记 no-op）",
        },

        # ───────────────────────── 落位前已实测的口径冲突（先注册后动手）
        "rulings_and_conflicts": [
            {"id": "D-1", "severity": "阻断级（前置）",
             "finding": "任务书 §0-1 前置「R321 Phase 1 handback.json 在盘」实测不成立：results/r321_author_drill/ "
                        "与 results/r320_* 两目录均不存在 ⇒ 终局批次 1/3、2/3 从未派出，"
                        "R319 亦无 verdict（末写 %s）" % C.time.strftime("%H:%M", C.time.localtime(
                            max((os.path.getmtime(p) for p in glob.glob(
                                os.path.join(C.ROOT, "results/r319_travel_exemption_split/**"),
                                recursive=True) if os.path.isfile(p)), default=0))),
             "ruling": "不改判据、不伪造前置：G1 如实记 FAIL(前置未满足)，判级按 §2 走 PARTIAL "
                       "（'有门未过但已如实登记未放宽'）。R322 的可交付部分（T1/T3/T4/T5/T6 与索引）"
                       "全部按现盘实测基线执行；索引的'现役'清单只列在盘件，缺件（第三张取料件、作者自述）"
                       "标 'PENDING·R320/R321 未派' 而非静默省略。",
             "why_not_block": "T1/T3/T4/T5 的实质内容不依赖 R320/R321 的产物；把它们的前置当'产物在盘'来断言"
                              "才是错的（本单正是来堵'声明与制品不符'的）。"},
            {"id": "D-2", "severity": "判据缺陷（任务书字面口径会造成假绿）",
             "finding": "T3-4 字面口径 `ls docs/*取料件*.md` 现盘 = %d 件，其中 1 件是 "
                        "`架构师交接文档_20260921_夜间轮_…_作者Word取料件_…md`（文件名恰含 '取料件' 三字）⇒ "
                        "按字面写守卫会得到 GREEN 的 3，而真实作者取料件只有 %d 件、"
                        "任务书预期的 3（含 R320 产的第三张）从未在盘。且该假绿会在任何人新增一份交接文档时翻转。"
                        % (len(rose_loose), len(rose_precise)),
             "ruling": "G-4 改用**精确口径 + 集合冻结**：作者取料件 = docs/ 下含 '作者取料件' 且不含 '交接文档' 的 md，"
                       "断言其集合 == 退役时刻冻结集合（新增即红）。这严于'恰为 3'（任何新增都红，且不受交接文档计数干扰）。"
                       "同时把 measured 2 / 预期 3 / 差额原因（R320 未派）三条并写入 verdict，不隐藏。"},
            {"id": "D-3", "severity": "任务书内部矛盾",
             "finding": "T3-5 白名单只放 `治理退役声明.md`，而 T6 要求**再新建一份**交接文档 "
                        "`docs/架构师交接文档_20260923_终局轮_治理退役与三单签发.md` ⇒ 按字面执行 T6 必打红 T3-5。",
             "ruling": "白名单扩为恰两件（退役声明 + 终局轮交接文档），并对交接家族加一条**更强**的判据："
                       "此后任何新增 `docs/架构师交接文档_2026*` 一律 RED。净效果严于原文（原只禁'新解释文档'，"
                       "现在连'最后一份之后的第二份交接'也被机器拦住）。"},
            {"id": "D-4", "severity": "T2 判据实测结果（非冲突，是分支判定）",
             "finding": "三条守卫逐条断言已读到行级：r200::test_t4 与 r206::test_t5/t5:85 只做**子串在场**断言；"
                        "但 r257::test_11 的入门侧用 `isec = intro[intro.index('R257'):]`（**切到文件末尾**）"
                        "并要求 5 条磁盘治理纪律词在 isec 内**各恰出现 1 次**。",
             "ruling": "任务书 T2 步骤 2 的前件（'不断言某串只出现一次'）**不成立** ⇒ 走步骤 3："
                       "不改 `项目入门_从这里开始.md`，终局索引由新建 `results/_retired_narrative_20261001/治理退役声明.md` 承载，"
                       "并在索引里登记该文件被守卫锁定的事实与锁定的确切形状（只锁 5 个纪律词的首现窗口）。"},
            {"id": "D-5", "severity": "口径声明",
             "finding": "红面全集需全量 pytest，任务书 §0-3 禁跑 ⇒ 本单只能给下界。",
             "ruling": "按坑 100 三条齐报：口径范围 = RESCAN_SCOPE 七文件（终局方案 §6.2 点名族 ∪ 输入件 4 文件）；"
                       "枚举方式 = pytest -q --tb=no 逐节点 + 失败断言原文回读；下界声明 = 范围外红账本单**不可知**，"
                       "禁把本件读作'全仓只有这些红'。"},
            {"id": "D-6", "severity": "读数时效",
             "finding": "所有'静默/收口/在飞'读数取自预注册时刻；R319 若此后写盘，验收者须现权重取。",
             "ruling": "G-1/G-2/G-3 三条守卫按**现盘**判定（不缓存），本单证据件逐件带 measured_at。"},
            {"id": "D-7", "severity": "属主面（不代修）",
             "finding": "scripts/r281_queue_registry.py 仍输出 next_number = 323（自动续号器活着）。",
             "ruling": "r281 属面，按 R48 登记不代改；机器拦截改由 G-1（新增任务书文件即红）承担——"
                       "续号器给号，但号落地为任务书文件的那一刻必红。"},
        ],

        "write_surface": {
            "allowed": ["results/r322_final_settlement/**", "scripts/r322_*.py",
                        "scripts/r322_common.py", "tests/test_governance_retired.py",
                        "results/_retired_narrative_20261001/治理退役声明.md",
                        "docs/架构师交接文档_20260923_终局轮_治理退役与三单签发.md",
                        "AGENTS.md (仅末尾 append-only 一节)"],
            "forbidden": ["docs/项目入门_从这里开始.md（D-4 判定锁定）",
                          "交付面三件（本单不写交付面，G7 前后双测）", "全部他线 results/ 与 tests/ 属主件",
                          "results/digital_ledger.json（台账零新增）", "tasks/R7R8_线状态看板.md（T3-2 冻结对象）",
                          "models/ data/ E: 盘", "投稿包 / 英文稿 / 复现包 / 导师包（挂起账）"],
            "no_new_scheduled_task": "坑 97：本单零新建计划任务/守护/自启动；零重启任何 *_chain",
        },

        "baseline_at_prereg": {
            "kanban": face(KANBAN), "agents": face(AGENTS), "intro": face(INTRO),
            "ledger": face(LEDGER), "plan": face(PLAN), "handover_last": face(HANDOVER_LAST),
            "delivery_face": [face(p) for p in DELIVERY_FACE],
            "counts": {"tasks_R_md": len(tkb), "handover_docs": len(hov),
                       "extraction_loose": len(rose_loose), "extraction_precise": len(rose_precise),
                       "extraction_loose_list": rose_loose, "extraction_precise_list": rose_precise,
                       "tech_terminal_docs": len(glob.glob(os.path.join(
                           C.ROOT, "docs", u"技术路线_终局_*.md"))),
                       "口径": "ls <glob> | wc -l（与任务书同口径）"},
            "disk": C.df_c(),
            "ufffd_agents": C.ufffd_count(C.read_text(AGENTS)),
            "discipline_counts_intro_isec": {d: C.read_text(INTRO)[
                C.read_text(INTRO).index("R257"):].count(d) for d in DISCIPLINES},
        },

        "discipline_words_must_not_enter_intro_append": list(DISCIPLINES),
        "frozen_anchors_red_line": ["36.687", "12.37", "0.37"],
        "number_discipline_red_line": {
            "锚三点禁作废": "30.50 / 27.01–34.25 / 9.443",
            "两栏严禁混引": "配对增益 0.6769–2.2745° vs 绝对水平 1.3833–4.3738°",
            "12.37 禁入对外面": True,
            "对外零量化承诺": True,
        },
        "watermark": "治理退役机制与红账分账 · 禁入对外承诺文本 · 研究线数字零新增",
    }

    rec = C.dump_json(os.path.join(C.OUT, "prereg.json"), prereg)
    print("prereg:", rec["path"], rec["bytes"], "sha", rec["sha256"][:12])
    print("next_number check:", len(rose_loose), len(rose_precise))


if __name__ == "__main__":
    main()
