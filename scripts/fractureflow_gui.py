# -*- coding: utf-8 -*-
"""FractureFlow 极简桌面 GUI — 消灭命令行门槛.

窗口 800x600, 三个控件纵向排列:
  顶部: 标签 + 「选择项目文件夹」按钮
  中部: 进度条 + 日志文本框 (带滚动条)
  底部: 「开始处理」按钮 + 「打开报告」按钮 (处理完之前禁用)

后台线程跑处理, root.after(100, poll) 轮询同步.
处理逻辑: spawn 子进程执行 scripts/auto_deliver_pipeline.py (不是进程内函数调用)。

⚠ 数据口径 (guichain_20260925 实测确认, 客户可见):
  被调脚本没有输入通道 —— 它固定读仓内 data/real/loaded_real_nets_setid.pt,
  用户选的文件夹只用作输出目录。因此:
    1. 选中的文件夹里若存在疑似编录表 (.csv/.xlsx/.xls) ⇒ 本 GUI 拒绝生成报告
       (preflight_demo_guard), 因为那些表一字不会被读;
    2. 文件夹为空 (纯输出) ⇒ 允许跑, 但日志、HTML 报告、弹窗、per-well 报告首部
       全部明示「演示口径 · 非你提供的数据」。
  要处理自己的编录表, 走现役命令行入口 (scripts/borehole_excel_entry.py +
  scripts/auto_label_borehole.py); 该链路未接入本 GUI。

用法:
  python scripts/fractureflow_gui.py
"""
import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 本 GUI 用 encoding='utf-8' 解码子进程 stdout, 但中文 Windows 的 cmd 默认码是 cp936:
# 不设此变量则子进程按 gbk 发射, 日志区整片乱码 (guicli_20260925 G5 实测 12/20 行含 U+FFFD)。
# 只管子进程启动时的解释器配置, 不改任何计算口径或默认参数。
os.environ.setdefault("PYTHONUTF8", "1")

# ---------------------------------------------------------------------------
# 项目路径常量
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = "E:/app2/anacoda/Anacoda/envs/pytorch_env/python.exe"
PIPELINE_SCRIPT = os.path.join(ROOT, "scripts", "auto_deliver_pipeline.py")

# ---------------------------------------------------------------------------
# 演示口径守卫 (guichain_20260925 T1)
# ---------------------------------------------------------------------------
# 事实: 本 GUI 唯一的子进程目标 auto_deliver_pipeline.py **没有输入通道** —— 它从仓内
# data/real/loaded_real_nets_setid.pt 取数, 客户选的文件夹只用作 out_dir。所以"选了自己的
# 文件夹"和"拿到关于自己数据的报告"这两件事之间从来没有连接。守卫判定单一实现放在被包件
# (数据真相在那一侧), GUI 只做调用与拒办。
# 惰性导入: 被包件 import 时就拉 torch, 模块级导入会把 GUI 冷启动拖慢数秒 ⇒ 首次点击
# "开始处理"时再导 (那次点击后面本来就要 spawn 一个 python 子进程)。
sys.path.insert(0, os.path.join(ROOT, "scripts"))
_GUARD = {"mod": None, "error": None, "tried": False}


def _guard_module():
    """载入并缓存守卫实现; 失败即记 error, 调用方必须 fail closed。"""
    if not _GUARD["tried"]:
        _GUARD["tried"] = True
        try:
            import auto_deliver_pipeline as _adp
            for attr in ("DEMO_DISCLOSURE", "DEMO_DISCLOSURE_TOKEN", "MAE_CALIBER_LABEL",
                         "SOURCE_REL", "find_candidate_input_tables"):
                if not hasattr(_adp, attr):
                    raise AttributeError("守卫缺属性 %s" % attr)
            _GUARD["mod"] = _adp
        except Exception as e:
            _GUARD["error"] = "%s: %s" % (type(e).__name__, e)
    return _GUARD["mod"], _GUARD["error"]


# 导入失败时的兜底文案 (此时一律拒办, 所以它只需说明"为什么拒", 不需描述数据)
_GUARD_FAIL_REASON = ("守卫模块 auto_deliver_pipeline 载入失败 (%s), 已拒绝启动 —— "
                      "宁可不跑, 也不产出一份没有数据出处的报告。")
_DEMO_TITLE_PREFIX = "演示·仓内示例数据"


def preflight_demo_guard(folder, out_subdir="fractureflow_output"):
    """启动前判定: (ok, reason, candidate_tables)。纯逻辑, 不碰 Tk ⇒ 可被机检直接调用。

    规则 (INV-1 的前置条件): 文件夹里存在**任何**疑似编录表 ⇒ 不启动子进程。
    理由: 那些表一字不会被读, 而产物看起来像一份交付报告 —— 这正是必须消灭的失效形态。
    空文件夹 (纯输出用途) ⇒ 允许跑, 但日志与产物必须全程带演示声明。
    """
    adp, err = _guard_module()
    if adp is None:
        return (False, _GUARD_FAIL_REASON % err, [])
    if not folder or not os.path.isdir(folder):
        return (False, "文件夹不存在或不可读: %s" % folder, [])
    cands = adp.find_candidate_input_tables(folder, exclude_subdirs=(out_subdir,))
    if cands:
        return (False,
                "本入口**没有客户数据输入通道**: 它取的是仓内示例数据 (%s), 你选的文件夹只用作输出目录。"
                "检测到 %d 个疑似编录表文件一字未被读取, 因此拒绝生成报告 (否则你会拿到一份"
                "看起来描述你自己数据、实际描述别人数据的报告):\n  %s\n"
                "要处理你自己的编录表, 请用现役命令行入口: scripts/borehole_excel_entry.py --input <表> "
                "再接 scripts/auto_label_borehole.py --data real --real-path <产出的 .pt> --out-dir <目录> "
                "(该链路目前未接入本 GUI, 且需要组数 K 等口径决策)。"
                % (adp.SOURCE_REL, len(cands), "\n  ".join(cands[:10])),
                cands)
    return (True, "", [])

# ---------------------------------------------------------------------------
# GUI 主类
# ---------------------------------------------------------------------------
class FractureFlowGUI:
    """FractureFlow 极简桌面 GUI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FractureFlow — 结构面统计一键出报")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # 状态变量
        self.folder_path = tk.StringVar(value="")
        self.report_path = tk.StringVar(value="")
        self.worker_thread = None
        self.is_running = False
        self.poll_id = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI 构建
    # -----------------------------------------------------------------------
    def _build_ui(self):
        """构建三个控件区域."""
        pad = {"padx": 20, "pady": 8}

        # ---- 顶部: 文件夹选择 ----
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, **pad)

        tk.Label(
            top_frame, text="项目文件夹:", font=("Microsoft YaHei", 11),
        ).pack(side=tk.LEFT)

        self.folder_label = tk.Label(
            top_frame, text="(未选择)", fg="gray",
            font=("Microsoft YaHei", 10), anchor="w", width=40,
            wraplength=420, justify=tk.LEFT,
        )
        self.folder_label.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)

        tk.Button(
            top_frame, text="选择项目文件夹", font=("Microsoft YaHei", 10),
            command=self._choose_folder, width=16, height=1,
        ).pack(side=tk.RIGHT)

        # ---- 中部: 进度条 + 日志 ----
        mid_frame = tk.Frame(self.root)
        mid_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.progress = ttk.Progressbar(
            mid_frame, mode="indeterminate", length=760,
        )
        self.progress.pack(fill=tk.X, pady=(0, 8))

        log_frame = tk.Frame(mid_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_frame, font=("Consolas", 9), wrap=tk.WORD,
            state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", selectbackground="#264f78",
        )
        scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview,
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- 底部: 操作按钮 ----
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, **pad)

        self.start_btn = tk.Button(
            bottom_frame, text="开始处理", font=("Microsoft YaHei", 12, "bold"),
            command=self._start_processing, width=16, height=1,
            bg="#2a9d8f", fg="white", activebackground="#21867a",
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 20))

        self.open_btn = tk.Button(
            bottom_frame, text="打开报告", font=("Microsoft YaHei", 12),
            command=self._open_report, width=16, height=1,
            state=tk.DISABLED,
        )
        self.open_btn.pack(side=tk.LEFT)

    # -----------------------------------------------------------------------
    # 日志输出
    # -----------------------------------------------------------------------
    def _log(self, message: str):
        """向日志文本框追加一行."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_clear(self):
        """清空日志文本框."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # 文件夹选择
    # -----------------------------------------------------------------------
    def _choose_folder(self):
        """弹出目录选择对话框."""
        folder = filedialog.askdirectory(
            title="选择项目文件夹",
            initialdir=ROOT,
        )
        if folder:
            self.folder_path.set(folder)
            self.folder_label.configure(text=folder, fg="black")
            self._log(f"[选择] {folder}")

    # -----------------------------------------------------------------------
    # 开始处理
    # -----------------------------------------------------------------------
    def _start_processing(self):
        """启动后台线程跑管线 (启动前先过演示口径守卫)."""
        if self.is_running:
            return

        folder = self.folder_path.get()
        if not folder:
            messagebox.showwarning("提示", "请先选择项目文件夹")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("错误", f"文件夹不存在:\n{folder}")
            return

        # ---- 守卫: 文件夹里有疑似编录表 ⇒ 拒办, 不 spawn (INV-1 前置) ----
        ok, reason, _cands = preflight_demo_guard(folder)
        if not ok:
            self._log_clear()
            self._log("=" * 50)
            self._log("[被拒] " + reason)
            self._log("=" * 50)
            messagebox.showerror("本入口不读客户数据 · 已拒绝生成报告", reason)
            return

        # 准备 UI
        self.is_running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.open_btn.configure(state=tk.DISABLED)
        self.report_path.set("")
        self.progress.start(10)
        self._log_clear()
        self._log("=" * 50)
        self._log("处理开始")
        self._log(f"项目文件夹: {folder}")
        self._log("=" * 50)
        # ---- 演示声明: 无条件先说, 且说不止一次 (日志区第一屏就要看得见) ----
        adp, err = _guard_module()
        if adp is None:
            self._log("[被拒] 守卫不可用: %s" % err)
            self.is_running = False
            self.progress.stop()
            self.start_btn.configure(state=tk.NORMAL)
            messagebox.showerror("守卫不可用", _GUARD_FAIL_REASON % err)
            return
        self._log("⚠ " + adp.DEMO_DISCLOSURE)
        self._log("⚠ 该文件夹在本流程里的唯一作用 = 存放输出 (fractureflow_output/)。")
        self._log("⚠ 取数文件: %s" % adp.SOURCE_REL)

        # 启动后台线程
        self.worker_thread = threading.Thread(
            target=self._run_pipeline, args=(folder,), daemon=True,
        )
        self.worker_thread.start()

        # 启动轮询
        self.poll_id = self.root.after(100, self._poll_worker)

    def _run_pipeline(self, folder: str):
        """后台线程: 调用 auto_deliver_pipeline.py."""
        try:
            out_dir = os.path.join(folder, "fractureflow_output")
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                PYTHON_EXE, PIPELINE_SCRIPT,
                "--dataset", "all",
                "--out-dir", out_dir,
                "--max-wells", "5",
            ]
            self._append_result("__cmd__", " ".join(cmd))

            proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                self._append_result("__log__", line)
            proc.wait()

            if proc.returncode != 0:
                self._append_result("__error__", f"管线退出码 {proc.returncode}")
                return

            # 查找生成的报告
            summary_path = os.path.join(out_dir, "pipeline_summary.json")
            if os.path.isfile(summary_path):
                import json as _json
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = _json.load(f)
                # 找第一份报告作为代表
                for grade in ("A", "B", "C"):
                    items = summary.get("grade_details", {}).get(grade, [])
                    if items and items[0].get("report"):
                        self._append_result("__report__", items[0]["report"])
                        break

                # 将汇总写入 HTML 报告
                html_path = self._make_html_report(summary, out_dir)
                if html_path:
                    self._append_result("__report__", html_path)

            adp, _err = _guard_module()
            done_msg = ("处理完成 · 报告描述的是仓内示例数据 %s, 非你提供的数据"
                        % (adp.SOURCE_REL if adp else "取数文件"))
            self._append_result("__done__", done_msg)

        except Exception as e:
            self._append_result("__error__", str(e))

    def _make_html_report(self, summary: dict, out_dir: str) -> str:
        """从管线汇总生成 HTML 报告.

        首部强制演示声明卡 + 数据溯源块: 日志区会滚掉, HTML 不会 —— 客户打开的那一件
        必须自己说清数据是谁的。溯源字段从 pipeline_summary.json 直读, 不在本件二次手打路径。
        """
        try:
            dataset = summary.get("dataset", "unknown")
            n_wells = summary.get("n_wells", 0)
            grade_dist = summary.get("grade_distribution", {})
            prov = summary.get("data_provenance") or {}
            adp, _err = _guard_module()
            src_rel = prov.get("source_file_rel") or (adp.SOURCE_REL if adp else "未核对")
            well_ids = prov.get("well_ids_in_report") or [
                it.get("wid", "") for g in ("A", "B", "C")
                for it in summary.get("grade_details", {}).get(g, [])
            ]

            disclosure_html = (
                "<div class='warn'>"
                "<b>⚠ 演示口径 · 本报告非你提供的数据</b><br>"
                "数据来源 = 仓内示例文件 <code>%s</code> (dataset 过滤 <code>%s</code>, 取 %s 口井);"
                "你选择的文件夹在本流程里只用作<b>输出目录</b>, 其中的任何表格文件都没有被读取。<br>"
                "报告中的 MAE 口径: %s"
                "</div>" % (src_rel, prov.get("dataset_filter", dataset), n_wells,
                            (adp.MAE_CALIBER_LABEL if adp else "未核对"))
            )

            rows_html = ""
            for grade in ("A", "B", "C"):
                for item in summary.get("grade_details", {}).get(grade, []):
                    rows_html += (
                        f"<tr><td>{item.get('wid', '')}</td>"
                        f"<td><span class='grade grade-{grade}'>{grade}</span></td>"
                        f"<td>{item.get('mae', '')}</td>"
                        f"<td>{item.get('n_groups', '')}</td>"
                        f"<td>{item.get('boundary_frac', '')}</td></tr>\n"
                    )

            html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{_DEMO_TITLE_PREFIX} — FractureFlow 报告 — {dataset}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; background: #f5f5f5; }}
h1 {{ color: #2a9d8f; }}
.warn {{ background: #fff3cd; border: 2px solid #e9a23b; border-radius: 8px; padding: 16px 20px; margin: 16px 0; color: #5c3d00; line-height: 1.7; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #2a9d8f; color: white; }}
.grade {{ padding: 2px 10px; border-radius: 10px; color: white; font-weight: bold; }}
.grade-A {{ background: #2a9d8f; }}
.grade-B {{ background: #e9a23b; }}
.grade-C {{ background: #e76f51; }}
</style>
</head>
<body>
<h1>FractureFlow 结构面统计报告 <small style="color:#e76f51;">[{_DEMO_TITLE_PREFIX}]</small></h1>
{disclosure_html}
<div class="card">
<h2>数据溯源</h2>
<p><b>取数文件:</b> <code>{src_rel}</code>（仓内示例数据）</p>
<p><b>是否读取了你选择的文件夹:</b> {'否 (仅用作输出目录)' if prov.get('reads_user_input', False) is False else '是'}</p>
<p><b>报告所描述的井:</b> {', '.join(str(w) for w in well_ids) if well_ids else '(无)'}</p>
</div>
<div class="card">
<h2>汇总</h2>
<p><b>数据集:</b> {dataset}</p>
<p><b>井数:</b> {n_wells}</p>
<p><b>分级分布:</b> A={grade_dist.get('A', 0)}  B={grade_dist.get('B', 0)}  C={grade_dist.get('C', 0)}</p>
</div>
<div class="card">
<h2>井详情</h2>
<table>
<tr><th>井 ID</th><th>级别</th><th>MAE (°)</th><th>组数</th><th>边界占比</th></tr>
{rows_html}
</table>
</div>
<p style="color:#999;font-size:12px;">Generated by FractureFlow GUI · 演示口径 (数据源 {src_rel}) · 非你提供的数据</p>
</body>
</html>"""
            html_path = os.path.join(out_dir, "fractureflow_report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            return html_path
        except Exception as e:
            self._append_result("__error__", f"HTML 报告生成失败: {e}")
            return ""

    # -----------------------------------------------------------------------
    # 线程安全的结果传递 (简单 list + after 轮询)
    # -----------------------------------------------------------------------
    def _append_result(self, kind: str, data: str):
        """后台线程调用: 向结果队列追加."""
        if not hasattr(self, "_result_queue"):
            self._result_queue = []
        self._result_queue.append((kind, data))

    def _poll_worker(self):
        """主线程 100ms 轮询: 消费结果 + 检测线程结束."""
        # 消费所有待处理结果
        queue = getattr(self, "_result_queue", [])
        self._result_queue = []

        for kind, data in queue:
            if kind == "__log__":
                self._log(data)
            elif kind == "__cmd__":
                self._log(f"$ {data}")
            elif kind == "__error__":
                self._log(f"[ERROR] {data}")
            elif kind == "__report__":
                self.report_path.set(data)
                self._log(f"[报告] {data}")
            elif kind == "__done__":
                self._log("[完成] 处理完毕")

        if self.worker_thread and self.worker_thread.is_alive():
            # 继续轮询
            self.poll_id = self.root.after(100, self._poll_worker)
        else:
            # 线程结束
            self._on_finished()

    def _on_finished(self):
        """处理完成回调."""
        self.is_running = False
        self.progress.stop()
        self.start_btn.configure(state=tk.NORMAL)

        report = self.report_path.get()
        if report and os.path.isfile(report):
            self.open_btn.configure(state=tk.NORMAL)
            adp, _err = _guard_module()
            _decl = adp.DEMO_DISCLOSURE if adp else "取数来源未核对, 请勿当作你方数据的报告。"
            messagebox.showinfo(
                "完成 (演示口径)",
                "处理完成!\n报告已生成:\n%s\n\n%s" % (report, _decl)
            )
        elif hasattr(self, "_result_queue") and any(
            k == "__error__" for k, _ in getattr(self, "_result_queue", [])
        ):
            messagebox.showerror("错误", "处理过程中发生错误, 请查看日志")
        else:
            messagebox.showwarning("警告", "处理完成但未找到报告文件")

    # -----------------------------------------------------------------------
    # 打开报告
    # -----------------------------------------------------------------------
    def _open_report(self):
        """用系统默认程序打开报告."""
        report = self.report_path.get()
        if not report or not os.path.isfile(report):
            messagebox.showwarning("提示", "报告文件不存在")
            return
        try:
            os.startfile(report)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开报告:\n{e}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    FractureFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
