"""
gui.py — Desktop GUI for AI English Essay Corrector  (folder → ZIP edition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Select a folder of PDFs → every PDF is corrected → all annotated PDFs are
packed into a single ZIP file.

Layout:
  ┌──────────────────────────────────────────────────────────┐
  │  🎓  AI English Essay Corrector                          │
  ├──────────────────────────────────────────────────────────┤
  │  📁 Input Folder   [C:/essays/          ] [Browse]       │
  │  💾 Output ZIP     [C:/essays_out.zip   ] [Browse]       │
  │     Found: 8 PDF(s)                                      │
  ├──────────────── Settings ────────────────────────────────┤
  │  Model [phi3]   Ollama URL [http://localhost:...]        │
  │  [✓] Use GPU    [ ] Save intermediate JSON files in ZIP  │
  ├──────────────────────────────────────────────────────────┤
  │  [ ▶  Process Folder ]                                   │
  ├──────────────── Overall progress ────────────────────────┤
  │  File 3 / 8 — essay3.pdf          [████████░░░░]  37%    │
  ├──────────────── Per-file steps ──────────────────────────┤
  │  Step 2/3 — Grammar analysis…     [████████████] 100%    │
  ├──────────────── Log ─────────────────────────────────────┤
  │  18:49:07  INFO   essay3.pdf — OCR: 174 words            │
  ├──────────────── Results ─────────────────────────────────┤
  │  File          #err  SPE  GRA  SEM  Status               │
  │  essay1.pdf    12     3    7    2   ✅ done               │
  │  essay2.pdf     5     1    4    0   ✅ done               │
  │  essay3.pdf     —     —    —    —   ⏳ processing…        │
  │                                                          │
  │  [ 📂 Open Output ZIP ]                                  │
  └──────────────────────────────────────────────────────────┘

Run:  python gui.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ── project path ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Colour palette (Catppuccin Mocha)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG      = "#1e1e2e"
BG2     = "#2a2a3e"
BG3     = "#313145"
ACCENT  = "#7c6af7"
ACCENT2 = "#5a4fd4"
FG      = "#cdd6f4"
FG2     = "#a6adc8"
SUCCESS = "#a6e3a1"
WARNING = "#f9e2af"
ERROR_C = "#f38ba8"
BORDER  = "#45475a"

TYPE_COLORS = {
    "spelling": "#f38ba8",
    "grammar":  "#fab387",
    "semantic": "#89b4fa",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GUI log handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _GUIHandler(logging.Handler):
    def __init__(self, cb):
        super().__init__()
        self._cb = cb
        self.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            self._cb(self.format(record), record.levelno)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AI English Essay Corrector — Batch Mode")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(780, 740)

        self._running    = False
        self._zip_path   = ""
        self._pdf_list   = []      # list of Path objects in selected folder

        self._setup_styles()
        self._build_ui()
        self._attach_logging()

        self.update_idletasks()
        w, h = 900, 820
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── styles ────────────────────────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",
                     background=BG, foreground=FG,
                     fieldbackground=BG3, bordercolor=BORDER,
                     troughcolor=BG2, selectbackground=ACCENT,
                     selectforeground=FG, font=("Segoe UI", 10))
        s.configure("TFrame",  background=BG)
        s.configure("TLabel",  background=BG, foreground=FG)
        s.configure("TEntry",  fieldbackground=BG3, foreground=FG,
                               insertcolor=FG, bordercolor=BORDER)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("Accent.TButton",
                     background=ACCENT, foreground="#ffffff",
                     borderwidth=0, focuscolor=ACCENT,
                     font=("Segoe UI", 11, "bold"), padding=(16, 8))
        s.map("Accent.TButton",
              background=[("active", ACCENT2), ("disabled", BORDER)])
        s.configure("TButton",
                     background=BG2, foreground=FG,
                     borderwidth=1, bordercolor=BORDER,
                     font=("Segoe UI", 10), padding=(8, 5))
        s.map("TButton", background=[("active", BG3)])
        s.configure("Accent.Horizontal.TProgressbar",
                     troughcolor=BG3, background=ACCENT,
                     borderwidth=0, thickness=8)
        s.configure("Sub.Horizontal.TProgressbar",
                     troughcolor=BG3, background=ACCENT2,
                     borderwidth=0, thickness=6)
        s.configure("Treeview",
                     background=BG3, foreground=FG,
                     fieldbackground=BG3, borderwidth=0,
                     rowheight=24, font=("Segoe UI", 10))
        s.configure("Treeview.Heading",
                     background=BG2, foreground=FG2,
                     borderwidth=0, font=("Segoe UI", 10, "bold"))
        s.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
        s.configure("Success.TLabel", background=BG,
                     foreground=SUCCESS, font=("Segoe UI", 10))
        s.configure("Warn.TLabel",    background=BG,
                     foreground=WARNING, font=("Segoe UI", 10))
        s.configure("Error.TLabel",   background=BG,
                     foreground=ERROR_C, font=("Segoe UI", 10))

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG2, height=52)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎓  AI English Essay Corrector — Batch Mode",
                 bg=BG2, fg=FG, font=("Segoe UI", 13, "bold")).pack(
            side="left", padx=16, pady=12)
        tk.Label(hdr, text="PyMuPDF · EasyOCR · Phi-3",
                 bg=BG2, fg=FG2, font=("Segoe UI", 9)).pack(
            side="right", padx=16)
        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=16, pady=10)

        # ── folder / zip paths ────────────────────────────────────────────────
        self._build_section(main, "📁  Input / Output")
        paths = tk.Frame(main, bg=BG2)
        paths.pack(fill="x", pady=(0, 8))
        paths.columnconfigure(1, weight=1)

        # Input folder row
        tk.Label(paths, text="Input Folder:", bg=BG2, fg=FG2,
                 font=("Segoe UI", 10), width=13, anchor="e").grid(
            row=0, column=0, padx=(12, 6), pady=6, sticky="e")
        self._folder_var = tk.StringVar()
        self._folder_var.trace_add("write", self._on_folder_changed)
        ttk.Entry(paths, textvariable=self._folder_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(paths, text="Browse…", width=9,
                    command=self._browse_folder).grid(
            row=0, column=2, padx=(0, 12))

        # PDF count label
        self._pdf_count_lbl = tk.Label(paths, text="No folder selected",
                                        bg=BG2, fg=FG2,
                                        font=("Segoe UI", 9, "italic"))
        self._pdf_count_lbl.grid(row=1, column=1, sticky="w", padx=(0, 6),
                                  pady=(0, 4))

        # Output ZIP row
        tk.Label(paths, text="Output ZIP:", bg=BG2, fg=FG2,
                 font=("Segoe UI", 10), width=13, anchor="e").grid(
            row=2, column=0, padx=(12, 6), pady=6, sticky="e")
        self._zip_var = tk.StringVar()
        ttk.Entry(paths, textvariable=self._zip_var).grid(
            row=2, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(paths, text="Browse…", width=9,
                    command=self._browse_zip).grid(
            row=2, column=2, padx=(0, 12), pady=(0, 8))

        # ── settings ──────────────────────────────────────────────────────────
        self._build_section(main, "⚙️  Settings")
        cfg = tk.Frame(main, bg=BG2)
        cfg.pack(fill="x", pady=(0, 8))
        cfg.columnconfigure(3, weight=1)

        tk.Label(cfg, text="Model:", bg=BG2, fg=FG2,
                 font=("Segoe UI", 10)).grid(
            row=0, column=0, sticky="w", padx=(12, 4), pady=6)
        self._model_var = tk.StringVar(value="phi3")
        ttk.Entry(cfg, textvariable=self._model_var, width=14).grid(
            row=0, column=1, sticky="w", padx=(0, 16))

        tk.Label(cfg, text="Ollama URL:", bg=BG2, fg=FG2,
                 font=("Segoe UI", 10)).grid(
            row=0, column=2, sticky="w", padx=(0, 4))
        self._url_var = tk.StringVar(
            value="http://localhost:11434/api/generate")
        ttk.Entry(cfg, textvariable=self._url_var, width=38).grid(
            row=0, column=3, sticky="ew", padx=(0, 12))

        self._gpu_var  = tk.BooleanVar(value=True)
        self._json_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg, text="Use GPU  (EasyOCR)",
                         variable=self._gpu_var).grid(
            row=1, column=0, columnspan=2, sticky="w",
            padx=(12, 0), pady=(0, 8))
        ttk.Checkbutton(cfg,
                         text="Include intermediate JSON files in ZIP",
                         variable=self._json_var).grid(
            row=1, column=2, columnspan=2, sticky="w",
            padx=(0, 12), pady=(0, 8))

        # ── run button + status ───────────────────────────────────────────────
        btn_row = ttk.Frame(main)
        btn_row.pack(fill="x", pady=(0, 10))
        self._run_btn = ttk.Button(btn_row, text="▶   Process Folder",
                                    style="Accent.TButton",
                                    command=self._on_run)
        self._run_btn.pack(side="left")
        self._status_lbl = ttk.Label(btn_row, text="", style="Success.TLabel")
        self._status_lbl.pack(side="left", padx=16)

        # ── overall progress ──────────────────────────────────────────────────
        self._build_section(main, "⏳  Progress")
        prog = tk.Frame(main, bg=BG2)
        prog.pack(fill="x", pady=(0, 4))

        self._overall_lbl = tk.Label(prog, text="Waiting…",
                                      bg=BG2, fg=FG2,
                                      font=("Segoe UI", 9), anchor="w")
        self._overall_lbl.pack(fill="x", padx=12, pady=(6, 2))
        self._overall_bar = ttk.Progressbar(
            prog, style="Accent.Horizontal.TProgressbar",
            mode="determinate", maximum=100)
        self._overall_bar.pack(fill="x", padx=12, pady=(0, 4))

        self._step_lbl = tk.Label(prog, text="",
                                   bg=BG2, fg=FG2,
                                   font=("Segoe UI", 9), anchor="w")
        self._step_lbl.pack(fill="x", padx=12, pady=(2, 2))
        self._step_bar = ttk.Progressbar(
            prog, style="Sub.Horizontal.TProgressbar",
            mode="determinate", maximum=100)
        self._step_bar.pack(fill="x", padx=12, pady=(0, 8))

        # ── log ───────────────────────────────────────────────────────────────
        self._build_section(main, "📋  Log")
        log_f = tk.Frame(main, bg=BG3)
        log_f.pack(fill="x", pady=(0, 8))
        self._log_text = tk.Text(log_f, height=6, bg=BG3, fg=FG2,
                                  font=("Consolas", 9), bd=0,
                                  relief="flat", state="disabled",
                                  wrap="word", insertbackground=FG)
        lsb = ttk.Scrollbar(log_f, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._log_text.tag_config("INFO",    foreground=FG2)
        self._log_text.tag_config("WARNING", foreground=WARNING)
        self._log_text.tag_config("ERROR",   foreground=ERROR_C)
        self._log_text.tag_config("DEBUG",   foreground=BORDER)

        # ── results table ─────────────────────────────────────────────────────
        self._build_section(main, "🔍  Results per File")
        tree_f = tk.Frame(main, bg=BG3)
        tree_f.pack(fill="both", expand=True, pady=(0, 8))

        cols = ("file", "errors", "spe", "gra", "sem", "status")
        self._tree = ttk.Treeview(tree_f, columns=cols,
                                   show="headings", selectmode="browse")
        for col, w, anc, label in [
            ("file",   260, "w",      "File"),
            ("errors",  52, "center", "Errors"),
            ("spe",     48, "center", "SPE"),
            ("gra",     48, "center", "GRA"),
            ("sem",     48, "center", "SEM"),
            ("status", 130, "w",      "Status"),
        ]:
            self._tree.heading(col, text=label)
            self._tree.column(col, width=w, anchor=anc, minwidth=30)

        tsb = ttk.Scrollbar(tree_f, command=self._tree.yview)
        self._tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)

        self._tree.tag_configure("done",    foreground=SUCCESS)
        self._tree.tag_configure("active",  foreground=WARNING)
        self._tree.tag_configure("error",   foreground=ERROR_C)
        self._tree.tag_configure("waiting", foreground=FG2)

        # ── bottom bar ────────────────────────────────────────────────────────
        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(0, 4))
        self._open_btn = ttk.Button(bottom, text="📂  Open Output ZIP",
                                     command=self._open_zip,
                                     state="disabled")
        self._open_btn.pack(side="left")
        self._summary_lbl = ttk.Label(bottom, text="", foreground=FG2)
        self._summary_lbl.pack(side="right")

    def _build_section(self, parent, text: str):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(6, 2))
        tk.Label(row, text=text, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Frame(row, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=8)

    # ── logging ───────────────────────────────────────────────────────────────

    def _attach_logging(self):
        h = _GUIHandler(self._append_log)
        h.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(h)
        logging.getLogger().setLevel(logging.DEBUG)

    def _append_log(self, msg: str, level: int):
        tag = {logging.DEBUG: "DEBUG", logging.INFO: "INFO",
               logging.WARNING: "WARNING", logging.ERROR: "ERROR"}.get(
            level, "INFO")
        def _ins():
            self._log_text.configure(state="normal")
            self._log_text.insert("end", msg + "\n", tag)
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        self.after(0, _ins)

    # ── folder / zip browsing ─────────────────────────────────────────────────

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select folder containing PDFs")
        if path:
            self._folder_var.set(path)

    def _on_folder_changed(self, *_):
        folder = self._folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            self._pdf_list = []
            self._pdf_count_lbl.config(text="No folder selected")
            return
        pdfs = sorted(Path(folder).glob("*.pdf"))
        self._pdf_list = pdfs
        n = len(pdfs)
        self._pdf_count_lbl.config(
            text=f"Found {n} PDF{'s' if n != 1 else ''}" if n else
                 "⚠  No PDF files found in this folder")
        # auto-fill ZIP path
        if not self._zip_var.get():
            self._zip_var.set(str(Path(folder).parent /
                                  (Path(folder).name + "_annotated.zip")))
        # pre-populate table
        self._tree.delete(*self._tree.get_children())
        for p in pdfs:
            self._tree.insert("", "end",
                               iid=p.name,
                               values=(p.name, "—", "—", "—", "—", "⏸ waiting"),
                               tags=("waiting",))

    def _browse_zip(self):
        path = filedialog.asksaveasfilename(
            title="Save output ZIP as",
            defaultextension=".zip",
            filetypes=[("ZIP archives", "*.zip")])
        if path:
            self._zip_var.set(path)

    # ── open output zip ───────────────────────────────────────────────────────

    def _open_zip(self):
        if not self._zip_path or not Path(self._zip_path).exists():
            messagebox.showwarning("Not found", "Output ZIP not found.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(self._zip_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self._zip_path])
            else:
                subprocess.Popen(["xdg-open", self._zip_path])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── progress helpers ──────────────────────────────────────────────────────

    def _set_overall(self, label: str, pct: int):
        def _u():
            self._overall_lbl.config(text=label)
            self._overall_bar["value"] = pct
        self.after(0, _u)

    def _set_step(self, label: str, pct: int):
        def _u():
            self._step_lbl.config(text=label)
            self._step_bar["value"] = pct
        self.after(0, _u)

    def _set_status(self, msg: str, style: str = "Success.TLabel"):
        self.after(0, lambda: self._status_lbl.config(text=msg, style=style))

    def _update_row(self, name: str, errors: list | None,
                    status: str, tag: str):
        def _u():
            if errors is not None:
                sp = sum(1 for e in errors if e.get("type") == "spelling")
                gr = sum(1 for e in errors if e.get("type") == "grammar")
                se = sum(1 for e in errors if e.get("type") == "semantic")
                self._tree.item(name, values=(
                    name, len(errors), sp, gr, se, status), tags=(tag,))
            else:
                cur = list(self._tree.item(name, "values"))
                cur[5] = status
                self._tree.item(name, values=cur, tags=(tag,))
            self._tree.see(name)
        self.after(0, _u)

    # ── run ───────────────────────────────────────────────────────────────────

    def _on_run(self):
        folder = self._folder_var.get().strip()
        zip_out = self._zip_var.get().strip()

        if not folder or not Path(folder).is_dir():
            messagebox.showwarning("No folder", "Please select an input folder.")
            return
        if not self._pdf_list:
            messagebox.showwarning("No PDFs", "No PDF files found in the folder.")
            return
        if not zip_out:
            messagebox.showwarning("No output", "Please specify an output ZIP path.")
            return
        if self._running:
            return

        # reset UI
        self._open_btn.config(state="disabled")
        self._set_status("Running…", "Warn.TLabel")
        self._run_btn.config(state="disabled")
        self._zip_path = ""
        self._summary_lbl.config(text="")

        # reset table rows to waiting
        for p in self._pdf_list:
            self._tree.item(p.name,
                             values=(p.name, "—", "—", "—", "—", "⏸ waiting"),
                             tags=("waiting",))

        threading.Thread(target=self._run_batch, daemon=True).start()

    def _run_batch(self):
        self._running = True
        t0 = time.time()
        log = logging.getLogger("gui")

        try:
            from ocr_engine import OCREngine
            from nlp_engine  import NLPEngine
            from annotator   import Annotator

            use_gpu   = self._gpu_var.get()
            model     = self._model_var.get().strip() or "phi3"
            url       = self._url_var.get().strip()
            save_json = self._json_var.get()
            zip_out   = self._zip_var.get().strip()
            pdfs      = self._pdf_list
            total     = len(pdfs)

            # Initialise engines once (avoid reloading model per file)
            self._set_overall("Initialising engines…", 0)
            self._set_step("", 0)
            ocr = OCREngine(use_gpu=use_gpu)
            nlp = NLPEngine(model=model, ollama_url=url)
            ann = Annotator()

            succeeded, failed = 0, 0
            all_errors = 0

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)

                for idx, pdf_path in enumerate(pdfs):
                    name = pdf_path.name
                    file_pct = int(idx / total * 100)
                    self._set_overall(
                        f"File {idx+1}/{total} — {name}", file_pct)
                    self._update_row(name, None, "⏳ processing…", "active")
                    log.info(f"━━━ [{idx+1}/{total}] {name} ━━━")

                    try:
                        # Step 1 — OCR
                        self._set_step("Step 1/3 — OCR…", 5)
                        ocr_words = ocr.process_pdf(str(pdf_path))
                        total_w   = sum(len(v) for v in ocr_words.values())
                        self._set_step(
                            f"Step 1/3 — OCR done  ({total_w} words)", 33)

                        if total_w == 0:
                            raise ValueError("OCR found 0 words")

                        # Step 2 — NLP
                        self._set_step("Step 2/3 — Grammar analysis…", 38)
                        errors = nlp.analyse(ocr_words)
                        self._set_step(
                            f"Step 2/3 — {len(errors)} error(s) found", 66)

                        # Step 3 — Annotate
                        self._set_step("Step 3/3 — Annotating PDF…", 70)
                        out_pdf = str(tmp / (pdf_path.stem + "_annotated.pdf"))
                        ann.annotate_pdf(str(pdf_path), ocr_words,
                                         errors, out_pdf)
                        self._set_step("Step 3/3 — Done", 100)

                        # Optional JSON files
                        if save_json:
                            ocr_f = tmp / (pdf_path.stem + "_ocr.json")
                            err_f = tmp / (pdf_path.stem + "_errors.json")
                            ocr_f.write_text(
                                json.dumps(ocr_words, ensure_ascii=False,
                                           indent=2), encoding="utf-8")
                            err_f.write_text(
                                json.dumps(errors, ensure_ascii=False,
                                           indent=2), encoding="utf-8")

                        self._update_row(name, errors, "✅ done", "done")
                        all_errors += len(errors)
                        succeeded  += 1

                    except Exception as e:
                        log.error(f"  {name} failed: {e}", exc_info=False)
                        self._update_row(name, None, f"❌ {e}", "error")
                        failed += 1

                # Pack everything into ZIP
                self._set_overall("Packing ZIP…", 99)
                self._set_step("", 0)
                with zipfile.ZipFile(zip_out, "w",
                                     compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in tmp.iterdir():
                        zf.write(f, f.name)

            self._zip_path = zip_out
            elapsed = time.time() - t0
            summary = (f"✅  {succeeded}/{total} files  |  "
                       f"{all_errors} total errors  |  {elapsed:.0f}s")
            self._set_overall(f"Complete — {succeeded}/{total} files OK", 100)
            self._set_status(summary, "Success.TLabel")
            self.after(0, lambda: self._summary_lbl.config(
                text=f"{succeeded} OK  {failed} failed"))
            self.after(0, lambda: self._open_btn.config(state="normal"))
            log.info(f"ZIP saved → {zip_out}  ({succeeded} files, {elapsed:.1f}s)")

        except Exception as e:
            log.error(f"Batch failed: {e}", exc_info=True)
            self._set_status(f"❌  {e}", "Error.TLabel")
            self._set_overall("Failed", 0)
        finally:
            self._running = False
            self.after(0, lambda: self._run_btn.config(state="normal"))


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()