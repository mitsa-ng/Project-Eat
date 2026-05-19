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
# Colour palette
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG = "#10131a"
SURFACE = "#171b24"
SURFACE_2 = "#202633"
SURFACE_3 = "#273040"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#3d73d6"
FG = "#edf2ff"
FG_MUTED = "#aeb8cc"
SUCCESS = "#7bd88f"
WARNING = "#f6c768"
ERROR_C = "#ff7d90"
BORDER = "#354052"

FONT_UI = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SECTION = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 9)

TYPE_COLORS = {
    "spelling": ERROR_C,
    "grammar": WARNING,
    "semantic": "#8ab4ff",
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
        self._live_running = False
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
                    fieldbackground=SURFACE_3, bordercolor=BORDER,
                    troughcolor=SURFACE_2, selectbackground=ACCENT,
                    selectforeground="#ffffff", font=FONT_UI)
        s.configure("TFrame", background=BG)
        s.configure("Surface.TFrame", background=SURFACE)
        s.configure("Toolbar.TFrame", background=SURFACE_2)
        s.configure("TLabel", background=BG, foreground=FG, font=FONT_UI)
        s.configure("Muted.TLabel", background=SURFACE, foreground=FG_MUTED,
                    font=FONT_SMALL)
        s.configure("Title.TLabel", background=SURFACE_2, foreground=FG,
                    font=FONT_TITLE)
        s.configure("Subtitle.TLabel", background=SURFACE_2,
                    foreground=FG_MUTED, font=FONT_SMALL)
        s.configure("TEntry", fieldbackground=SURFACE_3, foreground=FG,
                    insertcolor=FG, bordercolor=BORDER, lightcolor=BORDER,
                    darkcolor=BORDER, padding=(8, 5))
        s.configure("TCheckbutton", background=SURFACE, foreground=FG,
                    font=FONT_UI)
        s.map("TCheckbutton",
              background=[("active", SURFACE)],
              foreground=[("disabled", FG_MUTED)])
        s.configure("TRadiobutton", background=SURFACE, foreground=FG,
                    font=FONT_UI)
        s.map("TRadiobutton",
              background=[("active", SURFACE)],
              foreground=[("disabled", FG_MUTED)])
        s.configure("Accent.TButton",
                    background=ACCENT, foreground="#ffffff",
                    borderwidth=0, focuscolor=ACCENT,
                    font=("Segoe UI", 11, "bold"), padding=(18, 9))
        s.map("Accent.TButton",
              background=[("active", ACCENT_HOVER), ("disabled", BORDER)],
              foreground=[("disabled", FG_MUTED)])
        s.configure("Secondary.TButton",
                    background=SURFACE_2, foreground=FG,
                    borderwidth=1, bordercolor=BORDER,
                    font=FONT_UI, padding=(12, 8))
        s.map("Secondary.TButton",
              background=[("active", SURFACE_3), ("disabled", SURFACE)],
              foreground=[("disabled", FG_MUTED)])
        s.configure("Danger.TButton",
                    background=SURFACE_2, foreground=ERROR_C,
                    borderwidth=1, bordercolor=BORDER,
                    font=FONT_UI, padding=(12, 8))
        s.map("Danger.TButton",
              background=[("active", SURFACE_3), ("disabled", SURFACE)],
              foreground=[("disabled", FG_MUTED)])
        s.configure("TButton",
                    background=SURFACE_2, foreground=FG,
                    borderwidth=1, bordercolor=BORDER,
                    font=FONT_UI, padding=(10, 6))
        s.map("TButton", background=[("active", SURFACE_3)])
        s.configure("Accent.Horizontal.TProgressbar",
                    troughcolor=SURFACE_3, background=ACCENT,
                    borderwidth=0, thickness=10)
        s.configure("Sub.Horizontal.TProgressbar",
                    troughcolor=SURFACE_3, background=ACCENT_HOVER,
                    borderwidth=0, thickness=7)
        s.configure("Treeview",
                    background=SURFACE_3, foreground=FG,
                    fieldbackground=SURFACE_3, borderwidth=0,
                    rowheight=28, font=FONT_UI)
        s.configure("Treeview.Heading",
                    background=SURFACE_2, foreground=FG_MUTED,
                    borderwidth=0, font=FONT_SECTION)
        s.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
        s.configure("Success.TLabel", background=BG,
                    foreground=SUCCESS, font=FONT_UI)
        s.configure("Warn.TLabel", background=BG,
                    foreground=WARNING, font=FONT_UI)
        s.configure("Error.TLabel", background=BG,
                    foreground=ERROR_C, font=FONT_UI)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = ttk.Frame(self, style="Toolbar.TFrame", padding=(18, 14))
        hdr.pack(fill="x")

        title_block = ttk.Frame(hdr, style="Toolbar.TFrame")
        title_block.pack(side="left", fill="x", expand=True)
        ttk.Label(title_block, text="AI English Essay Corrector",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Batch PDF correction with OCR, Phi-3 analysis, and annotated ZIP export",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(hdr, text="PyMuPDF / EasyOCR / Ollama",
                  background=SURFACE_2, foreground=FG_MUTED,
                  font=FONT_SMALL).pack(side="right")
        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=18, pady=12)

        # ── folder / zip paths ────────────────────────────────────────────────
        self._build_section(main, "Input / Output")
        paths = self._surface(main)
        paths.columnconfigure(1, weight=1)

        # Input folder row
        self._field_label(paths, "Input folder", 0)
        self._folder_var = tk.StringVar()
        self._folder_var.trace_add("write", self._on_folder_changed)
        ttk.Entry(paths, textvariable=self._folder_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 8), pady=5)
        ttk.Button(paths, text="Browse...", width=11,
                    command=self._browse_folder).grid(
            row=0, column=2, pady=5)

        # PDF count label
        self._pdf_count_lbl = ttk.Label(paths, text="No folder selected",
                                        style="Muted.TLabel")
        self._pdf_count_lbl.grid(row=1, column=1, sticky="w", padx=(0, 6),
                                 pady=(0, 8))

        # Output ZIP row
        self._field_label(paths, "Output ZIP", 2)
        self._zip_var = tk.StringVar()
        ttk.Entry(paths, textvariable=self._zip_var).grid(
            row=2, column=1, sticky="ew", padx=(0, 8), pady=5)
        ttk.Button(paths, text="Browse...", width=11,
                    command=self._browse_zip).grid(
            row=2, column=2, pady=5)

        # ── settings ──────────────────────────────────────────────────────────
        self._build_section(main, "Settings")
        cfg = self._surface(main)
        cfg.columnconfigure(3, weight=1)

        ttk.Label(cfg, text="Model", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        self._model_var = tk.StringVar(value="phi3")
        ttk.Entry(cfg, textvariable=self._model_var, width=14).grid(
            row=0, column=1, sticky="w", padx=(0, 18), pady=6)

        ttk.Label(cfg, text="Ollama URL", style="Muted.TLabel").grid(
            row=0, column=2, sticky="w", padx=(0, 8), pady=6)
        self._url_var = tk.StringVar(
            value="http://localhost:11434/api/generate")
        ttk.Entry(cfg, textvariable=self._url_var, width=38).grid(
            row=0, column=3, sticky="ew", pady=6)

        self._gpu_var  = tk.BooleanVar(value=True)
        self._json_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg, text="Use GPU  (EasyOCR)",
                         variable=self._gpu_var).grid(
            row=1, column=0, columnspan=2, sticky="w",
            pady=(4, 0))
        ttk.Checkbutton(cfg,
                         text="Include intermediate JSON files in ZIP",
                         variable=self._json_var).grid(
            row=1, column=2, columnspan=2, sticky="w",
            pady=(4, 0))

# ── run button + status ───────────────────────────────────────────────
        btn_row = ttk.Frame(main, style="TFrame")
        btn_row.pack(fill="x", pady=(0, 10))
        self._run_btn = ttk.Button(btn_row, text="Process Folder",
                                     style="Accent.TButton",
                                     command=self._on_run)
        self._run_btn.pack(side="left")
        ttk.Button(btn_row, text="Live Mode", style="Secondary.TButton",
                       command=self._on_live_mode).pack(side="left", padx=(8, 0))
        self._stop_live_btn = ttk.Button(btn_row, text="Stop",
                                        style="Danger.TButton",
                                        command=self._stop_live_mode,
                                        state="disabled")
        self._stop_live_btn.pack(side="left", padx=(8, 0))
        self._status_lbl = ttk.Label(btn_row, text="", style="Success.TLabel")
        self._status_lbl.pack(side="left", padx=16)

        # ── overall progress ──────────────────────────────────────────────────
        self._build_section(main, "Progress")
        prog = self._surface(main, pady=(0, 8))

        self._overall_lbl = ttk.Label(prog, text="Waiting...",
                                      style="Muted.TLabel")
        self._overall_lbl.pack(fill="x", pady=(0, 4))
        self._overall_bar = ttk.Progressbar(
            prog, style="Accent.Horizontal.TProgressbar",
            mode="determinate", maximum=100)
        self._overall_bar.pack(fill="x", pady=(0, 8))

        self._step_lbl = ttk.Label(prog, text="", style="Muted.TLabel")
        self._step_lbl.pack(fill="x", pady=(2, 4))
        self._step_bar = ttk.Progressbar(
            prog, style="Sub.Horizontal.TProgressbar",
            mode="determinate", maximum=100)
        self._step_bar.pack(fill="x")

        # ── log ───────────────────────────────────────────────────────────────
        self._build_section(main, "Log")
        log_f = tk.Frame(main, bg=SURFACE_3, highlightthickness=1,
                         highlightbackground=BORDER)
        log_f.pack(fill="x", pady=(0, 8))
        self._log_text = tk.Text(log_f, height=7, bg=SURFACE_3, fg=FG_MUTED,
                                 font=FONT_MONO, bd=0,
                                 relief="flat", state="disabled",
                                 wrap="word", insertbackground=FG)
        lsb = ttk.Scrollbar(log_f, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self._log_text.tag_config("INFO",    foreground=FG_MUTED)
        self._log_text.tag_config("WARNING", foreground=WARNING)
        self._log_text.tag_config("ERROR",   foreground=ERROR_C)
        self._log_text.tag_config("DEBUG",   foreground=BORDER)

        # ── results table ─────────────────────────────────────────────────────
        self._build_section(main, "Results per File")
        tree_f = tk.Frame(main, bg=SURFACE_3, highlightthickness=1,
                          highlightbackground=BORDER)
        tree_f.pack(fill="both", expand=True, pady=(0, 8))

        cols = ("file", "errors", "spe", "gra", "sem", "status")
        self._tree = ttk.Treeview(tree_f, columns=cols,
                                   show="headings", selectmode="browse")
        for col, w, anc, label in [
            ("file",   300, "w",      "File"),
            ("errors",  70, "center", "Errors"),
            ("spe",     60, "center", "SPE"),
            ("gra",     60, "center", "GRA"),
            ("sem",     60, "center", "SEM"),
            ("status", 160, "w",      "Status"),
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
        self._tree.tag_configure("waiting", foreground=FG_MUTED)

        # ── bottom bar ────────────────────────────────────────────────────────
        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(0, 4))
        self._open_btn = ttk.Button(bottom, text="Open Output ZIP",
                                     style="Secondary.TButton",
                                     command=self._open_zip,
                                     state="disabled")
        self._open_btn.pack(side="left")
        self._summary_lbl = ttk.Label(bottom, text="", foreground=FG_MUTED)
        self._summary_lbl.pack(side="right")

    def _surface(self, parent, pady=(0, 10)):
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=14)
        frame.pack(fill="x", pady=pady)
        return frame

    def _field_label(self, parent, text: str, row: int):
        ttk.Label(parent, text=text, style="Muted.TLabel").grid(
            row=row, column=0, sticky="e", padx=(0, 10), pady=6)

    def _build_section(self, parent, text: str):
        row = ttk.Frame(parent, style="TFrame")
        row.pack(fill="x", pady=(6, 2))
        ttk.Label(row, text=text, foreground=ACCENT,
                  font=FONT_SECTION).pack(side="left")
        ttk.Separator(row, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=(10, 0))

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
                 "No PDF files found in this folder")
        # auto-fill ZIP path
        if not self._zip_var.get():
            self._zip_var.set(str(Path(folder).parent /
                                  (Path(folder).name + "_annotated.zip")))
        # pre-populate table
        self._tree.delete(*self._tree.get_children())
        for p in pdfs:
            self._tree.insert("", "end",
                               iid=p.name,
                               values=(p.name, "-", "-", "-", "-", "Waiting"),
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
                             values=(p.name, "-", "-", "-", "-", "Waiting"),
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
                    self._update_row(name, None, "Processing...", "active")
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

                        self._update_row(name, errors, "Done", "done")
                        all_errors += len(errors)
                        succeeded  += 1

                    except Exception as e:
                        log.error(f"  {name} failed: {e}", exc_info=False)
                        self._update_row(name, None, f"Failed: {e}", "error")
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
            summary = (f"Complete: {succeeded}/{total} files  |  "
                       f"{all_errors} total errors  |  {elapsed:.0f}s")
            self._set_overall(f"Complete - {succeeded}/{total} files OK", 100)
            self._set_status(summary, "Success.TLabel")
            self.after(0, lambda: self._summary_lbl.config(
                text=f"{succeeded} OK  {failed} failed"))
            self.after(0, lambda: self._open_btn.config(state="normal"))
            log.info(f"ZIP saved -> {zip_out}  ({succeeded} files, {elapsed:.1f}s)")

        except Exception as e:
            log.error(f"Batch failed: {e}", exc_info=True)
            self._set_status(f"Failed: {e}", "Error.TLabel")
            self._set_overall("Failed", 0)
        finally:
            self._running = False
            self.after(0, lambda: self._run_btn.config(state="normal"))


# ── entry point ───────────────────────────────────────────────────────────────
    def _on_live_mode(self):
        if self._running:
            messagebox.showwarning("Busy", "Processing in progress.")
            return
        if self._live_running:
            messagebox.showwarning("Busy", "Live mode already running.")
            return

        try:
            from camera import CameraSelector
        except ImportError as e:
            messagebox.showerror("Missing dependency",
                          f"Install opencv-python-headless:\n{e}")
            return

        devices = CameraSelector.enumerate_devices()
        if not devices:
            messagebox.showwarning("No camera", "No cameras found.")
            return

        cam_id = 0
        if len(devices) > 1:
            choices = "\n".join(f"{d['id']}: {d['name']} ({d['resolution']})"
                              for d in devices)
            cam_id = self._ask_camera(choices, devices)
            if cam_id is None:
                return

        self._run_live_mode(cam_id)

    def _ask_camera(self, choices: str, devices: list) -> int | None:
        dialog = tk.Toplevel(self)
        dialog.title("Select Camera")
        dialog.geometry("420x260")
        dialog.configure(bg=BG)
        dialog.transient(self)

        content = ttk.Frame(dialog, style="Surface.TFrame", padding=14)
        content.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(content, text="Select camera", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 8))

        var = tk.IntVar(value=0)
        for d in devices:
            ttk.Radiobutton(
                content,
                text=f"{d['name']} ({d['resolution']})",
                variable=var,
                value=d['id'],
            ).pack(anchor="w", pady=3)

        def on_ok():
            dialog.result = var.get()
            dialog.destroy()

        def on_cancel():
            dialog.result = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        btn_frame = ttk.Frame(content, style="Surface.TFrame")
        btn_frame.pack(fill="x", pady=(14, 0))
        ttk.Button(btn_frame, text="OK", style="Accent.TButton",
                   command=on_ok).pack(side="left")
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton",
                   command=on_cancel).pack(side="left", padx=(8, 0))

        dialog.result = None
        dialog.grab_set()
        self.wait_window(dialog)
        return getattr(dialog, "result", None)

    def _run_live_mode(self, camera_id: int):
        from live_camera import LiveModeController
        from nlp_engine import NLPEngine

        self._live_running = True
        self._stop_live_btn.config(state="normal")
        self._run_btn.config(state="disabled")
        self._set_status("Live mode active... press Stop to exit", "Warn.TLabel")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"Starting live mode on camera {camera_id}...\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

        def on_results(ocr_words, errors):
            n = sum(len(v) for v in ocr_words.values())
            self._log_text.configure(state="normal")
            self._log_text.insert("end",
                f"  → {n} words, {len(errors)} errors\n")
            for e in errors[:3]:
                self._log_text.insert("end",
                    f"     [{e['type']:8s}] {e['original']} → {e['suggestion']}\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")

        threading.Thread(
            target=self._run_live_thread,
            args=(camera_id, on_results),
            daemon=True
        ).start()

    def _run_live_thread(self, camera_id: int, callback):
        from live_camera import LiveModeController
        from nlp_engine import NLPEngine

        model = self._model_var.get().strip() or "phi3"
        url = self._url_var.get().strip()
        use_gpu = self._gpu_var.get()

        nlp = NLPEngine(model=model, ollama_url=url)
        controller = LiveModeController(use_gpu=use_gpu)

        try:
            controller.start(camera_id=camera_id, callback=callback, nlp_engine=nlp)
        except Exception as e:
            import traceback
            self._log_text.configure(state="normal")
            self._log_text.insert("end", f"Error: {e}\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        finally:
            self._live_running = False
            self.after(0, lambda: self._run_btn.config(state="normal"))
            self.after(0, lambda: self._stop_live_btn.config(state="disabled"))
            self._set_status("Live mode stopped", "Success.TLabel")

    def _stop_live_mode(self):
        """Stop live mode capture."""
        self._live_running = False
        self._stop_live_btn.config(state="disabled")
        self._set_status("Stopping live mode...", "Warn.TLabel")
        self._log_text.configure(state="normal")
        self._log_text.insert("end", "\nStopping live mode...\n")
        self._log_text.configure(state="disabled")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
