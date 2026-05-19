# GUI UI Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the Project-EAT Tkinter GUI into a clearer, more polished desktop tool without changing the OCR, NLP, annotation, ZIP, or Live Mode behavior.

**Architecture:** Keep the current single-window `App` class in `gui.py`. Add small UI helper methods for repeated section and field construction, update the existing styles and layout, and preserve all current callbacks and background-thread behavior.

**Tech Stack:** Python, Tkinter, ttk, existing Project-EAT modules.

---

## File Structure

- Modify: `gui.py`
  - Owns the full desktop UI, logging handler, batch-processing callbacks, progress updates, results table, and Live Mode controls.
  - This plan changes only presentation helpers, layout construction, and camera dialog styling.

No new runtime dependencies are allowed.

---

### Task 1: Refresh Theme Constants And Styles

**Files:**
- Modify: `gui.py`

- [ ] **Step 1: Update color and layout constants**

Replace the existing palette constants near the top of `gui.py` with a more structured dark desktop palette:

```python
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
```

Keep `TYPE_COLORS` but update it to reference the new readable status colors:

```python
TYPE_COLORS = {
    "spelling": ERROR_C,
    "grammar": WARNING,
    "semantic": "#8ab4ff",
}
```

- [ ] **Step 2: Update `_setup_styles`**

Update ttk styles so frames, labels, buttons, entries, progress bars, and the tree view use the new constants. Include styles named:

```python
"Surface.TFrame"
"Toolbar.TFrame"
"Muted.TLabel"
"Title.TLabel"
"Subtitle.TLabel"
"Accent.TButton"
"Secondary.TButton"
"Danger.TButton"
"Accent.Horizontal.TProgressbar"
"Sub.Horizontal.TProgressbar"
"Success.TLabel"
"Warn.TLabel"
"Error.TLabel"
```

Expected behavior:

- Primary button remains visually dominant.
- Secondary buttons are quieter.
- Tree rows are taller and easier to scan.
- Progress bars have better contrast.

- [ ] **Step 3: Verify syntax**

Run:

```bash
python3 -m py_compile gui.py
```

Expected: command exits successfully.

---

### Task 2: Rebuild Main Layout With Clearer Sections

**Files:**
- Modify: `gui.py`

- [ ] **Step 1: Add layout helper methods**

Add helper methods inside `App`:

```python
def _surface(self, parent, pady=(0, 10)):
    frame = ttk.Frame(parent, style="Surface.TFrame", padding=14)
    frame.pack(fill="x", pady=pady)
    return frame

def _field_label(self, parent, text: str, row: int):
    ttk.Label(parent, text=text, style="Muted.TLabel").grid(
        row=row, column=0, sticky="e", padx=(0, 10), pady=6)

def _build_section(self, parent, text: str):
    row = ttk.Frame(parent, style="TFrame")
    row.pack(fill="x", pady=(8, 4))
    ttk.Label(row, text=text, foreground=ACCENT, font=FONT_SECTION).pack(side="left")
    ttk.Separator(row, orient="horizontal").pack(
        side="left", fill="x", expand=True, padx=(10, 0))
```

If `_build_section` already exists, replace its body rather than creating a duplicate.

- [ ] **Step 2: Update header**

In `_build_ui`, replace the current header with a darker toolbar using:

```python
hdr = ttk.Frame(self, style="Toolbar.TFrame", padding=(18, 14))
hdr.pack(fill="x")

title_block = ttk.Frame(hdr, style="Toolbar.TFrame")
title_block.pack(side="left", fill="x", expand=True)
ttk.Label(title_block, text="AI English Essay Corrector", style="Title.TLabel").pack(anchor="w")
ttk.Label(
    title_block,
    text="Batch PDF correction with OCR, Phi-3 analysis, and annotated ZIP export",
    style="Subtitle.TLabel",
).pack(anchor="w", pady=(3, 0))

ttk.Label(hdr, text="PyMuPDF / EasyOCR / Ollama", style="Muted.TLabel").pack(side="right")
tk.Frame(self, bg=ACCENT, height=2).pack(fill="x")
```

- [ ] **Step 3: Update path and settings surfaces**

Use `_surface(main)` for Input / Output and Settings sections. Keep the same variables and callbacks:

```python
self._folder_var
self._folder_var.trace_add("write", self._on_folder_changed)
self._zip_var
self._model_var
self._url_var
self._gpu_var
self._json_var
```

Expected behavior:

- Browse buttons still call `_browse_folder` and `_browse_zip`.
- PDF count label still updates in `_on_folder_changed`.
- ZIP path still auto-populates.

- [ ] **Step 4: Verify syntax**

Run:

```bash
python3 -m py_compile gui.py
```

Expected: command exits successfully.

---

### Task 3: Improve Action, Progress, Log, And Results Areas

**Files:**
- Modify: `gui.py`

- [ ] **Step 1: Restyle action bar**

Keep the same buttons and commands:

```python
self._run_btn = ttk.Button(..., command=self._on_run, style="Accent.TButton")
ttk.Button(..., command=self._on_live_mode, style="Secondary.TButton")
self._stop_live_btn = ttk.Button(..., command=self._stop_live_mode, style="Danger.TButton", state="disabled")
```

Use button text without emoji dependency:

```python
"Process Folder"
"Live Mode"
"Stop"
```

- [ ] **Step 2: Restyle progress panel**

Keep `self._overall_lbl`, `self._overall_bar`, `self._step_lbl`, and `self._step_bar`. Place them inside a surface frame, with labels using `Muted.TLabel` and progress bars using the existing style names.

Expected behavior:

- `_set_overall` and `_set_step` continue working unchanged.
- Initial overall text remains `Waiting...`.

- [ ] **Step 3: Restyle log panel**

Keep `self._log_text` and its tags. Use:

```python
height=7
bg=SURFACE_3
fg=FG_MUTED
font=FONT_MONO
wrap="word"
```

Expected behavior:

- `_append_log`, `_run_live_mode`, `_run_live_thread`, and `_stop_live_mode` still write to the log.

- [ ] **Step 4: Restyle results table and bottom bar**

Keep the same tree columns and tags. Adjust widths to:

```python
("file", 300)
("errors", 70)
("spe", 60)
("gra", 60)
("sem", 60)
("status", 160)
```

Keep `self._open_btn` and `self._summary_lbl`.

- [ ] **Step 5: Verify syntax**

Run:

```bash
python3 -m py_compile gui.py
```

Expected: command exits successfully.

---

### Task 4: Theme Camera Dialog And Run Final Smoke Checks

**Files:**
- Modify: `gui.py`

- [ ] **Step 1: Theme `_ask_camera` dialog**

Update `_ask_camera` so the dialog uses the same dark palette:

```python
dialog.configure(bg=BG)
content = ttk.Frame(dialog, style="Surface.TFrame", padding=14)
content.pack(fill="both", expand=True, padx=12, pady=12)
```

Use `ttk.Label`, `ttk.Radiobutton`, and `ttk.Button` where possible. Keep the same return behavior:

- OK returns selected camera id.
- Cancel returns `None`.
- Closing without selecting returns `None`.

- [ ] **Step 2: Run syntax verification**

Run:

```bash
python3 -m py_compile gui.py
```

Expected: command exits successfully.

- [ ] **Step 3: Run import-level smoke check**

Run:

```bash
python3 - <<'PY'
import gui
print(gui.App.__name__)
PY
```

Expected output:

```text
App
```

- [ ] **Step 4: Review local diff**

Run:

```bash
git diff -- gui.py
```

Expected:

- Only UI presentation, layout, and camera dialog styling changed.
- No changes to OCR, NLP, annotation, ZIP packaging, or threading control flow.

