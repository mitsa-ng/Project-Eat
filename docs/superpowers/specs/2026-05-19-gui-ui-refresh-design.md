# GUI UI Refresh Design

## Goal

Improve the Tkinter batch-processing UI so Project-EAT feels more like a polished desktop tool while preserving the existing workflow and implementation stack.

This change keeps OCR, NLP, PDF annotation, ZIP packaging, and live camera behavior intact. The implementation is scoped to `gui.py`.

## Approach

Use a conservative Tkinter refresh:

- Keep the single-window layout.
- Keep all existing controls and actions.
- Improve hierarchy, spacing, colors, and status presentation.
- Avoid new dependencies.
- Avoid changing the processing pipeline or file formats.

## UI Structure

The window remains organized around the current workflow:

1. Header with product name and concise capability subtitle.
2. Input / Output section for PDF folder and ZIP destination.
3. Settings section for model, Ollama URL, GPU, and JSON output.
4. Action bar with `Process Folder` as the primary action and Live Mode controls as secondary actions.
5. Progress section with clearer overall and current-step labels.
6. Log section for diagnostic output.
7. Results table for per-file status and error counts.
8. Bottom summary and output ZIP action.

## Visual Treatment

- Retain a dark desktop-tool theme.
- Reduce decorative visual noise and improve readability.
- Use consistent surfaces for sections, fields, logs, and tables.
- Use stronger contrast for primary actions and status states.
- Make the camera selection dialog match the main window theme.

## Behavior

The UI refresh must preserve:

- Folder selection and automatic ZIP path population.
- PDF count detection.
- Batch processing thread behavior.
- Progress updates.
- Log streaming.
- Result row updates.
- Output ZIP opening.
- Live Mode launch and stop controls.
- Camera selection behavior.

## Error Handling

Existing warning and error dialogs remain. Failed file rows should still show the error message and use the error visual state.

## Verification

There are no automated tests in this repository. Verification will use:

- `python3 -m py_compile gui.py`
- A GUI smoke check that starts and closes the Tkinter app without running OCR, NLP, or Live Mode.

Manual full processing remains outside this UI refresh and requires PDFs in `pdfs/`, a running Ollama server, and an installed Phi-3 model.
