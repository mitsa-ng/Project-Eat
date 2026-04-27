# Project-EAT

## Overview

Project-EAT appears to be a Python application involving OCR, NLP, and a GUI interface. This README provides instructions for setting up the development environment and running the application.

## Prerequisites

- Ollama https://ollama.com/
- phi3 model
  ```bash
   ollama pull phi3
   ```
  ```bash
   ollama pull phi3:14b
   ```
- Python 3.8 or later
- `pip` for installing dependencies
- (Optional) A virtual environment tool such as `venv` or `virtualenv`

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd Project-EAT
   ```

2. **Create a virtual environment** (optional):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (cmd.exe)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Initialization

1. Ensure the `pdfs/` directory exists and contains any PDF files you intend to process. The app may generate additional output directories as needed.

2. Run the application:
   ```bash
   python main.py
   ```
   or
   ```bash
   python gui.py
   ```

### Live Camera Mode

Project-EAT supports live camera mode for real-time essay document analysis:

#### Requirements

```bash
pip install opencv-python-headless
```

#### CLI Usage

```bash
python main.py --live --camera 0
```

- `--live`: Enable live camera mode
- `--camera N`: Select camera device (default: 0)
- Use `--no-gpu` to force CPU processing

Press `Ctrl+C` to stop live mode.

#### GUI Usage

Click the "📷 Live Mode" button in the GUI. Select your camera when prompted, then point at your essay document. The analysis results will stream in real-time.

### Hardware Permissions

- **macOS**: Grant camera access in System Preferences → Security & Privacy → Privacy → Camera
- **Windows**: Allow camera access when prompted by the application
- **Linux**: Add user to the `video` group: `sudo usermod -a -G video $USER`
   
## Additional Notes

- If you add new dependencies, update `requirements.txt` by running:
  ```bash
  pip freeze > requirements.txt
  ```

- For development, consider installing a linter or formatter like `flake8` or `black`.

- Refer to individual source files (`gui.py`, `nlp_engine.py`, `ocr_engine.py`, etc.) for more details on functionality.

---

Feel free to expand this README with usage examples, tests, or contribution guidelines in the future.
