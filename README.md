# Project-EAT

## Overview

Project-EAT appears to be a Python application involving OCR, NLP, and a GUI interface. This README provides instructions for setting up the development environment and running the application.

## Prerequisites

- Ollama https://ollama.com/
- Python 3.8 or later
- `pip` for installing dependencies
- (Optional) A virtual environment tool such as `venv` or `virtualenv`

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd Project-EAT
   ```

2. **Create a virtual environment** (strongly recommended):
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
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
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

3. Use the GUI or command-line prompts to interact with the OCR and NLP features.

## Additional Notes

- If you add new dependencies, update `requirements.txt` by running:
  ```bash
  pip freeze > requirements.txt
  ```

- For development, consider installing a linter or formatter like `flake8` or `black`.

- Refer to individual source files (`gui.py`, `nlp_engine.py`, `ocr_engine.py`, etc.) for more details on functionality.

---

Feel free to expand this README with usage examples, tests, or contribution guidelines in the future.