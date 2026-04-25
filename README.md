# Gradio RL Model Comparison Demo

## File structure

- `app.py` - Main Gradio application
- `requirements.txt` - Python dependencies

## Run on Windows (PowerShell)

1. Create virtual environment:

   ```powershell
   py -m venv .venv
   ```

2. Activate virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Run app:

   ```powershell
   python app.py
   ```

5. Open URL shown in terminal (usually http://127.0.0.1:7860).

## Configure APIs

Edit `app.py` and replace:

- `BASE_MODEL_API`
- `TRAINED_MODEL_API`

with your actual API endpoints.
