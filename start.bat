@echo off
REM One-click launcher for the TrialMatch-LLM browser app (Windows).
REM First run: creates .venv, installs CPU PyTorch + requirements, downloads models (~3.5 GB).
setlocal
cd /d "%~dp0"
if "%TM_ADAPTER%"=="" set TM_ADAPTER=runs/qwen15_lora/best_adapter
if not exist runs\qwen15_lora\best_adapter if exist adapter.txt set /p TM_ADAPTER=<adapter.txt
if not exist .venv (
  echo [setup] creating virtual environment...
  python -m venv .venv || goto :fail
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
  pip install -r requirements.txt || goto :fail
) else (
  call .venv\Scripts\activate.bat
)
echo [run] adapter = %TM_ADAPTER%
echo [run] opening http://127.0.0.1:7860 when models are loaded...
python app\app.py
goto :eof
:fail
echo Setup failed. See README.md "Troubleshooting".
pause
