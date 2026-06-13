@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\tinker_env\Scripts\python.exe"

if not exist "%PYTHON%" (
  set "PYTHON=python"
)

cd /d "%ROOT%"
"%PYTHON%" -m streamlit run "%ROOT%\streamlit_tinker_dashboard.py" %*

endlocal
