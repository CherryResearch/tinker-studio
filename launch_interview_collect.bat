@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\tinker_env\Scripts\python.exe"

cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\collect_interview_qa.py" %*

endlocal
