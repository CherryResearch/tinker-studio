@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\tinker_env\Scripts\python.exe"

if not exist "%PYTHON%" (
  set "PYTHON=python"
)

"%PYTHON%" "%ROOT%\tinker_launcher.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Tinker launcher exited with code %EXIT_CODE%.
  pause
)

exit /b %EXIT_CODE%
