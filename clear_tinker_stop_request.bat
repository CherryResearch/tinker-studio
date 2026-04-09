@echo off
setlocal

for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"
set "PYTHON_EXE=%SCRIPT_DIR%\tinker_env\Scripts\python.exe"
set "CLI_SCRIPT=%SCRIPT_DIR%\tinker_stop_cli.py"

if not exist "%PYTHON_EXE%" (
  echo Could not find:
  echo %PYTHON_EXE%
  pause
  exit /b 1
)

if not exist "%CLI_SCRIPT%" (
  echo Could not find:
  echo %CLI_SCRIPT%
  pause
  exit /b 1
)

pushd "%SCRIPT_DIR%"
"%PYTHON_EXE%" "%CLI_SCRIPT%" --workspace "%SCRIPT_DIR%" --action clear
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if "%EXIT_CODE%"=="0" (
  echo Stop request cleared.
) else (
  echo Failed to clear the stop request.
)
pause
exit /b %EXIT_CODE%
