@echo off
setlocal

for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"
set "PYTHON_EXE=%SCRIPT_DIR%\tinker_env\Scripts\python.exe"
set "NOTEBOOK_PATH=%SCRIPT_DIR%\tinker_train_and_eval.ipynb"

if not exist "%PYTHON_EXE%" (
  echo Could not find the virtual environment interpreter:
  echo %PYTHON_EXE%
  echo.
  echo Make sure tinker_env exists in this folder before launching.
  pause
  exit /b 1
)

if not defined TINKER_API_KEY (
  for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','User'); if (-not $v) { $v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','Machine') }; if ($v) { [Console]::Write($v) }"`) do set "TINKER_API_KEY=%%A"
)

if not defined TINKER_API_KEY (
  echo Could not find TINKER_API_KEY in the current, User, or Machine environment.
  echo.
  echo Set TINKER_API_KEY and relaunch this file.
  pause
  exit /b 1
)

pushd "%SCRIPT_DIR%"

echo Launching Jupyter Lab from:
echo %SCRIPT_DIR%
echo.

"%PYTHON_EXE%" -m jupyter lab "%NOTEBOOK_PATH%" --ServerApp.root_dir="%SCRIPT_DIR%" %*

set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Jupyter exited with code %EXIT_CODE%.
  pause
)

exit /b %EXIT_CODE%
