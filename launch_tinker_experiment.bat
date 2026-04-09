@echo off
setlocal EnableDelayedExpansion

for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"
set "PYTHON_EXE=%SCRIPT_DIR%\tinker_env\Scripts\python.exe"
set "RUN_SCRIPT=%SCRIPT_DIR%\run_tinker_experiment.py"
set "RUN_NAME="
set "HAS_RUN_NAME=0"

if "%~1"=="" (
  set "RUN_NAME=essay_recent_r16"
  set "HAS_RUN_NAME=1"
) else (
  set "FIRST_ARG=%~1"
  if "!FIRST_ARG:~0,2!"=="--" (
    set "HAS_RUN_NAME=0"
  ) else (
    set "RUN_NAME=%~1"
    set "HAS_RUN_NAME=1"
    shift
  )
)

if not exist "%PYTHON_EXE%" (
  echo Could not find the virtual environment interpreter:
  echo %PYTHON_EXE%
  pause
  exit /b 1
)

if not exist "%RUN_SCRIPT%" (
  echo Could not find the run script:
  echo %RUN_SCRIPT%
  pause
  exit /b 1
)

if not defined TINKER_API_KEY (
  for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','User'); if (-not $v) { $v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','Machine') }; if ($v) { [Console]::Write($v) }"`) do set "TINKER_API_KEY=%%A"
)

if not defined TINKER_API_KEY (
  echo Could not find TINKER_API_KEY in the current, User, or Machine environment.
  pause
  exit /b 1
)

pushd "%SCRIPT_DIR%"
if "%HAS_RUN_NAME%"=="1" (
  "%PYTHON_EXE%" "%RUN_SCRIPT%" --workspace "%SCRIPT_DIR%" --run-name "%RUN_NAME%" %*
) else (
  "%PYTHON_EXE%" "%RUN_SCRIPT%" --workspace "%SCRIPT_DIR%" %*
)
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Run exited with code %EXIT_CODE%.
  pause
)

exit /b %EXIT_CODE%
