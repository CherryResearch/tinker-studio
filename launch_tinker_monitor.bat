@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\tinker_env\Scripts\python.exe"

if not defined TINKER_API_KEY (
  for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','User'); if (-not $v) { $v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','Machine') }; if ($v) { [Console]::Write($v) }"`) do set "TINKER_API_KEY=%%A"
)

if not defined TINKER_API_KEY (
  echo Could not find TINKER_API_KEY in the current, User, or Machine environment.
  pause
  exit /b 1
)

cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\monitor_tinker_runs.py" --recent 6 --refresh 15 %*

endlocal
