@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\tinker_env\Scripts\python.exe"

if not exist "%PYTHON%" (
  set "PYTHON=python"
)

if not defined TINKER_API_KEY (
  for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','User'); if (-not $v) { $v=[Environment]::GetEnvironmentVariable('TINKER_API_KEY','Machine') }; if ($v) { [Console]::Write($v) }"`) do set "TINKER_API_KEY=%%A"
)

cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\serve_tinker_endpoint.py" %*

endlocal
