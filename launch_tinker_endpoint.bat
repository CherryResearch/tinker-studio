@echo off
call "%~dp0tinker.bat" endpoint %*
exit /b %ERRORLEVEL%
