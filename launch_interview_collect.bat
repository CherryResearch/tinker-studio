@echo off
call "%~dp0tinker.bat" interview-collect %*
exit /b %ERRORLEVEL%
