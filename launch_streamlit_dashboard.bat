@echo off
call "%~dp0tinker.bat" streamlit %*
exit /b %ERRORLEVEL%
