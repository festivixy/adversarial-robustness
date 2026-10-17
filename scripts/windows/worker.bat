@echo off
cd /d "%~dp0..\.."
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run robustness-gap worker %*
pause
