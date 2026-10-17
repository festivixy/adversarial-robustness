@echo off
setlocal
cd /d "%~dp0..\.."
where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv, the Python project manager ...
    powershell -NoProfile -ExecutionPolicy Bypass ^
        -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
echo Installing Python and the project's packages. The first time takes a few minutes ...
uv sync --frozen
if errorlevel 1 (
    echo.
    echo Setup failed. Check the internet connection and run this script again.
    pause
    exit /b 1
)
if "%~1"=="" (
    uv run robustness-gap lab --setup
) else (
    uv run robustness-gap lab --coordinator "%~1"
)
