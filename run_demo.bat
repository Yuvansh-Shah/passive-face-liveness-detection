@echo off
rem ============================================================
rem  Khoji Liveness Detection - one-click demo launcher
rem
rem  Double-click this file. It will:
rem    1. Find Python 3.10-3.13 on this PC, or silently install
rem       Python 3.12 for the current user (no admin needed)
rem    2. Create an isolated Python environment under
rem       %LocalAppData%\KhojiLivenessDemo (kept out of this
rem       folder so it never gets copied/synced by accident)
rem    3. Download the required packages (one-time, ~300 MB)
rem    4. Start the demo server and open it in your browser
rem
rem  To remove everything later: delete the folder
rem  %LocalAppData%\KhojiLivenessDemo and, if Python was installed
rem  by this script, uninstall "Python 3.12" from Installed apps.
rem ============================================================
setlocal
title Khoji Liveness Demo
cd /d "%~dp0"

set "APPDIR=%LocalAppData%\KhojiLivenessDemo"
set "VENV=%APPDIR%\venv"
set "VPY=%VENV%\Scripts\python.exe"
set "REQS=%~dp0demo_requirements.txt"
set "PYEXE="

echo ============================================================
echo    Khoji Liveness Detection - demo setup
echo ============================================================
echo(

rem --- sanity: are the demo files next to this script? ---------
if not exist "%REQS%" (
    echo [ERROR] demo_requirements.txt is missing from this folder.
    echo         If you are running this from inside a ZIP file,
    echo         extract the WHOLE folder first, then run it again.
    goto :fail
)
if not exist "%~dp0liveness\demo.py" (
    echo [ERROR] The liveness folder is missing next to this script.
    echo         If you are running this from inside a ZIP file,
    echo         extract the WHOLE folder first, then run it again.
    goto :fail
)
if not exist "%~dp0liveness\checkpoints\liveness_mnv3.pt" (
    echo [ERROR] Model file liveness\checkpoints\liveness_mnv3.pt is missing.
    goto :fail
)

rem --- step 1: find or install Python --------------------------
echo [1/4] Looking for Python 3.10-3.13 ...
call :probe py -3.12
if not defined PYEXE call :probe py -3.11
if not defined PYEXE call :probe py -3.13
if not defined PYEXE call :probe py -3.10
if not defined PYEXE call :probe py -3
if not defined PYEXE call :probe python
if not defined PYEXE (
    echo        Python was not found - installing Python 3.12 for the
    echo        current user only ^(~25 MB download, no admin needed^).
    call :install_python
)
if not defined PYEXE (
    echo [ERROR] Could not find or install Python automatically.
    echo         Please install Python 3.12 from
    echo         https://www.python.org/downloads/ and run this file again.
    goto :fail
)
echo        Using: %PYEXE%

rem --- step 2: create the isolated environment -----------------
echo [2/4] Preparing the Python environment ...
if exist "%VPY%" (
    "%VPY%" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo        Existing environment is broken - recreating it.
        rmdir /s /q "%APPDIR%"
    )
)
if not exist "%VPY%" "%PYEXE%" -m venv "%VENV%"
if not exist "%VPY%" (
    echo [ERROR] Could not create the Python environment.
    goto :fail
)

rem --- step 3: install packages (first run only) ---------------
if exist "%VENV%\.deps_ok" (
    echo [3/4] Packages already installed - skipping download.
) else (
    echo [3/4] Downloading packages: PyTorch, OpenCV, FastAPI, ...
    echo        One-time download of roughly 300 MB - this can take a few
    echo        minutes depending on your internet connection.
    echo(
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r "%REQS%"
    if errorlevel 1 (
        echo(
        echo [ERROR] Package installation failed - see messages above.
        echo         Check your internet connection and run this file again.
        goto :fail
    )
    "%VPY%" -c "import torch, torchvision, cv2, timm, fastapi, uvicorn, PIL"
    if errorlevel 1 (
        echo [ERROR] Packages installed but could not be loaded.
        goto :fail
    )
    echo ok> "%VENV%\.deps_ok"
)

rem --- step 4: run the demo ------------------------------------
echo [4/4] Starting the demo server ...
echo(
echo        The demo opens in your browser at http://127.0.0.1:8000
echo        KEEP THIS WINDOW OPEN while using the demo.
echo        Close this window ^(or press Ctrl+C^) to stop it.
echo(
"%VPY%" "%~dp0liveness\demo.py"
echo(
echo The demo server has stopped. You can close this window.
pause
exit /b 0

rem --- helpers --------------------------------------------------

:probe
rem Usage: call :probe <command ...>
rem Sets PYEXE if the command is a working Python 3.10-3.13.
rem The "if exist" check rejects the Microsoft Store stub, which
rem prints an ad instead of a real interpreter path.
for /f "delims=" %%P in ('%* -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13); print(sys.executable)" 2^>nul') do (
    if exist "%%P" set "PYEXE=%%P"
)
exit /b 0

:install_python
set "PYVER=3.12.10"
set "PYURL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe"
set "PYINST=%TEMP%\khoji-python-%PYVER%-amd64.exe"
where curl >nul 2>nul
if not errorlevel 1 (
    curl -L --fail -o "%PYINST%" "%PYURL%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYURL%' -OutFile '%PYINST%'"
)
if not exist "%PYINST%" (
    echo [ERROR] Could not download the Python installer.
    exit /b 1
)
echo        Installing Python %PYVER% ^(current user only^) ...
start /wait "" "%PYINST%" /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 Include_launcher=0 Include_test=0 PrependPath=1 AssociateFiles=0
del "%PYINST%" >nul 2>nul
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)
exit /b 0

:fail
echo(
echo Setup did not finish. Fix the issue above and double-click this
echo file again - steps that already completed will be skipped.
pause
exit /b 1
