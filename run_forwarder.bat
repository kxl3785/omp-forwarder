@echo off
rem Start the forwarder with its tray icon and no console window.
rem
rem pythonw rather than python: no console, and it keeps running after the
rem launching shell closes. Make a desktop shortcut to this file if you want
rem it one click away; point the shortcut's icon at assets\omp-forwarder.ico.
setlocal
set "LOGDIR=%LOCALAPPDATA%\omp-forwarder"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "OMP_FORWARDER_LOG=%LOGDIR%\omp-forwarder.log"
cd /d "%~dp0"
start "" /b pythonw.exe -m omp_forwarder --tray >> "%OMP_FORWARDER_LOG%" 2>&1
endlocal
