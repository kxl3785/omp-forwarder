@echo off
rem The GPU 1 lane: same launcher as run_forwarder.bat, no tray, --gpu 1,
rem listening on 8891 with the GPU 0 lane (8890) as its peer.
setlocal
set "LOGDIR=%LOCALAPPDATA%\omp-forwarder"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "OMP_FORWARDER_LOG=%LOGDIR%\omp-forwarder-gpu1.log"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
cd /d "%~dp0"
start "" /b pythonw.exe -m omp_forwarder --port 8891 --gpu 1 --name "GPU 1" --peer 8890 --upstream-exe .unsloth >> "%OMP_FORWARDER_LOG%" 2>&1
endlocal
