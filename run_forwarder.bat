@echo off
rem Start the forwarder with its tray icon and no console window.
rem
rem pythonw rather than python: no console, and it keeps running after the
rem launching shell closes. Make a desktop shortcut to this file if you want
rem it one click away; point the shortcut's icon at assets\omp-forwarder.ico.
rem
rem Works straight from a clone -- src\ goes on PYTHONPATH so no install is
rem needed. If you did `pip install .`, use `omp-forwarder --tray` instead.
setlocal
set "LOGDIR=%LOCALAPPDATA%\omp-forwarder"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "OMP_FORWARDER_LOG=%LOGDIR%\omp-forwarder.log"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
cd /d "%~dp0"
rem --upstream-exe: this box also runs a llama-server for another project.
rem Without the filter, discovery can pick that one and its model answers
rem silently. Drop the flag if Studio's is the only llama-server you run.
start "" /b pythonw.exe -m omp_forwarder --tray --upstream-exe .unsloth >> "%OMP_FORWARDER_LOG%" 2>&1
endlocal
