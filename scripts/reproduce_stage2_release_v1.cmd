@echo off
setlocal
pushd "%~dp0.."
python -m internal_time_rl.analysis.release_stage2 reproduce %*
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
