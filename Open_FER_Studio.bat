@echo off
setlocal
pushd "%~dp0"
set "CONDA_EXE="
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\mambaforge\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\mambaforge\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
if not defined CONDA_EXE for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if defined CONDA_EXE (
    "%CONDA_EXE%" run --no-capture-output -n facial_emotion_ai python "%~dp0main.py"
) else (
    echo Anaconda/Conda was not found. Install Anaconda, then run this file again.
    pause
)
if errorlevel 1 pause
popd
endlocal
