@echo off
REM Copyright (C) 2026 Intel Corporation
REM SPDX-License-Identifier: Apache-2.0
REM
REM Launch the multimodal Gemma4-12B chatbot GUI.
REM Prerequisites (see the repo's main README): the prebuilt package is unzipped
REM into install\, and Stage 1 has exported the INT4 IR into models\gemma-4-12B-it\.
REM
REM Run from anywhere; this resolves the repo root from its own location.

setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
popd

REM Put the package's OpenVINO + GenAI DLLs on PATH for the spawned sample exe.
call "%REPO_ROOT%\install\setupvars.bat"

REM Activate the venv created by scripts\setup_env.bat (transformers / ov-tokenizers),
REM then make sure gradio is available for the GUI.
if exist "%REPO_ROOT%\python-env\Scripts\activate.bat" call "%REPO_ROOT%\python-env\Scripts\activate.bat"
python -c "import gradio" 2>nul || python -m pip install -r "%SCRIPT_DIR%requirements.txt"

python "%SCRIPT_DIR%app.py"
endlocal
