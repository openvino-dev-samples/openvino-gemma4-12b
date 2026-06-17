@echo off
REM Copyright (C) 2026 Intel Corporation
REM SPDX-License-Identifier: Apache-2.0
REM
REM One-time environment setup for the Gemma4-12B OpenVINO Windows sample.
REM Run from the repository root:  scripts\setup_env.bat
REM
REM Creates a Python venv and installs the packages needed for model download
REM and tokenizer conversion. The prebuilt runtime package (install\) must be
REM unzipped separately - see README.md section "Get the prebuilt package".

setlocal

REM Optional corporate proxy (uncomment / edit if needed):
REM set "HTTPS_PROXY=http://your-proxy:port"
REM set "HTTP_PROXY=http://your-proxy:port"

if not exist python-env (
    echo [setup] creating venv python-env ...
    python -m venv python-env
)

call python-env\Scripts\activate.bat
python -m pip install --upgrade pip

REM openvino-tokenizers MUST match the OpenVINO version inside install\
REM (this package is OpenVINO 2026.2). transformers is used to read tokenizer.json.
python -m pip install "transformers" "openvino-tokenizers==2026.2.*" "huggingface_hub"

echo.
echo [setup] done. Next:
echo   1) unzip the prebuilt package into .\install   (see README)
echo   2) python samples\common\download_model.py
echo   3) samples\stage1_safetensors\run.bat
endlocal
