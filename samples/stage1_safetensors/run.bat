@echo off
REM Copyright (C) 2026 Intel Corporation
REM SPDX-License-Identifier: Apache-2.0
REM
REM Stage 1 - deploy Gemma4-12B from safetensors with openvino.pipeline.mx.
REM
REM The first run builds INT4 OpenVINO IR in-place inside the model directory
REM (openvino_language_model.xml, openvino_text_embeddings_model.xml,
REM  openvino_vision_embeddings_model.xml, openvino_audio_embeddings_model.xml)
REM and then generates text. Subsequent runs reuse that IR (Stage 2).
REM
REM Run from the REPOSITORY ROOT:
REM   samples\stage1_safetensors\run.bat ["your prompt"]

setlocal

REM --- resolve repo root (two levels up from this script) ---
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
popd

call "%REPO_ROOT%\install\setupvars.bat"
if exist "%REPO_ROOT%\python-env\Scripts\activate.bat" call "%REPO_ROOT%\python-env\Scripts\activate.bat"

set "SAMPLE=%REPO_ROOT%\install\samples\cpp\yaml_pipeline_sample.exe"
set "CONFIG=%SCRIPT_DIR%config\config_modeling_text_img_audio_cb_st.yaml"
set "PROMPT=%~1"
if "%PROMPT%"=="" set "PROMPT=How do black holes work?"

if not exist "%SAMPLE%" (
    echo [error] %SAMPLE% not found. Unzip the prebuilt package into install\ first.
    exit /b 1
)

REM The config's model_path is "models/gemma-4-12B-it/" relative to the repo
REM root, so run from there.
pushd "%REPO_ROOT%"
echo [stage1] config = %CONFIG%
echo [stage1] prompt = %PROMPT%
"%SAMPLE%" "%CONFIG%" "prompt=%PROMPT%"
set RC=%ERRORLEVEL%
popd

exit /b %RC%
endlocal
