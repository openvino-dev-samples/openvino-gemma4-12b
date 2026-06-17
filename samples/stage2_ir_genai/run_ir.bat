@echo off
REM Copyright (C) 2026 Intel Corporation
REM SPDX-License-Identifier: Apache-2.0
REM
REM Stage 2 - redeploy the exported OpenVINO IR with openvino.genai.
REM
REM After Stage 1 has written openvino_*.xml into the model directory, this
REM re-runs the same pipeline. Because the IR already exists on disk, the
REM engine loads it directly (no safetensors rebuild / re-quantization) - i.e.
REM "deploy from IR". The LLM runs through the GenAI continuous-batching engine
REM (ov::genai) wrapped by the pipeline.
REM
REM Run from the REPOSITORY ROOT:
REM   samples\stage2_ir_genai\run_ir.bat ["your prompt"]

setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
popd

call "%REPO_ROOT%\install\setupvars.bat"
if exist "%REPO_ROOT%\python-env\Scripts\activate.bat" call "%REPO_ROOT%\python-env\Scripts\activate.bat"

set "SAMPLE=%REPO_ROOT%\install\samples\cpp\yaml_pipeline_sample.exe"
REM Reuse the same config; on a second run all openvino_*.xml already exist so
REM the engine deploys from IR instead of rebuilding from safetensors.
set "CONFIG=%REPO_ROOT%\samples\stage1_safetensors\config\config_modeling_text_img_audio_cb_st.yaml"
set "MODEL_DIR=%REPO_ROOT%\models\gemma-4-12B-it"
set "PROMPT=%~1"
if "%PROMPT%"=="" set "PROMPT=Explain how OpenVINO accelerates inference."

if not exist "%MODEL_DIR%\openvino_language_model.xml" (
    echo [error] No exported IR found in %MODEL_DIR%.
    echo         Run Stage 1 first: samples\stage1_safetensors\run.bat
    exit /b 1
)

pushd "%REPO_ROOT%"
echo [stage2] deploying from IR in %MODEL_DIR%
echo [stage2] prompt = %PROMPT%
"%SAMPLE%" "%CONFIG%" "prompt=%PROMPT%"
set RC=%ERRORLEVEL%
popd

exit /b %RC%
endlocal
