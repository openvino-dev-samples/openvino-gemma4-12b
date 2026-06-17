@echo off
REM Copyright (C) 2026 Intel Corporation
REM SPDX-License-Identifier: Apache-2.0
REM
REM Build the Stage-2 reference C++ samples (genai_facade_llm, genai_vlm_deploy)
REM against the prebuilt package. Requires VS2022 Build Tools + CMake + Ninja.
REM
REM Run from this directory:  samples\stage2_ir_genai\build.bat
REM
REM Optional: set GENAI_SRC_INCLUDE to an OpenVINO GenAI source include dir if
REM the package is missing some upstream genai headers (see CMakeLists.txt note):
REM   set GENAI_SRC_INCLUDE=D:\path\to\openvino.genai\src\cpp\include

setlocal

REM Adjust the VS edition if needed (Community / Professional / Enterprise / BuildTools).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
popd

call "%REPO_ROOT%\install\setupvars.bat"

set "CMAKE_EXTRA="
if defined GENAI_SRC_INCLUDE set "CMAKE_EXTRA=-DGENAI_SRC_INCLUDE=%GENAI_SRC_INCLUDE%"

cmake -S "%SCRIPT_DIR%." -B "%SCRIPT_DIR%build" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DOV_INSTALL_DIR=%REPO_ROOT%\install %CMAKE_EXTRA%
if errorlevel 1 exit /b 1
cmake --build "%SCRIPT_DIR%build" -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1

echo.
echo Build complete:
echo   %SCRIPT_DIR%build\genai_facade_llm.exe
echo   %SCRIPT_DIR%build\genai_vlm_deploy.exe
endlocal
