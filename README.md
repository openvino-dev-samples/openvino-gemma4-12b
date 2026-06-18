<!-- Copyright (C) 2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma4-12B on OpenVINO — Windows Deployment Sample

Run **Google Gemma4-12B-it** locally on an Intel GPU on Windows with OpenVINO, using a
**prebuilt distribution package** — no source compilation of the runtime required.

This repo demonstrates two deployment stages:

1. **Stage 1 — Safetensors deployment** (`openvino.pipeline`): load the Hugging Face
   safetensors model, build INT4 OpenVINO IR on the fly, and generate text.
2. **Stage 2 — IR deployment** (`openvino.genai`): redeploy the IR that Stage 1 exported
   through the OpenVINO GenAI engine — no rebuild from safetensors.

> The prebuilt Windows package (OpenVINO runtime + GenAI + the pipeline engine, ~150 MB) is
> published as a **GitHub Release asset**, not committed to the repository. Models are **not**
> included — download them from Hugging Face (instructions below).

---

## 1. Architecture

The package is a single unified install tree built from three repositories:

| Layer | Repo | Role |
|-------|------|------|
| Core runtime | `openvino` | OpenVINO inference engine (`openvino.dll`) |
| GenAI | `openvino.genai` | LLM/VLM pipelines, modeling, tokenizers (`openvino_genai.dll`) |
| Pipeline engine | `openvino.pipeline` | YAML-driven modular pipeline + `yaml_pipeline_sample.exe` |

Two ways to deploy the model:

```
                 Hugging Face safetensors (google/gemma-4-12B-it)
                                  |
        Stage 1: yaml_pipeline_sample.exe + config_..._cb_st.yaml
                                  |
                 builds + serializes INT4 OpenVINO IR in-place:
       openvino_language_model.xml, openvino_text_embeddings_model.xml,
       openvino_vision_embeddings_model.xml, openvino_audio_embeddings_model.xml
                                  |
        Stage 2: same sample re-run -> loads IR from disk (no rebuild)
                 == openvino.genai engine deploying the IR
```

> **Note on the exported IR contract.** The language model exported here takes
> `inputs_embeds` + `bidirectional_mask` (the pipeline engine's embedding-fusion split). It is
> consumed by the **pipeline engine / GenAI-compatible facade**, *not* by genai-native
> `ov::genai::VLMPipeline` (which expects an optimum-intel export with `token_type_ids`). See
> [`samples/stage2_ir_genai/NOTES.md`](samples/stage2_ir_genai/NOTES.md) for the full explanation.

---

## 2. Prerequisites

- **Windows 11** (x64).
- **Intel GPU** (Arc / Core Ultra iGPU) + recent Intel GPU driver. The sample configs target
  `device: GPU`. (CPU works too — edit the YAML `device:` fields.)
- **~40 GB free disk**: ~24 GB safetensors + ~7 GB exported IR + package.
- **Python 3.10+** — for model download and one-time tokenizer conversion.
- **(Optional, Stage-2 C++ samples only)** Visual Studio 2022 Build Tools, CMake ≥ 3.23, Ninja.
- **(Optional)** A corporate HTTP proxy. Where shown, set:
  ```cmd
  set HTTPS_PROXY=http://your-proxy:port
  set HTTP_PROXY=http://your-proxy:port
  ```

All commands below are run from the **repository root** in a `cmd` prompt.

---

## 3. Get the prebuilt package

Download the package zip from this repo's
[**Releases** page](https://github.com/openvino-dev-samples/openvino-gemma4-12b/releases/tag/v2026.06)
and unzip it into `install\` at the repository root:

```cmd
:: download the package (~150 MB) from the v2026.06 Release, e.g.:
curl -L -o pipeline_windows_x86_64.zip ^
  https://github.com/openvino-dev-samples/openvino-gemma4-12b/releases/download/v2026.06/pipeline_windows_20260617_x86_64.zip

powershell -Command "Expand-Archive -Path pipeline_windows_x86_64.zip -DestinationPath install -Force"

:: sanity check
dir install\setupvars.bat
dir install\samples\cpp\yaml_pipeline_sample.exe
```

`install\setupvars.bat` puts the OpenVINO + GenAI DLLs on `PATH`; the sample scripts call it for you.

---

## 4. Environment setup (Python)

```cmd
scripts\setup_env.bat
```

This creates `python-env\` and installs `transformers`, `huggingface_hub`, and
`openvino-tokenizers==2026.2.*` (pinned to the package's OpenVINO version). Behind a proxy, edit
the proxy lines at the top of the script first.

---

## 5. Download the model

```cmd
call python-env\Scripts\activate.bat
python samples\common\download_model.py google/gemma-4-12B-it models\gemma-4-12B-it
```

Result: `models\gemma-4-12B-it\` with `model.safetensors` (~24 GB), `config.json`,
`tokenizer.json`, etc. Behind a proxy, `set HTTPS_PROXY=...` first.

Then convert the tokenizer to OpenVINO IR (one time):

```cmd
python samples\stage1_safetensors\prepare_tokenizer.py models\gemma-4-12B-it
```

This writes `openvino_tokenizer.xml` / `openvino_detokenizer.xml` into the model dir.

---

## 6. Stage 1 — Safetensors deployment (build IR + generate)

```cmd
samples\stage1_safetensors\run.bat "How do black holes work?"
```

The **first** run loads the safetensors, quantizes to INT4, **serializes the IR in-place**, then
generates. Expect (abridged):

```
[Safetensors Copy] Loaded 677 tensors ...
[SafetensorsWeightFinalizer] In-flight quantization enabled (INT4_ASYM) ...
Pipeline constructed successfully.
Pipeline execution finished in ~? ms
Output 'generated_text': To understand how black holes work, ...
```

Confirm the IR was written:

```cmd
dir models\gemma-4-12B-it\openvino_language_model.xml
dir models\gemma-4-12B-it\openvino_text_embeddings_model.xml
dir models\gemma-4-12B-it\openvino_vision_embeddings_model.xml
```

---

## 7. Stage 2 — IR deployment (redeploy from IR via GenAI engine)

Re-run the pipeline. Because the `openvino_*.xml` now exist, the engine **loads them directly**
(no safetensors rebuild / re-quantization) — this is "deploy from IR":

```cmd
samples\stage2_ir_genai\run_ir.bat "Explain how OpenVINO accelerates inference."
```

You should *not* see the `[Safetensors Copy]` / `In-flight quantization` lines this time — just
`Pipeline constructed successfully` and the generated answer, much faster to first token.

### Optional — Stage-2 C++ reference samples

Two C++ samples show the OpenVINO GenAI APIs and how to link against the package:

```cmd
:: needs VS2022 Build Tools + CMake + Ninja
samples\stage2_ir_genai\build.bat
```

Produces `genai_facade_llm.exe` (`ov::pipeline::LLMPipeline`, the GenAI-compatible facade) and
`genai_vlm_deploy.exe` (`ov::genai::VLMPipeline`, genai-native reference). Read
[`samples/stage2_ir_genai/NOTES.md`](samples/stage2_ir_genai/NOTES.md) first — it explains which
sample consumes this IR and the genai-native contract difference.

> **Note (package-only users):** this prebuilt package ships the GenAI-compatible *shadow* headers,
> which `#include` a few upstream GenAI headers (e.g. `openvino/genai/chat_history.hpp`) that the
> package does not bundle. So the plain `build.bat` will fail with
> `fatal error C1083: ... openvino/genai/chat_history.hpp` **unless** you point it at an OpenVINO
> GenAI source checkout. These C++ samples are optional reference code — the verified runtime path
> (Stages 1, 2 and the multimodal demo) needs **no** compilation. To build them anyway:

```cmd
set GENAI_SRC_INCLUDE=D:\path\to\openvino.genai\src\cpp\include
samples\stage2_ir_genai\build.bat
```

---

## 8. Multimodal (image + text) — also a genai IR deployment

The bundled config is multimodal: the vision, text and language IR were all exported in Stage 1.
Passing an image just adds the vision branch — once the `openvino_*.xml` exist, this is the **same
"deploy from IR" path as Stage 2** (the engine loads the vision/text/language IR from disk; no
rebuild from safetensors), now consuming the image input too.

```cmd
call install\setupvars.bat
install\samples\cpp\yaml_pipeline_sample.exe ^
    samples\stage1_safetensors\config\config_modeling_text_img_audio_cb_st.yaml ^
    "image=samples\GoldenGate.png" "prompt=What is shown in this image?"
```

(A sample image `samples\GoldenGate.png` is included in the repo; swap in any `.png`/`.jpg` of your own.)

Expected (abridged — verified with the Golden Gate sample image, deployed from IR on GPU):

```
Pipeline constructed successfully.
Pipeline execution finished in ~13000 ms
Output 'generated_text': The image shows a close-up view of a large, complex metal or steel
structure, likely part of a bridge ... set against a background of water ...
```

(Audio works the same way with `audio=path\to\clip.wav`, using the exported
`openvino_audio_embeddings_model.xml`.)

> This is the multimodal counterpart of Stage 2: the GenAI / pipeline engine deploys the exported
> IR — it does **not** re-read the safetensors. See
> [`samples/stage2_ir_genai/NOTES.md`](samples/stage2_ir_genai/NOTES.md).

---

## 9. Multimodal chatbot demo (GUI)

A small **Gradio** web UI that wraps everything above: type a message, optionally attach an image
and/or a `.wav`, and get a streamed reply — all from the exported INT4 IR on the Intel GPU.

```cmd
samples\chatbot\run_chatbot.bat
```

This opens http://127.0.0.1:7860 in your browser. Under the hood each turn spawns one
`yaml_pipeline_sample.exe` run (the package's only interface), so it is **single-turn** (no
conversation memory) and the first token of each reply includes IR load + GPU compile. All three
input modalities are verified working (text, image, audio). See
[`samples/chatbot/README.md`](samples/chatbot/README.md) for details and notes.

---

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `xxx.dll not found` / app exits immediately | Run `call install\setupvars.bat` in the same shell (the `.bat` scripts do this for you). |
| `Failed to convert tokenizer` / `openvino_tokenizer.xml not found` | Run `prepare_tokenizer.py` (step 5); ensure the venv has `transformers` + `openvino-tokenizers`. |
| transformers treats the model path as a Hub repo id | Pass a real filesystem path (the scripts use an absolute path). |
| GPU not used / `Failed to create plugin ... GPU` | Update the Intel GPU driver, or switch `device:` to `CPU` in the YAML configs. |
| Download is slow / fails | Set `HTTPS_PROXY` / `HTTP_PROXY` before running `download_model.py`. |
| `Port for tensor name token_type_ids was not found` | You pointed genai-native `VLMPipeline` at this IR — that path is unsupported; use Stage 2 (`run_ir.bat`). See NOTES.md. |
| genai-native VLM `model_type` not recognized | The exported `config.json` uses `gemma4_unified` (engine type), not `gemma4`. This is expected for the engine path. |

---

## Repository layout

```
openvino-gemma4-12b/
  README.md
  scripts/setup_env.bat                       # venv + pip deps
  samples/
    common/download_model.py                  # HF snapshot download (proxy-aware)
    stage1_safetensors/
      run.bat                                  # safetensors -> IR + generate
      prepare_tokenizer.py                     # tokenizer -> OpenVINO IR (one time)
      config/                                  # Gemma4 pipeline YAML configs
    stage2_ir_genai/
      run_ir.bat                               # redeploy from IR (engine/GenAI)
      genai_facade_llm.cpp                     # ov::pipeline::LLMPipeline (facade)
      genai_vlm_deploy.cpp                     # ov::genai::VLMPipeline (reference; see NOTES)
      CMakeLists.txt, build.bat, NOTES.md
    chatbot/                                   # multimodal chatbot demo (Gradio GUI)
      run_chatbot.bat, app.py, pipeline_runner.py, requirements.txt, README.md
  install/      (you unzip the Release package here — gitignored)
  models/       (you download the model here — gitignored)
```

## License

[Apache License 2.0](LICENSE). The model `google/gemma-4-12B-it` is subject to Google's Gemma
license terms — review them before use.
