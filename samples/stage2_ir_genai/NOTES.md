# Stage 2 — deploying the exported IR with OpenVINO GenAI

After Stage 1, the model directory contains engine-flavored OpenVINO IR:

```
openvino_language_model.xml/bin          # LLM, INT4 — inputs: inputs_embeds, bidirectional_mask, attention_mask, position_ids, beam_idx
openvino_text_embeddings_model.xml/bin   # input_ids -> text embeds
openvino_vision_embeddings_model.xml/bin
openvino_audio_embeddings_model.xml/bin
openvino_tokenizer.xml/bin, openvino_detokenizer.xml/bin
```

There are two ways the **openvino.genai** stack can consume this IR. They differ
in how the language model receives its input, which is the key thing to understand.

## 1. Recommended & verified: redeploy through the pipeline engine

`run_ir.bat` re-runs `yaml_pipeline_sample.exe` against the same YAML. Because the
`openvino_*.xml` already exist on disk, the engine **loads the IR directly** instead of
rebuilding it from safetensors — this is "deploy from IR". The language model runs through
the OpenVINO GenAI continuous-batching engine (`ov::genai`) wrapped by the pipeline.

This is the path the README uses and the one verified end-to-end on GPU.

## 2. GenAI-compatible facade (C++) — `genai_facade_llm.cpp`

`ov::pipeline::LLMPipeline` is a drop-in replacement for `ov::genai::LLMPipeline`. When the
model directory contains a `pipeline.yaml`, the facade routes the request through the
pipeline engine (consuming the engine-flavored IR). This sample shows that API and how to
link against the package's `openvino_genai` library.

Caveat: the bundled Gemma4 YAML is multi-modal (it declares image/audio input ports). The
`yaml_pipeline_sample.exe` driver supplies empty optional image/audio inputs, but a bare
`LLMPipeline.generate(prompt)` does not, so driving the multi-modal graph through the plain
text facade reports `Can't find input data: audio`. To use the facade for text-only, supply
a text-only `pipeline.yaml` (no image/audio modules). The sample is provided as a build/API
reference; the verified runtime path is option 1.

## 3. genai-native `ov::genai::VLMPipeline` — `genai_vlm_deploy.cpp` (reference only)

This is the upstream genai-native VLM API. It does **not** consume the IR that
openvino.pipeline exports, because the two use different language-model input contracts:

| | pipeline export (this repo) | genai-native VLMPipeline expects |
|---|---|---|
| Language model input | `inputs_embeds` + `bidirectional_mask` | `inputs_embeds` + `token_type_ids` |
| Vision `pixel_values` packing | engine layout | optimum-intel layout |
| `config.json` `model_type` | `gemma4_unified` | `gemma4` |

Pointing `ov::genai::VLMPipeline` at this IR fails at inference with errors like
`Port for tensor name token_type_ids was not found` and a `pixel_values` shape mismatch.
To use genai-native VLMPipeline you need an **optimum-intel** export of the model
(`optimum-cli export openvino ...`), which is a different artifact than what this pipeline
produces. The sample is kept here to document the difference.

## Building the C++ samples

```
samples\stage2_ir_genai\build.bat
```

If the package is missing some upstream genai dev headers (e.g. `chat_history.hpp`),
supplement from an OpenVINO GenAI source checkout:

```
set GENAI_SRC_INCLUDE=D:\path\to\openvino.genai\src\cpp\include
samples\stage2_ir_genai\build.bat
```
