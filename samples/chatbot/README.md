<!-- Copyright (C) 2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Multimodal chatbot demo (GUI)

A small **Gradio** web UI that chats with **Gemma4-12B-it** on OpenVINO: type a
message, optionally attach an **image** and/or an **audio** clip, and get a reply
streamed back token by token — all running locally on your Intel GPU from the
exported INT4 IR.

![flow](../GoldenGate.png)

## Prerequisites

This demo *uses* an existing deployment — it does not create one. First complete
the main [README](../../README.md) through **Stage 1** so that:

- the prebuilt package is unzipped into the repo's `install\`, and
- `models\gemma-4-12B-it\` contains the exported IR
  (`openvino_language_model.xml`, `openvino_text_embeddings_model.xml`,
  `openvino_vision_embeddings_model.xml`, `openvino_tokenizer.xml`, …).

The launcher checks these and prints a clear hint if anything is missing.

## Run

From anywhere (the script resolves the repo root itself):

```cmd
samples\chatbot\run_chatbot.bat
```

It sources `install\setupvars.bat`, activates the `python-env` venv created by
`scripts\setup_env.bat`, installs `gradio` if needed, and opens the UI in your
browser (http://127.0.0.1:7860). Type a question, optionally attach an image or a
`.wav`, and press **Send**.

To run the UI module directly instead:

```cmd
call install\setupvars.bat
python samples\chatbot\app.py
```

## How it works

The prebuilt package exposes the model through the C++
`yaml_pipeline_sample.exe` (there is no `pipeline`/`openvino_genai` Python
binding). The exe runs in **resident server mode** (`--serve`): it builds the
pipeline from `config_modeling_text_img_audio_cb_st.yaml` **once** — paying the
GPU compile at startup — then reads one JSON request per line from stdin and
streams the reply to stdout, keeping the compiled model resident.
`pipeline_runner.py` starts one such process (puts the package DLLs on `PATH`,
runs from the repo root so the config's relative `model_path` resolves) and
reuses it for every question, yielding text incrementally; `app.py` is the
Gradio front end and pre-warms the process when the demo launches.

```
        demo launches → pipeline_runner.warmup()
                              │ spawns once (one-time GPU compile ~10 s)
                              ▼
        install\samples\cpp\yaml_pipeline_sample.exe --serve   (INT4 IR resident on GPU)
                              ▲ │
   text + (image?) + (audio?)│ │ streams tokens
   per turn (JSON on stdin)  │ ▼
                         Gradio chat bubble   (each reply starts in ~1 s)
```

## Notes & limitations

- **Single-turn / no memory.** Each message is fed to the model independently —
  the chat history shown is for your reference and is not fed back into the
  model. (The pipeline stays resident for speed, but the demo does not thread
  prior turns into the prompt.)
- **First load compiles the model; later turns are fast.** The model is compiled
  for the GPU **once** when the demo starts (~10 s) and then stays loaded, so
  after that each reply starts streaming in about a second. The first *image*
  question also pays a one-time vision-encoder compile (~10 s); subsequent image
  questions are fast. To make the one-time startup compile faster on later
  launches too, set `OV_PIPELINE_CACHE_DIR` to a writable folder before starting
  the demo — OpenVINO then persists the compiled GPU blob and reuses it.
- **Audio** input uses a `.wav` file. Two sample assets are bundled in `samples/`
  (`GoldenGate.png`, `journal1.wav`) and preloaded as examples, including an
  **image + audio together** example — you can attach an image *and* an audio clip
  in the same turn and ask about both at once (verified working).
  All three modalities were verified working (text, image, audio). The loader
  reads the first channel and **resamples to 16 kHz** (the rate the audio
  frontend expects), so WAV files at any sample rate (e.g. 44.1/48 kHz) are
  handled correctly.
- If a reply is empty or errors, see the main README's Troubleshooting section
  (GPU driver, missing IR, tokenizer conversion).
