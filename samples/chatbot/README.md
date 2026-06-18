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
binding), so each chat turn **spawns one pipeline process**: it builds the
pipeline from `config_modeling_text_img_audio_cb_st.yaml`, runs once with
`image=` / `audio=` / `prompt=` inputs, streams tokens to stdout via the
`text_streamer` input, and exits. `pipeline_runner.py` wraps that — it puts the
package DLLs on `PATH`, runs from the repo root so the config's relative
`model_path` resolves, and yields text incrementally; `app.py` is the Gradio
front end.

```
text + (image?) + (audio?)
        │
        ▼
 pipeline_runner.stream_generate()
        │  spawns
        ▼
 install\samples\cpp\yaml_pipeline_sample.exe   (deploys the exported INT4 IR on GPU)
        │  streams tokens
        ▼
   Gradio chat bubble
```

## Notes & limitations

- **Single-turn / no memory.** Each message is an independent process, matching
  what the sample supports — the chat history shown is for your reference and is
  not fed back into the model. (Multi-turn would require a resident server, which
  the prebuilt package doesn't expose.)
- **First-token latency includes startup.** Every turn reloads the IR and
  compiles for the GPU, so expect a few seconds before text appears for a
  text-only turn, and longer (~10 s) for image turns. This is a demo, not a
  production service. A natural future optimization is a GPU `cache_dir` to reuse
  the compiled blobs, or a long-lived server process.
- **Audio** input uses a `.wav` file. Two sample assets are bundled in `samples/`
  (`GoldenGate.png`, `journal1.wav`) and preloaded as examples, including an
  **image + audio together** example — you can attach an image *and* an audio clip
  in the same turn and ask about both at once (verified working).
  All three modalities were verified working (text, image, audio). Audio replies
  can be **prompt-sensitive**: an explicit instruction like *"Transcribe the
  following speech segment in its original language."* or *"What kind of sound is
  in this audio?"* reliably engages the audio branch, whereas some terse phrasings
  may make the model claim it received no audio even though the clip was processed.
- If a reply is empty or errors, see the main README's Troubleshooting section
  (GPU driver, missing IR, tokenizer conversion).
