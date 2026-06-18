#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""A small Gradio GUI for a multimodal Gemma4-12B chatbot on OpenVINO.

Type a message, optionally attach an image and/or an audio clip, and the model
(deployed from the exported INT4 IR, running on the Intel GPU) replies — with the
answer streamed token by token.

Single-turn by design: each message spawns one `yaml_pipeline_sample.exe` run
(the only interface the prebuilt package exposes), so the model has no memory of
previous turns. The chat history you see is for your reference only.

Run via `run_chatbot.bat`, or:  python app.py
"""
import threading
from pathlib import Path

import gradio as gr

import pipeline_runner as pr

STATUS_OK = (
    "**Model:** google/gemma-4-12B-it (INT4_ASYM)  •  "
    "**Device:** GPU  •  **Mode:** single-turn (no memory)\n\n"
    "A resident pipeline process compiles the model for the GPU **once at startup** "
    "(~10 s), then stays loaded — so after the first load each reply starts streaming "
    "in about a second. The first image question also pays a one-time vision-encoder "
    "compile."
)


def respond(message, image, audio, history):
    """Stream a reply for one turn. history is a list of {'role','content'} dicts."""
    message = (message or "").strip()
    if not message and not image and not audio:
        yield history, ""
        return

    # Build the user-side bubble (text + note of attachments).
    attach = []
    if image:
        attach.append("🖼️ image")
    if audio:
        attach.append("🔊 audio")
    user_text = message if message else "(describe the attached input)"
    if attach:
        user_text += f"   _[{', '.join(attach)}]_"

    history = (history or []) + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "▌"},
    ]
    yield history, ""

    err = pr.check_ready()
    if err:
        history[-1]["content"] = f"⚠️ Deployment not ready:\n\n{err}"
        yield history, ""
        return

    # First turn pays the one-time model compile; show a clear status while it loads.
    if not pr.is_warm():
        history[-1]["content"] = "⏳ Loading the model and compiling for the GPU (one-time, ~10 s)…"
        yield history, ""
        pr.warmup()
        history[-1]["content"] = "▌"
        yield history, ""

    prompt = message if message else "Describe the attached input in detail."
    try:
        for partial in pr.stream_generate(prompt, image=image, audio=audio):
            history[-1]["content"] = partial + " ▌"
            yield history, ""
        history[-1]["content"] = history[-1]["content"].rstrip(" ▌")
        if not history[-1]["content"]:
            history[-1]["content"] = "(no output — check the GPU driver and that Stage 1 exported the IR)"
        yield history, ""
    except Exception as e:  # surface runtime failures in the chat instead of crashing the UI
        history[-1]["content"] = f"⚠️ Error while generating: {e}"
        yield history, ""


def build_demo():
    with gr.Blocks(title="Gemma4-12B Multimodal Chatbot (OpenVINO)") as demo:
        gr.Markdown("# 🤖 Gemma4-12B Multimodal Chatbot — OpenVINO on Intel GPU")
        gr.Markdown(STATUS_OK)

        # gradio 5.x needs type="messages" for the {'role','content'} format;
        # gradio 6.x made that the default and removed the kwarg. Support both.
        try:
            chat = gr.Chatbot(type="messages", height=460, label="Conversation")
        except TypeError:
            chat = gr.Chatbot(height=460, label="Conversation")
        with gr.Row():
            image_in = gr.Image(type="filepath", label="Image (optional)", height=160)
            audio_in = gr.Audio(type="filepath", label="Audio (optional)")
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask something… (attach an image/audio above to ask about it)",
                scale=8, show_label=False, autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear", scale=1)

        # Bundled sample assets (resolved absolutely so they work regardless of CWD).
        _here = Path(__file__).resolve().parent
        _img = str(_here.parent / "GoldenGate.png")      # samples/GoldenGate.png
        _aud = str(_here.parent / "journal1.wav")        # samples/journal1.wav
        gr.Examples(
            examples=[
                ["How do black holes work?", None, None],
                ["What is shown in this image?", _img, None],
                ["What kind of sound is in this audio?", None, _aud],
                ["Describe the image and the audio together.", _img, _aud],
            ],
            inputs=[msg, image_in, audio_in],
            label="Examples (text / image / audio / image+audio)",
        )

        # Submitting clears the attachments after sending (single-turn).
        def _clear_inputs():
            return None, None

        send.click(respond, [msg, image_in, audio_in, chat], [chat, msg]) \
            .then(_clear_inputs, None, [image_in, audio_in])
        msg.submit(respond, [msg, image_in, audio_in, chat], [chat, msg]) \
            .then(_clear_inputs, None, [image_in, audio_in])
        clear.click(lambda: ([], None, None), None, [chat, image_in, audio_in])

    return demo


def _prewarm():
    """Start compiling the model as soon as the demo launches (in the background),
    so the GPU compile is done — or well underway — before the first question."""
    if pr.check_ready():
        return  # not deployed yet; respond() will surface the error
    try:
        print("[chatbot] pre-warming pipeline (compiling model for the GPU)…")
        pr.warmup()
        print("[chatbot] pipeline ready.")
    except Exception as e:
        print(f"[chatbot] pre-warm skipped: {e}")


if __name__ == "__main__":
    # Kick off the one-time compile at startup so it overlaps with the browser opening.
    threading.Thread(target=_prewarm, daemon=True).start()
    build_demo().queue().launch(server_name="127.0.0.1", inbrowser=True)
