#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper that drives the Gemma4-12B OpenVINO pipeline for the chatbot demo.

The prebuilt package ships only the C++ `yaml_pipeline_sample.exe` (there is no
`pipeline`/`openvino_genai` Python binding), so each chat turn spawns one
process: it builds the pipeline from the YAML config, runs once, streams tokens
to stdout via the `text_streamer` input, and exits. This module spawns that
process, puts the package's runtime DLLs on PATH, and yields generated text
incrementally so the GUI can stream the reply.

Because each turn is an independent process, this is a single-turn demo (no
conversation memory) — which matches what the sample actually supports.
"""
import os
import re
import subprocess
from pathlib import Path

# repo root = three levels up from this file (samples/chatbot/pipeline_runner.py)
REPO = Path(__file__).resolve().parents[2]
DEFAULT_INSTALL = REPO / "install"
DEFAULT_CONFIG = REPO / "samples" / "stage1_safetensors" / "config" / "config_modeling_text_img_audio_cb_st.yaml"
MODEL_DIR = REPO / "models" / "gemma-4-12B-it"

# The sample prints a config/banner dump, then the authoritative result on a
# single line after this banner:
#     --- Pipeline Output ---
#     Output 'generated_text': <the full reply>
# When run as a child process its `text_streamer` tokens do NOT show up in the
# captured stdout pipe (they go to a console handle), so we parse that final
# line as the source of truth and chunk it ourselves for a streaming feel.
_OUTPUT_BANNER = "--- Pipeline Output ---"
_RESULT_RE = re.compile(r"Output 'generated_text':\s*(.*)", re.S)


def resolve(install: str | None = None, config: str | None = None):
    install_dir = Path(install).resolve() if install else DEFAULT_INSTALL
    config_path = Path(config).resolve() if config else DEFAULT_CONFIG
    sample = install_dir / "samples" / "cpp" / "yaml_pipeline_sample.exe"
    return install_dir, config_path, sample


def check_ready(install: str | None = None, config: str | None = None) -> str | None:
    """Return an error string if the deployment isn't ready, else None."""
    install_dir, config_path, sample = resolve(install, config)
    if not sample.exists():
        return (f"yaml_pipeline_sample.exe not found at {sample}.\n"
                "Unzip the prebuilt package into the repo's install\\ (see main README §3).")
    if not config_path.exists():
        return f"config not found: {config_path}"
    if not (MODEL_DIR / "openvino_language_model.xml").exists():
        return (f"Exported IR not found under {MODEL_DIR}.\n"
                "Run Stage 1 first (samples\\stage1_safetensors\\run.bat) so the INT4 IR is built.")
    return None


def build_env(install_dir: Path) -> dict:
    env = os.environ.copy()
    env["DUMP_PERFORMANCE"] = "0"
    dll_dirs = [
        install_dir / "runtime" / "bin" / "intel64" / "Release",
        install_dir / "runtime" / "3rdparty" / "tbb" / "bin",
    ]
    env["PATH"] = os.pathsep.join(str(d) for d in dll_dirs) + os.pathsep + env.get("PATH", "")
    return env


def stream_generate(prompt: str, image: str | None = None, audio: str | None = None,
                    install: str | None = None, config: str | None = None):
    """Yield the reply for one turn, chunked word-by-word for a streaming feel.

    We run the sample to completion (it prints the full reply on the
    `Output 'generated_text':` line), then re-emit that text incrementally so the
    GUI shows it filling in. The pipeline runs once and exits — there is no live
    token stream available over the captured pipe.
    """
    install_dir, config_path, sample = resolve(install, config)
    args = [str(sample), str(config_path)]
    if image:
        args.append(f"image={image}")
    if audio:
        args.append(f"audio={audio}")
    args.append(f"prompt={prompt}")

    proc = subprocess.run(
        args, cwd=str(REPO), env=build_env(install_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    log = proc.stdout or ""

    # Extract the authoritative reply: everything after "Output 'generated_text':"
    # up to the next "Output '...':" line (other ports) or end of log.
    reply = ""
    if _OUTPUT_BANNER in log:
        tail = log.split(_OUTPUT_BANNER, 1)[1]
        m = _RESULT_RE.search(tail)
        if m:
            reply = m.group(1)
            # stop at the next "Output 'xxx':" port line, if any
            nxt = re.search(r"\nOutput '", reply)
            if nxt:
                reply = reply[:nxt.start()]
            reply = reply.strip()

    if not reply:
        # Surface a useful error rather than a blank bubble.
        if "error" in log.lower() or "exception" in log.lower():
            snippet = log.strip().splitlines()[-1] if log.strip() else "(no output)"
            yield f"(pipeline error: {snippet})"
        else:
            yield "(no output — check the GPU driver and that Stage 1 exported the IR)"
        return

    # Re-emit incrementally (word chunks) so the UI streams the answer in.
    words = reply.split(" ")
    acc = ""
    for i, w in enumerate(words):
        acc = w if i == 0 else acc + " " + w
        yield acc


def generate(prompt: str, image: str | None = None, audio: str | None = None,
             install: str | None = None, config: str | None = None) -> str:
    """Non-streaming convenience: return the final text (used by headless tests)."""
    last = ""
    for chunk in stream_generate(prompt, image, audio, install, config):
        last = chunk
    return last


if __name__ == "__main__":
    # Headless smoke test: python pipeline_runner.py "Hello" [image] [audio]
    import sys
    err = check_ready()
    if err:
        print("[runner] NOT READY:\n" + err)
        sys.exit(1)
    p = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    img = sys.argv[2] if len(sys.argv) > 2 else None
    aud = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"[runner] prompt={p!r} image={img} audio={aud}\n--- streaming ---")
    out = ""
    for chunk in stream_generate(p, img, aud):
        out = chunk
    print(out)
