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
incrementally as the tokens arrive so the GUI streams the reply live.

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

# The sample prints a config/banner dump, then "Running pipeline...", then streams
# the generated tokens live to stdout (via the `text_streamer` input), and finally
# echoes the full reply on a single authoritative line:
#     Running pipeline...
#     <tokens streamed here, live, during decode>
#     Pipeline execution finished in <N> ms
#     --- Pipeline Output ---
#     Output 'generated_text': <the full reply>
# We stream the live region between "Running pipeline..." and "Pipeline execution
# finished", and fall back to the authoritative line if the live capture is empty.
_RUN_MARKER = "Running pipeline..."
_END_MARKERS = ("Pipeline execution finished", "--- Pipeline Output ---", "Total generate time:")
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
    """Yield the reply for one turn, live — token by token as the model decodes.

    The sample streams tokens to stdout via `text_streamer`. We spawn it, read
    stdout unbuffered, and once past the "Running pipeline..." marker we emit the
    accumulated generated text as each chunk arrives. The trailing
    "Output 'generated_text':" line is used only as a fallback if nothing streamed.
    """
    install_dir, config_path, sample = resolve(install, config)
    args = [str(sample), str(config_path)]
    if image:
        args.append(f"image={image}")
    if audio:
        args.append(f"audio={audio}")
    args.append(f"prompt={prompt}")
    args.append("text_streamer")  # wire the live token stream (config routes it to the LLM module)

    proc = subprocess.Popen(
        args, cwd=str(REPO), env=build_env(install_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
    )

    buf = bytearray()          # raw bytes seen so far (whole stdout)
    started = False            # passed the "Running pipeline..." marker?
    reply = ""                 # live-streamed text accumulated after the marker
    emit_base = 0              # index in `buf` where the live region begins

    def _decode(b: bytes) -> str:
        return b.decode("utf-8", errors="replace")

    try:
        while True:
            chunk = proc.stdout.read(64)
            if not chunk:
                break
            buf.extend(chunk)

            if not started:
                text = _decode(bytes(buf))
                idx = text.find(_RUN_MARKER)
                if idx != -1:
                    started = True
                    # live region begins right after the marker line's newline
                    nl = text.find("\n", idx)
                    emit_base = len(text[: nl + 1].encode("utf-8")) if nl != -1 else len(buf)
                continue

            # We're in the live region — decode everything after emit_base and,
            # if an end marker has appeared, trim the streamed reply at it.
            region = _decode(bytes(buf[emit_base:]))
            cut = len(region)
            for m in _END_MARKERS:
                p = region.find(m)
                if p != -1:
                    cut = min(cut, p)
            candidate = region[:cut]
            if candidate != reply:
                reply = candidate
                yield reply.strip()
    finally:
        rest = proc.stdout.read()
        if rest:
            buf.extend(rest)
        proc.wait()

    full = _decode(bytes(buf))

    # Authoritative fallback: if streaming surfaced nothing (or looks empty),
    # parse the final "Output 'generated_text':" line.
    if not reply.strip():
        m = _RESULT_RE.search(full)
        if m:
            authoritative = m.group(1)
            nxt = re.search(r"\nOutput '", authoritative)
            if nxt:
                authoritative = authoritative[: nxt.start()]
            authoritative = authoritative.strip()
            if authoritative:
                yield authoritative
                return
        low = full.lower()
        if "error" in low or "exception" in low:
            snippet = full.strip().splitlines()[-1] if full.strip() else "(no output)"
            yield f"(pipeline error: {snippet})"
        else:
            yield "(no output — check the GPU driver and that Stage 1 exported the IR)"


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
    import time
    err = check_ready()
    if err:
        print("[runner] NOT READY:\n" + err)
        sys.exit(1)
    p = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    img = sys.argv[2] if len(sys.argv) > 2 else None
    aud = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"[runner] prompt={p!r} image={img} audio={aud}\n--- streaming (live) ---")
    t0 = time.time()
    n = 0
    out = ""
    for chunk in stream_generate(p, img, aud):
        n += 1
        new = chunk[len(out):] if chunk.startswith(out) else chunk
        out = chunk
        if n <= 3 or n % 20 == 0:
            print(f"[+{time.time()-t0:5.1f}s] update #{n}: …{new[-30:]!r}")
    print(f"--- done: {n} live updates in {time.time()-t0:.1f}s ---")
    print(out)
