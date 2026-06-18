#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Drive the Gemma4-12B OpenVINO pipeline for the chatbot demo.

The prebuilt package ships the C++ `yaml_pipeline_sample.exe`, which now has a
resident **server mode** (`--serve`): it builds the pipeline once — paying the
one-time GPU compile at startup — then reads one JSON request per line from
stdin and streams the reply to stdout, keeping the compiled model resident.

This module starts one such server process (lazily, on first use) and reuses it
for every question, so only the *first* turn waits for the model to load; after
that, each reply starts streaming in about a second. The previous design spawned
a fresh process per question, which recompiled the model for the GPU every time
(~10 s before the first token). A lock serializes turns over the single process.
"""
import json
import os
import subprocess
import threading
from pathlib import Path

# repo root = three levels up from this file (samples/chatbot/pipeline_runner.py)
REPO = Path(__file__).resolve().parents[2]
DEFAULT_INSTALL = REPO / "install"
DEFAULT_CONFIG = REPO / "samples" / "stage1_safetensors" / "config" / "config_modeling_text_img_audio_cb_st.yaml"
MODEL_DIR = REPO / "models" / "gemma-4-12B-it"

# Sentinels printed by the exe's --serve mode (keep in sync with the C++ source).
_READY = "<<<__PIPELINE_READY__>>>"
_END = "<<<__END_OF_RESPONSE__>>>"

# When an image/audio is attached, the exe prints a couple of media-load log lines
# ("Loading image: ...", "Shape: [...]", "Loading audio: ...", "Samples: ...") to
# stdout before the generated tokens. Drop those so they don't show in the bubble.
import re as _re
_MEDIA_LOG_RE = _re.compile(
    r"^\s*(Loading (image|audio|video):.*|Shape:.*|Samples:.*|Stacked shape:.*)$",
    _re.MULTILINE,
)


def _strip_media_logs(text: str) -> str:
    return _MEDIA_LOG_RE.sub("", text)


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


class PipelineServer:
    """A resident `yaml_pipeline_sample.exe --serve` process, reused across turns."""

    def __init__(self, install: str | None = None, config: str | None = None):
        self.install_dir, self.config_path, self.sample = resolve(install, config)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def is_started(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self):
        """Spawn the server and block until the model is compiled and resident."""
        if self.is_started():
            return
        args = [str(self.sample), str(self.config_path), "--serve"]
        self._proc = subprocess.Popen(
            args, cwd=str(REPO), env=build_env(self.install_dir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        # Wait for the READY sentinel (one-time GPU compile happens here).
        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                rc = self._proc.poll()
                raise RuntimeError(f"pipeline server exited during startup (code {rc})")
            if _READY in raw.decode("utf-8", "replace"):
                break

    def _read_response(self):
        """Yield the growing reply for the current turn until the END sentinel.

        Tokens stream to stdout without newlines, so we read small byte chunks
        (not lines) to surface text live. We keep a small tail buffer so the END
        sentinel is detected even if it straddles two reads, and decode a fresh
        copy of the whole accumulated byte buffer each time to avoid splitting a
        multi-byte UTF-8 character.
        """
        buf = bytearray()
        while True:
            raw = self._proc.stdout.read(32)
            if not raw:
                text = bytes(buf).decode("utf-8", "replace")
                text = text.split(_END, 1)[0]
                if text.strip():
                    yield text
                return
            buf.extend(raw)
            text = bytes(buf).decode("utf-8", "replace")
            if _END in text:
                yield text.split(_END, 1)[0]
                return
            yield text

    def generate(self, prompt: str, image: str | None = None, audio: str | None = None):
        """Stream the reply for one turn. Serialized via a lock (single process)."""
        with self._lock:
            if not self.is_started():
                self.start()
            req = {"prompt": prompt}
            if image:
                req["image"] = str(image)
            if audio:
                req["audio"] = str(audio)
            self._proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
            emitted = False
            for chunk in self._read_response():
                cleaned = _strip_media_logs(chunk).strip()
                if cleaned:
                    emitted = True
                    yield cleaned
            if not emitted:
                yield "(no output — check the GPU driver and that Stage 1 exported the IR)"

    def stop(self):
        if self.is_started():
            try:
                self._proc.stdin.write(b'{"cmd":"quit"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None


# Module-level singleton: one resident server shared by the whole GUI session.
_SERVER: PipelineServer | None = None
_SERVER_LOCK = threading.Lock()


def get_server(install: str | None = None, config: str | None = None) -> PipelineServer:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = PipelineServer(install, config)
        return _SERVER


def is_warm() -> bool:
    """True if the resident server is up (model already compiled)."""
    return _SERVER is not None and _SERVER.is_started()


def warmup(install: str | None = None, config: str | None = None):
    """Start the resident server now (compile the model) so the first chat turn is fast."""
    get_server(install, config).start()


def stream_generate(prompt: str, image: str | None = None, audio: str | None = None,
                    install: str | None = None, config: str | None = None):
    """Yield the reply for one turn, live — token by token — via the resident server."""
    yield from get_server(install, config).generate(prompt, image=image, audio=audio)


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
    print("[runner] starting resident server (one-time compile)…")
    t0 = time.time()
    warmup()
    print(f"[runner] server ready in {time.time()-t0:.1f}s")
    print(f"[runner] prompt={p!r} image={img} audio={aud}\n--- streaming (live) ---")
    t1 = time.time()
    n = 0
    out = ""
    for chunk in stream_generate(p, img, aud):
        n += 1
        out = chunk
        if n == 1:
            print(f"[+{time.time()-t1:.2f}s] first token")
    print(f"--- done: {n} updates, first-token shown above ---")
    print(out)
    get_server().stop()
