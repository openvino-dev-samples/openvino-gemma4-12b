#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Download the Gemma4-12B model from Hugging Face into models/<name>.

Usage:
    python download_model.py [REPO_ID] [DEST_DIR]

Defaults:
    REPO_ID  = google/gemma-4-12B-it
    DEST_DIR = models/gemma-4-12B-it   (relative to the current directory)

Honors the standard HTTPS_PROXY / HTTP_PROXY environment variables, e.g.:
    set HTTPS_PROXY=http://your-proxy:port
    set HTTP_PROXY=http://your-proxy:port

The model is ~24 GB (a single FP16 safetensors shard plus tokenizer/config
files). Make sure you have enough free disk space.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it"
    dest = sys.argv[2] if len(sys.argv) > 2 else str(Path("models") / "gemma-4-12B-it")

    Path(dest).mkdir(parents=True, exist_ok=True)
    print(f"[download_model] repo_id = {repo_id}")
    print(f"[download_model] dest    = {dest}")
    print("[download_model] downloading (this can take a while for ~24 GB)...")

    snapshot_download(repo_id=repo_id, local_dir=dest)

    print(f"[download_model] done -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
