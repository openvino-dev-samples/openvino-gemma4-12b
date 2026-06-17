#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Convert the HF tokenizer into OpenVINO tokenizer/detokenizer IR.

The pipeline's TextEncoderModule loads openvino_tokenizer.xml /
openvino_detokenizer.xml from the model directory. This script produces them
once from the model's tokenizer.json using openvino-tokenizers.

Usage:
    python prepare_tokenizer.py [MODEL_DIR]

Default MODEL_DIR = models/gemma-4-12B-it

Requirements (install into your venv first):
    pip install transformers openvino-tokenizers
    # pin openvino-tokenizers to the package's OpenVINO version, e.g. 2026.2.*

Note: pass MODEL_DIR as a real filesystem path. transformers treats a bare
"namespace/name" string as a Hub repo id, so a local path with backslashes
(Windows) or an absolute path is required.
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import openvino as ov


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path("models") / "gemma-4-12B-it")
    model_dir = str(Path(model_dir).resolve())  # absolute path so it is not read as a Hub repo id

    print(f"[prepare_tokenizer] model_dir = {model_dir}")
    tok_hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ov_tok, ov_detok = convert_tokenizer(tok_hf, with_detokenizer=True)

    tok_xml = str(Path(model_dir) / "openvino_tokenizer.xml")
    detok_xml = str(Path(model_dir) / "openvino_detokenizer.xml")
    ov.save_model(ov_tok, tok_xml)
    ov.save_model(ov_detok, detok_xml)

    print(f"[prepare_tokenizer] wrote {tok_xml}")
    print(f"[prepare_tokenizer] wrote {detok_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
