// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 (primary) - deploy the exported Gemma4 IR through the OpenVINO GenAI
// compatible facade: ov::pipeline::LLMPipeline.
//
// ov::pipeline::LLMPipeline is a drop-in replacement for ov::genai::LLMPipeline.
// When the model directory contains a pipeline.yaml, the facade routes the
// request through the pipeline engine (which loads the engine-flavored IR that
// Stage 1 exported: openvino_language_model.xml taking inputs_embeds, the
// text/vision/audio embedding models, and the tokenizer IR). Without a
// pipeline.yaml it would bypass to the upstream ov::genai implementation.
//
// This is the path that actually consumes the IR exported by
// openvino.pipeline.mx. For the genai-native ov::genai::VLMPipeline contrast
// (and why it does NOT consume this IR directly), see NOTES.md and the
// reference sample genai_vlm_deploy.cpp.
//
// The header path below resolves to the GenAI-compatible facade that the
// distribution package ships (shadow header).
//
// Usage:
//   genai_facade_llm <MODEL_DIR> [DEVICE] [PROMPT]
//     MODEL_DIR : model dir containing a pipeline.yaml + exported openvino_*.xml
//     DEVICE    : CPU | GPU   (default: GPU)  - note: device is set in the YAML
//     PROMPT    : text prompt

#include <openvino/genai/llm_pipeline.hpp>

#include <iostream>
#include <string>

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <MODEL_DIR> [DEVICE] [PROMPT]\n";
        return EXIT_FAILURE;
    }

    const std::string model_dir = argv[1];
    const std::string device = (argc >= 3) ? argv[2] : "GPU";
    const std::string prompt = (argc >= 4) ? argv[3] : "How do black holes work?";

    // ov::pipeline::LLMPipeline (drop-in for ov::genai::LLMPipeline). With a
    // pipeline.yaml present in model_dir this routes through the engine and
    // deploys the exported IR. The per-module device comes from the YAML; the
    // device argument here is forwarded for the upstream-bypass case.
    std::cout << "[genai-facade] Loading LLMPipeline from: " << model_dir << "\n";
    ov::genai::LLMPipeline pipe(model_dir, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 128;

    std::cout << "[genai-facade] Prompt: " << prompt << "\n--- generated ---\n";
    ov::genai::DecodedResults result = pipe.generate(prompt, config);
    std::cout << result << "\n--- done ---\n";

    return EXIT_SUCCESS;
} catch (const std::exception& e) {
    std::cerr << "\n[genai-facade] Error: " << e.what() << "\n";
    return EXIT_FAILURE;
}
