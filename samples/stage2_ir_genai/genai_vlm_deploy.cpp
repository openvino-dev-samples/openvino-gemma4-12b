// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 of the Gemma4 deployment workflow.
//
// This sample consumes the OpenVINO IR that Stage 1 (openvino.pipeline.mx /
// yaml_pipeline_sample) exported into the model directory, and runs it through
// the *real* OpenVINO GenAI API: ov::genai::VLMPipeline.
//
// The exported Gemma4 IR is a VLM split:
//     openvino_language_model.xml        (LLM, takes inputs_embeds)
//     openvino_text_embeddings_model.xml (input_ids -> text_embeds)
//     openvino_vision_embeddings_model.xml
//     openvino_tokenizer.xml / openvino_detokenizer.xml
//     config.json (must have "model_type": "gemma4")
//
// VLMPipeline is the matching consumer because the language model takes
// inputs_embeds (not input_ids): even the text-only turn flows through the
// text_embeddings model first.
//
// Usage:
//   genai_vlm_deploy <MODEL_DIR> [DEVICE] [PROMPT] [IMAGE]
//     MODEL_DIR : directory containing the exported openvino_*.xml IR + config.json
//     DEVICE    : CPU | GPU (default: GPU)
//     PROMPT    : text prompt (default: a text-only question)
//     IMAGE     : optional path to an image; when given, runs an image+text turn

#include <openvino/genai/visual_language/pipeline.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "load_image.hpp"

namespace {

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> [DEVICE] [PROMPT] [IMAGE]\n"
                     "  MODEL_DIR : dir with exported openvino_*.xml IR + config.json\n"
                     "  DEVICE    : CPU | GPU (default: GPU)\n"
                     "  PROMPT    : text prompt\n"
                     "  IMAGE     : optional image file for an image+text turn\n";
        return EXIT_FAILURE;
    }

    const std::string model_dir = argv[1];
    const std::string device = (argc >= 3) ? argv[2] : "GPU";
    const std::string prompt = (argc >= 4) ? argv[3] : "How do black holes work?";
    const std::string image_path = (argc >= 5) ? argv[4] : "";

    // Cache compiled blobs on disk for GPU so the second run skips recompilation.
    ov::AnyMap properties;
    if (device == "GPU") {
        properties.insert({ov::cache_dir("genai_vlm_cache")});
    }

    std::cout << "[genai] Loading VLMPipeline from: " << model_dir << "\n";
    std::cout << "[genai] Device: " << device << "\n";
    ov::genai::VLMPipeline pipe(model_dir, device, properties);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 128;

    ov::genai::VLMDecodedResults result;
    std::cout << "\n[genai] Prompt: " << prompt << "\n--- generated ---\n";

    if (!image_path.empty()) {
        if (!std::filesystem::exists(image_path)) {
            std::cerr << "Image not found: " << image_path << "\n";
            return EXIT_FAILURE;
        }
        std::vector<ov::Tensor> images = utils::load_images(image_path);
        result = pipe.generate(prompt,
                               ov::genai::images(images),
                               ov::genai::generation_config(config),
                               ov::genai::streamer(print_subword));
    } else {
        result = pipe.generate(prompt,
                               ov::genai::generation_config(config),
                               ov::genai::streamer(print_subword));
    }

    std::cout << "\n--- done ---\n";
    return result.texts.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
} catch (const std::exception& e) {
    std::cerr << "\n[genai] Error: " << e.what() << "\n";
    return EXIT_FAILURE;
}
