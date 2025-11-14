#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::llama {

LlamaDecoderLayer::LlamaDecoderLayer(const LlamaConfig &config, const infinicore::Device &device) {
    spdlog::info("LlamaDecoderLayer::LlamaDecoderLayer: START");

    spdlog::info("LlamaDecoderLayer::LlamaDecoderLayer: About to initialize input_layernorm...");
    // Initialize layer normalization layers
    INFINICORE_NN_MODULE_INIT(input_layernorm, config.hidden_size, config.rms_norm_eps,
                              infinicore::DataType::F32, device);

    spdlog::info("LlamaDecoderLayer::LlamaDecoderLayer: input_layernorm initialized, about to initialize post_attention_layernorm...");
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, config.hidden_size, config.rms_norm_eps,
                              infinicore::DataType::F32, device);

    spdlog::info("LlamaDecoderLayer::LlamaDecoderLayer: post_attention_layernorm initialized, about to initialize self_attn...");

    // Initialize attention and MLP modules
    INFINICORE_NN_MODULE_INIT(self_attn, config, device);

    spdlog::info("LlamaDecoderLayer::LlamaDecoderLayer: self_attn initialized, about to initialize mlp...");
    INFINICORE_NN_MODULE_INIT(mlp, config, device);

    spdlog::info("LlamaDecoderLayer::LlamaDecoderLayer: mlp initialized, constructor complete");
}

infinicore::Tensor LlamaDecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &position_ids,
                                               void *kv_cache) const {
    // Save residual for attention
    auto residual = hidden_states;

    // 1. Pre-attention layer normalization
    auto normed_states = input_layernorm_->forward(hidden_states);

    // 2. Self-attention with residual connection
    auto attn_output = self_attn_->forward(normed_states, position_ids, kv_cache);

    // Add residual: hidden_states = hidden_states + attn_output
    auto output = infinicore::op::add(residual, attn_output);

    // Save residual for MLP
    residual = output;

    // 3. Post-attention layer normalization
    normed_states = post_attention_layernorm_->forward(output);

    // 4. MLP with residual connection
    auto mlp_output = mlp_->forward(normed_states);

    // Add residual: output = output + mlp_output
    output = infinicore::op::add(residual, mlp_output);

    return output;
}

} // namespace infinilm::models::llama
