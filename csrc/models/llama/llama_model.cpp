#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const LlamaConfig &config, const infinicore::Device &device)
    : config_(config) {
    spdlog::info("LlamaModel::LlamaModel: START");

    spdlog::info("LlamaModel::LlamaModel: Device type={}, index={}",
                 static_cast<int>(device.getType()), device.getIndex());

    spdlog::info("LlamaModel::LlamaModel: About to initialize embed_tokens...");

    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size,
                              std::nullopt, infinicore::DataType::F32, device);

    spdlog::info("LlamaModel::LlamaModel: embed_tokens initialized, about to initialize layers (count={})...",
                 config.num_hidden_layers);

    // Initialize decoder layers
    spdlog::info("LlamaModel::LlamaModel: Starting to create {} decoder layers...",
                 config.num_hidden_layers);
    INFINICORE_NN_MODULE_VEC_INIT(layers, config.num_hidden_layers, LlamaDecoderLayer,
                                   config, device);

    spdlog::info("LlamaModel::LlamaModel: All {} decoder layers initialized, about to initialize norm...",
                 config.num_hidden_layers);

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps,
                              infinicore::DataType::F32, device);

    spdlog::info("LlamaModel::LlamaModel: norm initialized, about to initialize rotary_emb...");

    // Initialize Rotary Position Embeddings (shared across all layers)
    INFINICORE_NN_MODULE_INIT(rotary_emb, config.head_dim, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_J,
                              infinicore::DataType::F32, device);

    spdlog::info("LlamaModel::LlamaModel: rotary_emb initialized, constructor complete");
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                        const infinicore::Tensor &position_ids,
                                        std::vector<void *> *kv_caches) const {
    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = embed_tokens_->forward(input_ids);

    // 2. Process through all decoder layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        void *kv_cache = (kv_caches && i < kv_caches->size()) ? (*kv_caches)[i] : nullptr;
        hidden_states = layers_.at(i)->forward(hidden_states, position_ids, kv_cache);
    }

    // 3. Apply final layer normalization
    hidden_states = norm_->forward(hidden_states);

    return hidden_states;
}

} // namespace infinilm::models::llama
