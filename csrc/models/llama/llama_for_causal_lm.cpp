#include "llama_for_causal_lm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config, const infinicore::Device &device) {
    spdlog::info("LlamaForCausalLM::LlamaForCausalLM: START");

    spdlog::info("LlamaForCausalLM::LlamaForCausalLM: Device type={}, index={}",
                 static_cast<int>(device.getType()), device.getIndex());

    spdlog::info("LlamaForCausalLM::LlamaForCausalLM: About to initialize model...");

    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, config, device);

    spdlog::info("LlamaForCausalLM::LlamaForCausalLM: Model initialized, about to initialize lm_head...");

    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer
    INFINICORE_NN_MODULE_INIT(lm_head, config.hidden_size, config.vocab_size, false,
                              infinicore::DataType::F32, device);

    spdlog::info("LlamaForCausalLM::LlamaForCausalLM: lm_head initialized, constructor complete");
}

infinicore::Tensor LlamaForCausalLM::forward(const infinicore::Tensor &input_ids,
                                              const infinicore::Tensor &position_ids,
                                              std::vector<void *> *kv_caches) const {
    // 1. Forward through base model to get hidden states
    auto hidden_states = model_->forward(input_ids, position_ids, kv_caches);

    // 2. Apply language modeling head to get logits
    // Note: Linear::forward expects non-const Tensor&, but Tensor is a shared_ptr-like wrapper
    // so we can use it directly (the const-ness doesn't prevent usage)
    auto logits = lm_head_->forward(hidden_states);

    return logits;
}

} // namespace infinilm::models::llama
