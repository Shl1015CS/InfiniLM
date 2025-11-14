#include "llama_attention.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>
#include <cmath>

namespace infinilm::models::llama {

LlamaAttention::LlamaAttention(const LlamaConfig &config, const infinicore::Device &device)
    : hidden_size_(config.hidden_size),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      kv_dim_(config.kv_dim()),
      use_bias_(config.attention_bias) {
    spdlog::info("LlamaAttention::LlamaAttention: START");

    spdlog::info("LlamaAttention::LlamaAttention: About to initialize q_proj...");
    // Initialize projection layers
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, hidden_size_, use_bias_,
                               infinicore::DataType::F32, device);

    spdlog::info("LlamaAttention::LlamaAttention: q_proj initialized, about to initialize k_proj...");
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, kv_dim_, use_bias_,
                               infinicore::DataType::F32, device);

    spdlog::info("LlamaAttention::LlamaAttention: k_proj initialized, about to initialize v_proj...");
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, kv_dim_, use_bias_,
                               infinicore::DataType::F32, device);

    spdlog::info("LlamaAttention::LlamaAttention: v_proj initialized, about to initialize o_proj...");
    INFINICORE_NN_MODULE_INIT(o_proj, hidden_size_, hidden_size_, use_bias_,
                               infinicore::DataType::F32, device);

    spdlog::info("LlamaAttention::LlamaAttention: o_proj initialized, about to initialize rotary_emb...");

    // Initialize Rotary Position Embeddings
    INFINICORE_NN_MODULE_INIT(rotary_emb, head_dim_, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_J,
                              infinicore::DataType::F32, device);

    spdlog::info("LlamaAttention::LlamaAttention: rotary_emb initialized, constructor complete");
}

infinicore::Tensor LlamaAttention::forward(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &position_ids,
                                            void *kv_cache) const {
    spdlog::info("LlamaAttention::forward: START");
    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    spdlog::info("LlamaAttention::forward: Input shape: [{}, {}, {}]", shape[0], shape[1], shape[2]);

    // 1. Project Q, K, V
    spdlog::info("LlamaAttention::forward: Projecting Q, K, V");
    auto q = q_proj_->forward(hidden_states_mutable);  // [batch, seq_len, hidden_size]
    spdlog::info("LlamaAttention::forward: Q projected, shape: [{}, {}, {}]", q->shape()[0], q->shape()[1], q->shape()[2]);
    auto k = k_proj_->forward(hidden_states_mutable);  // [batch, seq_len, kv_dim]
    spdlog::info("LlamaAttention::forward: K projected, shape: [{}, {}, {}]", k->shape()[0], k->shape()[1], k->shape()[2]);
    auto v = v_proj_->forward(hidden_states_mutable);  // [batch, seq_len, kv_dim]
    spdlog::info("LlamaAttention::forward: V projected, shape: [{}, {}, {}]", v->shape()[0], v->shape()[1], v->shape()[2]);

    // 2. Reshape for multi-head attention
    spdlog::info("LlamaAttention::forward: Reshaping for multi-head attention");
    // Q: [batch, seq_len, hidden_size] -> [batch, seq_len, n_q_head, head_dim] -> [batch, n_q_head, seq_len, head_dim] -> [n_q_head, seq_len, head_dim]
    // K: [batch, seq_len, kv_dim] -> [batch, seq_len, n_kv_head, head_dim] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]
    // V: [batch, seq_len, kv_dim] -> [batch, seq_len, n_kv_head, head_dim] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]

    // Make tensors contiguous for reshaping
    spdlog::info("LlamaAttention::forward: Making tensors contiguous");
    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();
    spdlog::info("LlamaAttention::forward: Tensors made contiguous");

    // Reshape Q: [batch, seq_len, hidden_size] -> [batch, seq_len, n_q_head, head_dim] -> [batch, n_q_head, seq_len, head_dim] -> [n_q_head, seq_len, head_dim]
    spdlog::info("LlamaAttention::forward: Reshaping Q");
    spdlog::info("  num_attention_heads_: {}, head_dim_: {}, expected hidden_size: {}",
                 num_attention_heads_, head_dim_, num_attention_heads_ * head_dim_);
    spdlog::info("  q_cont shape: [{}, {}, {}]", q_cont->shape()[0], q_cont->shape()[1], q_cont->shape()[2]);

    // Validate dimensions before view
    size_t q_total_elements = q_cont->shape()[0] * q_cont->shape()[1] * q_cont->shape()[2];
    size_t q_expected_elements = batch_size * seq_len * num_attention_heads_ * head_dim_;
    if (q_total_elements != q_expected_elements) {
        spdlog::error("LlamaAttention::forward: Dimension mismatch for Q reshape!");
        spdlog::error("  Current total elements: {}", q_total_elements);
        spdlog::error("  Expected total elements: {}", q_expected_elements);
        spdlog::error("  Current shape: [{}, {}, {}]", q_cont->shape()[0], q_cont->shape()[1], q_cont->shape()[2]);
        spdlog::error("  Target shape: [{}, {}, {}, {}]", batch_size, seq_len, num_attention_heads_, head_dim_);
        throw std::runtime_error("Dimension mismatch in Q reshape");
    }

    spdlog::info("  About to call view({}, {}, {}, {})", batch_size, seq_len, num_attention_heads_, head_dim_);
    spdlog::default_logger()->flush();  // Ensure log is flushed before potentially hanging operation
    auto q_reshaped = q_cont->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    spdlog::info("  q_reshaped created, shape: [{}, {}, {}, {}]",
                 q_reshaped->shape()[0], q_reshaped->shape()[1], q_reshaped->shape()[2], q_reshaped->shape()[3]);
    spdlog::default_logger()->flush();  // Ensure log is flushed
    // Permute to [batch, n_q_head, seq_len, head_dim]
    spdlog::info("  About to permute Q");
    auto q_permuted = q_reshaped->permute({0, 2, 1, 3});
    spdlog::info("  q_permuted created, shape: [{}, {}, {}, {}]",
                 q_permuted->shape()[0], q_permuted->shape()[1], q_permuted->shape()[2], q_permuted->shape()[3]);
    // For batch=1 (common in inference), reshape to [n_q_head, seq_len, head_dim]
    // Note: For batch > 1, this would need to be handled differently
    // Make contiguous before final view since permute can make tensor non-contiguous
    spdlog::info("  Making q_permuted contiguous before final view");
    auto q_permuted_cont = q_permuted->contiguous();
    spdlog::info("  q_permuted_cont shape: [{}, {}, {}, {}]",
                 q_permuted_cont->shape()[0], q_permuted_cont->shape()[1], q_permuted_cont->shape()[2], q_permuted_cont->shape()[3]);
    spdlog::info("  About to final view Q to [{}, {}, {}]", num_attention_heads_, seq_len, head_dim_);
    spdlog::default_logger()->flush();
    auto q_attn = q_permuted_cont->view({num_attention_heads_, seq_len, head_dim_});
    spdlog::info("LlamaAttention::forward: Q reshaped to [{}, {}, {}]", q_attn->shape()[0], q_attn->shape()[1], q_attn->shape()[2]);

    // Reshape K: [batch, seq_len, kv_dim] -> [batch, seq_len, num_key_value_heads_, head_dim_] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]
    spdlog::info("LlamaAttention::forward: Reshaping K");
    auto k_reshaped = k_cont->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    // Permute to [batch, n_kv_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3});
    // Make contiguous before final view
    auto k_permuted_cont = k_permuted->contiguous();
    // Reshape to [n_kv_head, seq_len, head_dim]
    auto k_attn = k_permuted_cont->view({num_key_value_heads_, seq_len, head_dim_});
    spdlog::info("LlamaAttention::forward: K reshaped to [{}, {}, {}]", k_attn->shape()[0], k_attn->shape()[1], k_attn->shape()[2]);

    // Reshape V: [batch, seq_len, kv_dim] -> [batch, seq_len, num_key_value_heads_, head_dim_] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]
    spdlog::info("LlamaAttention::forward: Reshaping V");
    auto v_reshaped = v_cont->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    // Permute to [batch, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3});
    // Make contiguous before final view
    auto v_permuted_cont = v_permuted->contiguous();
    // Reshape to [n_kv_head, seq_len, head_dim]
    auto v_attn = v_permuted_cont->view({num_key_value_heads_, seq_len, head_dim_});
    spdlog::info("LlamaAttention::forward: V reshaped to [{}, {}, {}]", v_attn->shape()[0], v_attn->shape()[1], v_attn->shape()[2]);

    // 3. Prepare position_ids for RoPE
    // RoPE expects position_ids to be 1D [seq_len], but we receive [batch, seq_len]
    // Extract the first row and make it 1D
    spdlog::info("LlamaAttention::forward: Preparing position_ids for RoPE");
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;  // Initialize with position_ids (fallback for 1D case)
    if (pos_shape.size() == 2) {
        spdlog::info("  position_ids shape: [{}, {}], extracting first row", pos_shape[0], pos_shape[1]);
        // Extract first row: narrow dimension 0, start=0, length=1
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        // Make contiguous and view as 1D
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
        spdlog::info("  pos_ids_for_rope shape: [{}]", pos_ids_for_rope->shape()[0]);
    } else if (pos_shape.size() == 1) {
        spdlog::info("  position_ids shape: [{}], already 1D", pos_shape[0]);
        // pos_ids_for_rope already initialized with position_ids
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K
    // RoPE expects [seq_len, n_head, head_dim], but we have [n_head, seq_len, head_dim]
    // Permute to [seq_len, n_head, head_dim] before RoPE, then permute back
    spdlog::info("LlamaAttention::forward: Applying RoPE to Q");
    spdlog::info("  q_attn shape before permute: [{}, {}, {}]", q_attn->shape()[0], q_attn->shape()[1], q_attn->shape()[2]);
    // Permute from [n_head, seq_len, head_dim] to [seq_len, n_head, head_dim]
    auto q_for_rope = q_attn->permute({1, 0, 2});  // [seq_len, n_head, head_dim]
    spdlog::info("  q_for_rope shape: [{}, {}, {}]", q_for_rope->shape()[0], q_for_rope->shape()[1], q_for_rope->shape()[2]);
    auto q_rope_out = rotary_emb_->forward(q_for_rope, pos_ids_for_rope);
    // Permute back from [seq_len, n_head, head_dim] to [n_head, seq_len, head_dim]
    auto q_rope = q_rope_out->permute({1, 0, 2});  // [n_head, seq_len, head_dim]
    spdlog::info("LlamaAttention::forward: RoPE applied to Q, shape: [{}, {}, {}]",
                  q_rope->shape()[0], q_rope->shape()[1], q_rope->shape()[2]);

    spdlog::info("LlamaAttention::forward: Applying RoPE to K");
    spdlog::info("  k_attn shape before permute: [{}, {}, {}]", k_attn->shape()[0], k_attn->shape()[1], k_attn->shape()[2]);
    // Permute from [n_kv_head, seq_len, head_dim] to [seq_len, n_kv_head, head_dim]
    auto k_for_rope = k_attn->permute({1, 0, 2});  // [seq_len, n_kv_head, head_dim]
    spdlog::info("  k_for_rope shape: [{}, {}, {}]", k_for_rope->shape()[0], k_for_rope->shape()[1], k_for_rope->shape()[2]);
    auto k_rope_out = rotary_emb_->forward(k_for_rope, pos_ids_for_rope);
    // Permute back from [seq_len, n_kv_head, head_dim] to [n_kv_head, seq_len, head_dim]
    auto k_rope = k_rope_out->permute({1, 0, 2});  // [n_kv_head, seq_len, head_dim]
    spdlog::info("LlamaAttention::forward: RoPE applied to K, shape: [{}, {}, {}]",
                  k_rope->shape()[0], k_rope->shape()[1], k_rope->shape()[2]);

    // 5. Prepare KV caches for attention operation
    // The attention operation requires cache tensors with at least seq_len capacity
    // For first pass (pos=0), we create caches with seq_len capacity
    size_t cache_capacity = seq_len;  // Cache capacity (at least seq_len for first pass)
    auto k_cache = infinicore::Tensor::empty({num_key_value_heads_, cache_capacity, head_dim_},
                                             k_rope->dtype(), k_rope->device());
    auto v_cache = infinicore::Tensor::empty({num_key_value_heads_, cache_capacity, head_dim_},
                                             v_attn->dtype(), v_attn->device());

    // 6. Call attention operation
    // attention expects: q [n_q_head, seq_len, head_dim], k [n_kv_head, seq_len, head_dim], v [n_kv_head, seq_len, head_dim]
    // Returns: [seq_len, n_q_head, head_dim]
    // Note: V doesn't get RoPE applied, so we use v_attn directly
    size_t pos = 0;  // Position in cache (0 for first pass)
    spdlog::info("LlamaAttention::forward: About to call attention operation");
    spdlog::info("  q_rope shape: [{}, {}, {}]", q_rope->shape()[0], q_rope->shape()[1], q_rope->shape()[2]);
    spdlog::info("  k_rope shape: [{}, {}, {}]", k_rope->shape()[0], k_rope->shape()[1], k_rope->shape()[2]);
    spdlog::info("  v_attn shape: [{}, {}, {}]", v_attn->shape()[0], v_attn->shape()[1], v_attn->shape()[2]);
    spdlog::info("  cache capacity: {}", cache_capacity);
    auto attn_output = infinicore::op::attention(q_rope, k_rope, v_attn, k_cache, v_cache, pos);
    spdlog::info("LlamaAttention::forward: Attention operation completed");

    // 7. Reshape output back: [seq_len, n_q_head, head_dim] -> [batch, seq_len, hidden_size]
    // attention returns [seq_len, n_q_head, head_dim]
    spdlog::info("LlamaAttention::forward: Reshaping attention output");
    spdlog::info("  attn_output shape: [{}, {}, {}]", attn_output->shape()[0], attn_output->shape()[1], attn_output->shape()[2]);
    // Reshape to [n_q_head, seq_len, head_dim] -> [batch, n_q_head, seq_len, head_dim] -> [batch, seq_len, n_q_head, head_dim] -> [batch, seq_len, hidden_size]
    spdlog::info("  Making attn_output contiguous");
    auto attn_cont = attn_output->contiguous();
    spdlog::info("  attn_cont shape: [{}, {}, {}]", attn_cont->shape()[0], attn_cont->shape()[1], attn_cont->shape()[2]);
    // Permute from [seq_len, n_q_head, head_dim] to [n_q_head, seq_len, head_dim]
    spdlog::info("  Permuting attn_cont to [n_q_head, seq_len, head_dim]");
    auto attn_permuted = attn_cont->permute({1, 0, 2});  // [n_q_head, seq_len, head_dim]
    spdlog::info("  attn_permuted shape: [{}, {}, {}]", attn_permuted->shape()[0], attn_permuted->shape()[1], attn_permuted->shape()[2]);
    // Reshape to [batch, n_q_head, seq_len, head_dim]
    // Make contiguous before view since permute can make tensor non-contiguous
    spdlog::info("  Making attn_permuted contiguous before view");
    auto attn_permuted_cont = attn_permuted->contiguous();
    spdlog::info("  Viewing attn_permuted_cont to [{}, {}, {}, {}]", batch_size, num_attention_heads_, seq_len, head_dim_);
    auto attn_batch = attn_permuted_cont->view({batch_size, num_attention_heads_, seq_len, head_dim_});
    spdlog::info("  attn_batch shape: [{}, {}, {}, {}]", attn_batch->shape()[0], attn_batch->shape()[1], attn_batch->shape()[2], attn_batch->shape()[3]);
    // Permute to [batch, seq_len, n_q_head, head_dim]
    spdlog::info("  Permuting attn_batch to [batch, seq_len, n_q_head, head_dim]");
    auto attn_final = attn_batch->permute({0, 2, 1, 3});
    spdlog::info("  attn_final shape: [{}, {}, {}, {}]", attn_final->shape()[0], attn_final->shape()[1], attn_final->shape()[2], attn_final->shape()[3]);
    // Reshape to [batch, seq_len, hidden_size]
    // Make contiguous before view since permute can make tensor non-contiguous
    spdlog::info("  Making attn_final contiguous before final view");
    auto attn_final_cont = attn_final->contiguous();
    spdlog::info("  Viewing attn_final_cont to [{}, {}, {}]", batch_size, seq_len, hidden_size_);
    auto attn_flat = attn_final_cont->view({batch_size, seq_len, hidden_size_});
    spdlog::info("  attn_flat shape: [{}, {}, {}]", attn_flat->shape()[0], attn_flat->shape()[1], attn_flat->shape()[2]);

    // 8. Apply output projection
    spdlog::info("LlamaAttention::forward: Applying output projection");
    auto output = o_proj_->forward(attn_flat);
    spdlog::info("LlamaAttention::forward: Output projection completed, output shape: [{}, {}, {}]",
                 output->shape()[0], output->shape()[1], output->shape()[2]);
    spdlog::info("LlamaAttention::forward: END");
    return output;
}

} // namespace infinilm::models::llama
