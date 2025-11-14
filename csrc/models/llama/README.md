# Llama Model Skeleton Test

This directory contains tests to validate the Llama model skeleton architecture.

## Test Files

- `test_llama_skeleton.cpp` - C++ test for immediate validation
- `../scripts/test_llama_skeleton.py` - Python test (requires pybind11 bindings)

## Running Tests

### C++ Test

Build and run the C++ test:

```bash
# Build the test
xmake build test_llama_skeleton

# Run the test
xmake run test_llama_skeleton
```

Or directly:

```bash
# Build
xmake build test_llama_skeleton

# Run (from build directory)
./build/linux/x86_64/release/test_llama_skeleton
```

### Python Test

The Python test runs the C++ test binary and validates its output:

```bash
# Build the test first (if not already built)
xmake build test_llama_skeleton

# Run Python test wrapper
python3 scripts/test_llama_skeleton.py
```

The Python script automatically locates and executes the C++ test binary, making it convenient to run from the command line.

## What the Tests Validate

1. **Model Construction**: Verifies that LlamaModel and LlamaForCausalLM can be instantiated
2. **Parameter Registration**: Checks that all expected parameters are registered via `state_dict()`
3. **Parameter Shapes**: Validates that parameter shapes match the expected Llama architecture:
   - `embed_tokens.weight`: `[vocab_size, hidden_size]`
   - `layers.{i}.input_layernorm.weight`: `[hidden_size]`
   - `layers.{i}.self_attn.{q,k,v,o}_proj.weight`: Appropriate shapes for GQA
   - `layers.{i}.mlp.{gate,up,down}_proj.weight`: MLP projection shapes
   - `layers.{i}.post_attention_layernorm.weight`: `[hidden_size]`
   - `norm.weight`: `[hidden_size]`
   - `lm_head.weight`: `[vocab_size, hidden_size]` (for LlamaForCausalLM)

4. **Module Hierarchy**: Ensures the naming convention matches HuggingFace's structure

## Expected Output

The C++ test should output:

```
==============================================
Llama Model Skeleton Validation Test
==============================================

Configuration:
  vocab_size: 32000
  hidden_size: 2048
  ...
✓ Configuration validated

1. Creating LlamaModel...
   ✓ Model created successfully

2. Retrieving state_dict()...
   ✓ Found X parameters

3. Validating parameter structure...
   ✓ Parameter count matches: X
   ✓ All parameter shapes validated

4. Creating LlamaForCausalLM...
   ✓ LlamaForCausalLM created with X parameters
   ✓ lm_head.weight shape correct: [32000, 2048]

==============================================
✓ All tests PASSED!
==============================================
```

## Troubleshooting

If the test fails:

1. **Compilation errors**: Ensure InfiniCore is built and installed
2. **Missing parameters**: Check that all modules are properly registered using `INFINICORE_NN_MODULE_INIT`
3. **Shape mismatches**: Verify the configuration matches the expected architecture
4. **Link errors**: Ensure all required InfiniCore libraries are linked
