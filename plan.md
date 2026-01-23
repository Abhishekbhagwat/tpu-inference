# RFC: Qwen3-VL Vision-Language Model for vLLM TPU (tpu-inference)

**Author**: @abhishekbhgwt
**Date**: January 22, 2026
**Status**: Draft
**Base PR**: [#1374](https://github.com/vllm-project/tpu-inference/pull/1374)

---

## 1. Abstract

This RFC documents the integration and optimization plan for **Qwen3-VL** (Qwen3 Vision-Language Model) in vLLM's tpu-inference repository. The implementation builds on PR #1374, which provides a working foundation, and outlines improvements for:

- **Production-ready performance** on TPU v5e/v6e pods
- **Tensor parallelism** for 8B and 32B model variants
- **DeepStack visual fusion** with efficient caching
- **Interleaved M-RoPE** for superior spatial-temporal reasoning
- **FP8 quantization** support for memory efficiency

The initial target is **Qwen3-VL-8B-Instruct** with plans to extend to 2B, 4B, and 32B variants, as well as **NVIDIA Cosmos-Reason2** models which share the same architecture.

---

## 2. Motivation

### 2.1 Why Qwen3-VL?

Qwen3-VL represents a significant advancement over Qwen2.5-VL with three key architectural innovations:

| Feature | Qwen2.5-VL | Qwen3-VL |
|---------|------------|----------|
| **Vision Encoder** | ViT-based | SigLIP-2 (400M params) |
| **Position Encoding (Vision)** | Rotary (2D) | Learned + Bilinear Interpolation |
| **Position Encoding (Text)** | Grouped M-RoPE | Interleaved M-RoPE |
| **Visual Fusion** | Single injection point | DeepStack (multi-layer) |
| **Video Understanding** | Basic temporal | 2-hour video support |

### 2.2 Key Benchmark Improvements

| Benchmark | Qwen2.5-VL (72B) | Qwen3-VL (32B) | Notes |
|-----------|------------------|----------------|-------|
| DocVQA | 96.4% | 96.5% | With 2x fewer params |
| MMMU-Pro | 48.3% | 69.3% | +44% improvement |
| OCRBench | 877 | 875 | 39 languages |
| MMLongBench-Doc | - | 56.2% | Long document analysis |

### 2.3 Why tpu-inference?

- **Native TPU optimization**: JAX/XLA compilation with automatic sharding
- **High throughput**: Continuous batching and RadixCache prefix sharing
- **Existing multimodal framework**: Qwen2.5-VL already supported
- **Production ready**: OpenAI-compatible API

### 2.4 NVIDIA Cosmos-Reason2 Compatibility

The **nvidia/Cosmos-Reason2** models (2B and 8B) use identical architecture to Qwen3-VL, enabling direct support with configuration-only changes.

---

## 3. Architecture Overview

### 3.1 Model Architecture

```
                    +---------------------------------------+
                    |   Qwen3VLForConditionalGeneration     |
                    +---------------------------------------+
                                      |
              +-----------------------+------------------------+
              |                                                |
    +---------v---------+                         +------------v-----------+
    | Qwen3VLVisionTransformer                    |    Qwen3VLModel        |
    | (SigLIP-2 based)  |                         |  (Text Decoder)        |
    +-------------------+                         +------------------------+
              |                                                |
    +---------v---------+                         +------------v-----------+
    | - 3D PatchEmbed   |                         | - Token Embedding      |
    | - Learned PosEmb  |                         | - Decoder Layers (28+) |
    |   + Bilinear Interp                         |   + Interleaved MRoPE  |
    | - Vision Blocks   |--DeepStack Features-->  |   + GQA Attention      |
    |   (24-27 layers)  |   @ layers [8,16,24]    |   + SwiGLU MLP         |
    | - DeepStack Mergers                         | - Final RMSNorm        |
    | - Patch Merger    |                         +------------------------+
    +-------------------+
```

### 3.2 DeepStack Visual Fusion

Unlike Qwen2.5-VL which injects visual features only at the embedding layer, Qwen3-VL uses **DeepStack** to inject intermediate vision features into early LLM layers:

```
Vision Encoder:
  Layer 8  --> DeepStack Merger 0 --> inject at LLM Layer 0-N
  Layer 16 --> DeepStack Merger 1 --> inject at LLM Layer 0-N
  Layer 24 --> DeepStack Merger 2 --> inject at LLM Layer 0-N
  Final    --> Main Merger        --> Token Embedding Replace
```

**Benefits:**
- Captures multi-scale visual features (fine-grained to semantic)
- Sharpens image-text alignment for document understanding
- Enables 4x more visual tokens with minimal context length increase

### 3.3 Interleaved M-RoPE

Qwen3-VL uses **Interleaved M-RoPE** instead of grouped M-RoPE:

```python
# Grouped M-RoPE (Qwen2.5-VL): frequencies grouped by dimension
# head_dim=128 -> [T:0-42, H:43-85, W:86-127]
mrope_section = [24, 20, 20]  # 64 values total (half of head_dim)

# Interleaved M-RoPE (Qwen3-VL): frequencies interleaved
# Pattern: [t0, h0, w0, t1, h1, w1, t2, h2, w2, ...]
def apply_interleaved_mrope(freqs, mrope_section=[24, 20, 20]):
    t, h, w = mrope_section
    result = freqs[0].copy()
    h_indices = jnp.arange(1, h * 3, 3)  # [1, 4, 7, 10, ...]
    w_indices = jnp.arange(2, w * 3, 3)  # [2, 5, 8, 11, ...]
    result = result.at[..., h_indices].set(freqs[1][..., h_indices])
    result = result.at[..., w_indices].set(freqs[2][..., w_indices])
    return result
```

**Benefits:**
- Distributes frequencies uniformly across low and high-frequency bands
- Position IDs grow more slowly than vanilla RoPE
- Better long-context scaling (256K to 1M tokens with factor=2-3)

---

## 4. Model Configurations

### 4.1 Supported Variants

| Model | Vision Depth | Vision Hidden | Text Layers | Text Hidden | DeepStack Indices | Status |
|-------|--------------|---------------|-------------|-------------|-------------------|--------|
| **Qwen3-VL-2B** | 24 | 1024 | 28 | 2048 | [5, 11, 17] | Planned |
| **Qwen3-VL-4B** | 24 | 1024 | 36 | 2560 | [5, 11, 17] | Planned |
| **Qwen3-VL-8B** | 27 | 1152 | 36 | 4096 | [8, 16, 24] | Priority |
| **Qwen3-VL-32B** | 27 | 1152 | 64 | 5120 | [8, 16, 24] | Planned |
| **Cosmos-Reason2-2B** | 24 | 1024 | 28 | 2048 | [5, 11, 17] | Compatible |
| **Cosmos-Reason2-8B** | 27 | 1152 | 36 | 4096 | [8, 16, 24] | Compatible |

### 4.2 Key Configuration Parameters

```python
# Vision Configuration (Qwen3VLVisionConfig)
hidden_size: int = 1152          # SigLIP-2 hidden dimension
intermediate_size: int = 4096    # Vision MLP intermediate
depth: int = 27                  # Number of vision transformer blocks
num_heads: int = 16              # Vision attention heads
patch_size: int = 16             # Spatial patch size (SigLIP-2)
temporal_patch_size: int = 2     # Temporal patch size for video
spatial_merge_size: int = 2      # 2x2 spatial merge
num_position_embeddings: int = 2304  # 48x48 learned position grid
deepstack_visual_indexes: tuple = (8, 16, 24)  # DeepStack extraction layers
out_hidden_size: int = 4096      # Output projection to LLM dimension

# Text Configuration (Qwen3VLTextConfig)
vocab_size: int = 151936
hidden_size: int = 4096
intermediate_size: int = 22016   # SwiGLU MLP
num_hidden_layers: int = 36
num_attention_heads: int = 32
num_key_value_heads: int = 8     # GQA (4x compression)
head_dim: int = 128
rope_theta: float = 5_000_000    # Extended context RoPE
mrope_section: tuple = (24, 20, 20)  # T, H, W partitions (sum=64=head_dim/2)
rms_norm_eps: float = 1e-6
```

---

## 5. Current Implementation Analysis (PR #1374)

### 5.1 What's Implemented

| Component | File | Status |
|-----------|------|--------|
| Vision Transformer | `qwen3_vl.py:1066-1453` | Complete |
| 3D Patch Embedding | `qwen3_vl.py:735-787` | Complete |
| Learned Position Embed | `qwen3_vl.py:1103-1143` | Complete |
| Bilinear Interpolation | `qwen3_vl.py:1250-1347` | Complete |
| DeepStack Extraction | `qwen3_vl.py:1366-1386` | Complete |
| DeepStack Mergers | `qwen3_vl.py:996-1063` | Complete |
| Text Decoder (MRoPE) | `qwen3_vl.py:1456-1586` | Complete |
| Interleaved MRoPE | `qwen3_vl.py:310-330` | Complete |
| Weight Loading | `qwen3_vl.py:1970-2036` | Complete |
| Model Registration | `model_loader.py:66-76` | Complete |
| Multimodal Manager | `multimodal_manager.py` | DeepStack support added |
| Unit Tests | `test_qwen3_vl.py` | ~40 tests |

### 5.2 Identified Issues for Improvement

#### 5.2.1 Code Duplication (High Priority)

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Custom RMSNorm | `qwen3_vl.py:333-348` | Replace with `nnx.RMSNorm` |
| Custom RoPE application | `qwen3_vl.py:351-389` | Use `rope_interface.apply_rope` |
| Text attention reimplemented | `qwen3_vl.py:469-596` | Extend `Qwen3Attention` |
| Text MLP reimplemented | `qwen3_vl.py:599-630` | Reuse `Qwen3MLP` from qwen3.py |

**Before (current):**
```python
class Qwen3VLTextRMSNorm(nnx.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=jnp.bfloat16):
        self.weight = nnx.Param(jnp.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        # Custom implementation...
```

**After (recommended):**
```python
# Use existing nnx.RMSNorm
self.input_layernorm = nnx.RMSNorm(
    hidden_size, epsilon=rms_norm_eps, param_dtype=dtype,
    scale_init=nnx.with_partitioning(init_fn, (None,)), rngs=rng
)
```

#### 5.2.2 Performance Optimizations (Medium Priority)

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| Vision encoder not parallelized | Bottleneck for large images | Add tensor parallelism |
| DeepStack injection uses cumsum | Suboptimal for long sequences | Use scatter operations |
| No vision encoder warmup shapes | JIT recompilation | Add common resolution precompilation |
| Segment ID generation not cached | Repeated computation | Cache for common grid_thw |

#### 5.2.3 Missing Features (Low Priority)

| Feature | Status | Notes |
|---------|--------|-------|
| FP8 quantization | Not implemented | Planned for Phase 5 |
| Dynamic batching optimization | Basic | Optimize for mixed modalities |
| Video frame batching | Not optimized | Add temporal batching |
| KV cache quantization | Uses base implementation | Verify compatibility |

---

## 6. Implementation Plan

### Phase 1: Code Refactoring (Week 1-2)

**Goal:** Reduce code duplication, improve maintainability.

#### Task 1.1: Integrate with Existing Components

```python
# Replace custom RMSNorm
from flax import nnx
# Use nnx.RMSNorm directly instead of Qwen3VLTextRMSNorm

# Replace custom RoPE application
from tpu_inference.layers.jax.rope_interface import apply_rope
# apply_rope already supports M-RoPE via mrope_section in rope_scaling
```

#### Task 1.2: Extend Qwen3 Base Classes

```python
from tpu_inference.models.jax.qwen3 import Qwen3DecoderLayer, Qwen3Attention

class Qwen3VLTextDecoderLayer(Qwen3DecoderLayer):
    """Extended decoder layer with DeepStack injection support."""

    def __call__(self, kv_cache, hidden_states, attention_metadata,
                 deepstack_embeds=None, visual_mask=None):
        # Call parent implementation
        kv_cache, hidden_states = super().__call__(kv_cache, hidden_states,
                                                    attention_metadata)
        # Add DeepStack injection
        if deepstack_embeds is not None and visual_mask is not None:
            hidden_states = self._inject_visual_features(
                hidden_states, visual_mask, deepstack_embeds)
        return kv_cache, hidden_states
```

#### Task 1.3: Update rope_interface for Interleaved M-RoPE

The existing `rope_interface.apply_rope` supports grouped M-RoPE. Add interleaved variant:

```python
# In rope_interface.py, add support for interleaved=True
def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rope_theta: float = 10000,
    rope_scaling: Dict[str, Any] = None,
    rope_input_ordering: str = "split",
    interleaved_mrope: bool = False,  # NEW: Enable interleaved pattern
) -> jax.Array:
    if positions.ndim == 2 and positions.shape[0] == 3:
        if interleaved_mrope:
            return _apply_interleaved_mrope(inputs, positions, head_dim,
                                            rope_theta, rope_scaling)
        else:
            return _apply_grouped_mrope(...)  # Existing implementation
```

### Phase 2: Vision Encoder Optimization (Week 2-3)

**Goal:** Optimize vision encoder for TPU performance.

#### Task 2.1: Tensor Parallelism for Vision Blocks

```python
# Current: Vision encoder runs on single device
# Target: Distribute across TP dimension for large models

vision_sharding = {
    "patch_embed.proj.kernel": P(None, None, None, None, "model"),
    "blocks.*.attn.qkv_proj.kernel": P(None, "model"),
    "blocks.*.attn.proj.kernel": P("model", None),
    "blocks.*.mlp.fc1.kernel": P(None, "model"),
    "blocks.*.mlp.fc2.kernel": P("model", None),
}
```

#### Task 2.2: Flash Attention Tuning

```python
# Optimize block sizes for vision sequence lengths
self.flash_attention = sharded_flash_attention(
    mesh=mesh,
    causal=False,
    sm_scale=1.0 / math.sqrt(self.head_dim),
    vmem_limit_bytes=256 * 1024 * 1024,  # Increase for larger images
    block_q=64,   # Tune based on typical vision sequence lengths
    block_k_major=128,
)
```

#### Task 2.3: Comprehensive Vision Encoder Warmup

```python
def precompile_vision_encoder(self, run_compilation_fn):
    """Pre-compile for common image resolutions."""
    common_shapes = [
        (336, 336),    # Base resolution
        (448, 448),    # 1.33x
        (672, 672),    # 2x
        (896, 896),    # 2.67x (Qwen3-VL default max)
        (1344, 1344),  # 4x
        (504, 336),    # 3:2 aspect ratio
        (336, 504),    # 2:3 aspect ratio
        (672, 448),    # 3:2 at higher res
    ]
    for h, w in common_shapes:
        self._compile_for_resolution(run_compilation_fn, h, w)
```

### Phase 3: DeepStack Optimization (Week 3-4)

**Goal:** Optimize DeepStack feature injection and caching.

#### Task 3.1: Efficient DeepStack Injection

```python
# Current implementation uses cumsum + gather
def _inject_visual_features_current(self, hidden_states, visual_pos_mask,
                                     visual_embeds):
    gather_indices = jnp.cumsum(flat_mask, dtype=jnp.int32)
    # ... gather and add

# Optimized: Use direct scatter operations
def _inject_visual_features_optimized(self, hidden_states, visual_pos_mask,
                                        visual_embeds):
    """Optimized DeepStack injection using scatter."""
    indices = jnp.nonzero(visual_pos_mask, size=visual_embeds.shape[0])[0]
    return hidden_states.at[indices].add(visual_embeds)
```

#### Task 3.2: DeepStack Cache Integration

The multimodal_manager.py already has DeepStack caching. Verify and optimize:

```python
# In multimodal_manager.py
def execute_mm_encoder(self, scheduler_output):
    # ... existing code ...
    if deepstack_group_outputs is not None:
        for (mm_hash, pos_info), output, deepstack_output in zip(...):
            # Cache DeepStack embeddings
            self.runner.deepstack_cache[mm_hash] = deepstack_output
```

### Phase 4: MRoPE Optimization (Week 4)

**Goal:** Optimize M-RoPE computation for multimodal sequences.

#### Task 4.1: Cache MRoPE Frequencies

```python
@functools.lru_cache(maxsize=32)
def _get_cached_inv_freq(dim: int, rope_theta: float) -> jax.Array:
    """Cache inverse frequencies for common configurations."""
    return 1.0 / (rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

class Qwen3VLTextRotaryEmbedding(nnx.Module):
    def __call__(self, position_ids):
        inv_freq = _get_cached_inv_freq(self.dim, self.rope_theta)
        # ... rest of implementation
```

#### Task 4.2: Pre-compute Position IDs

Position ID computation happens in `get_mrope_input_positions` during request preprocessing. Ensure this is efficient:

```python
def get_mrope_input_positions(
    input_tokens: List[int],
    image_grid_thw: Optional[List[Tuple[int, int, int]]],
    # ...
) -> Tuple[jax.Array, int]:
    # Current implementation is CPU-based (uses list operations)
    # This is intentional as it runs during request preprocessing
    # Verify no JAX operations that could cause sync
```

### Phase 5: Quantization Support (Week 5-6)

**Goal:** Enable FP8/INT8 quantization for production deployment.

#### Task 5.1: Create Qwix Quantization Config

```yaml
# tpu_inference/models/jax/utils/quantization/configs/qwen3_vl_fp8.yaml
global_config:
  dtype: float8_e4m3fn
  weight_block_size: 128
  use_abstract_model: true

layer_configs:
  # Vision encoder (may need higher precision for attention)
  - pattern: "visual.blocks.*.attn.qkv_proj"
    dtype: bfloat16  # Keep BF16 for vision attention stability
  - pattern: "visual.blocks.*.mlp.*"
    dtype: float8_e4m3fn

  # Text decoder
  - pattern: "language_model.layers.*.self_attn.*_proj"
    dtype: float8_e4m3fn
  - pattern: "language_model.layers.*.mlp.*"
    dtype: float8_e4m3fn
```

#### Task 5.2: Verify KV Cache Quantization

```python
# Ensure KV cache quantization works with MRoPE positions
# In Qwen3VLTextAttention.__call__:
if self.kv_cache_quantized_dtype:
    from tpu_inference.layers.common.quantization import quantize_kv
    k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v,
                       self._k_scale, self._v_scale)
```

### Phase 6: Testing & Verification (Week 6-7)

**Goal:** Comprehensive testing and benchmarking.

#### Task 6.1: Numerical Equivalence Tests

| Test | Reference | Tolerance |
|------|-----------|-----------|
| Vision patch embedding | HuggingFace | rtol=1e-3, atol=1e-4 |
| Vision attention block | HuggingFace | rtol=1e-3, atol=1e-4 |
| DeepStack merger output | HuggingFace | rtol=1e-3, atol=1e-4 |
| Interleaved MRoPE | HuggingFace | rtol=1e-5, atol=1e-6 |
| Full forward pass | HuggingFace | rtol=1e-2, atol=1e-3 |

#### Task 6.2: Integration Tests

```python
def test_qwen3_vl_single_image():
    """Test single image captioning."""

def test_qwen3_vl_multi_image():
    """Test multi-image understanding."""

def test_qwen3_vl_video():
    """Test video understanding (short clips)."""

def test_qwen3_vl_deepstack_caching():
    """Test DeepStack cache correctness across requests."""

def test_qwen3_vl_continuous_batching():
    """Test dynamic batching with mixed modalities."""

def test_cosmos_reason2_compatibility():
    """Test NVIDIA Cosmos-Reason2 model loading and inference."""
```

#### Task 6.3: Performance Benchmarks

| Metric | Target (8B on TPU v5e-8) | Notes |
|--------|--------------------------|-------|
| Vision encoding latency | < 50ms (448x448) | Single image |
| Prefill throughput | > 2000 tokens/sec | With images |
| Decode throughput | > 400 tokens/sec | Generation |
| Memory usage | < 20GB per chip | With KV cache |

---

## 7. File Structure

```
tpu_inference/
├── models/
│   ├── common/
│   │   └── model_loader.py            # [MOD] Model registration
│   └── jax/
│       ├── qwen3.py                   # [REF] Base Qwen3 (reuse components)
│       ├── qwen3_vl.py                # [MOD] Main implementation
│       └── utils/
│           ├── multi_modal_utils.py   # Multimodal utilities
│           └── quantization/
│               └── configs/
│                   └── qwen3_vl_fp8.yaml  # [NEW] FP8 config
├── layers/
│   ├── jax/
│   │   └── rope_interface.py          # [MOD] Add interleaved MRoPE
│   └── common/
│       └── attention_interface.py     # Flash attention
├── runner/
│   └── multimodal_manager.py          # [MOD] DeepStack cache support
└── tests/
    └── models/
        └── jax/
            └── test_qwen3_vl.py        # [MOD] Expand test coverage
```

---

## 8. Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| JAX | >= 0.4.30 | TPU support |
| Flax | >= 0.8.0 | NNX API |
| transformers | >= 4.57.0 | Qwen3VL tokenizer/processor |
| vllm | >= 0.8.0 | Core serving infrastructure |
| safetensors | >= 0.4.0 | Weight loading |
| Pillow | >= 10.0.0 | Image preprocessing |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| JIT recompilation for varying image sizes | High | Medium | Bucketed warmup, dynamic padding |
| Memory OOM for 32B model | Medium | High | Tensor parallelism, gradient checkpointing |
| Numerical drift from refactoring | Low | Medium | Comprehensive HuggingFace comparison tests |
| DeepStack cache memory pressure | Medium | Medium | LRU eviction, configurable cache size |
| Video processing bottleneck | High | Low | Temporal batching, sparse attention |

---

## 10. Success Metrics

1. **Correctness**: Token-level match with HuggingFace reference on standard benchmarks
2. **Performance**: 2x throughput improvement over naive implementation
3. **Memory**: Support 8B model on TPU v5e-8 with 896x896 images
4. **Code Quality**: 50% reduction in duplicated code vs current PR
5. **Test Coverage**: >90% coverage for new/modified code
6. **Latency**: <100ms time-to-first-token for single image inference

---

## 11. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Code Refactoring | 2 weeks | Clean, maintainable codebase |
| Phase 2: Vision Optimization | 1 week | TP-enabled vision encoder |
| Phase 3: DeepStack Optimization | 1 week | Efficient DeepStack injection |
| Phase 4: MRoPE Optimization | 0.5 week | Cached frequency computation |
| Phase 5: Quantization | 1.5 weeks | FP8 config and testing |
| Phase 6: Testing & Verification | 1 week | Comprehensive test suite |
| **Total** | **7 weeks** | Production-ready Qwen3-VL |

---

## 12. Open Questions

1. **DeepStack injection frequency**: Should DeepStack features be injected at every early layer or only at specific layers matching the extraction indices?

2. **Vision encoder batch size**: What's the optimal batch size for vision prefill when processing multiple images in a single request?

3. **Interleaved MRoPE upstream**: Should the interleaved M-RoPE implementation be added to `rope_interface.py` or kept model-specific?

4. **Dynamic resolution handling**: Should we pad to power-of-2 sizes or use exact bucket sizes for vision encoder JIT?

5. **Cosmos-Reason2 testing**: Do we need separate tests for Cosmos models or is config-level verification sufficient?

---

## 13. References

1. **Qwen3-VL Technical Report**: [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
2. **Qwen3-VL HuggingFace**: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
3. **DeepStack Paper**: [arXiv:2406.04334](https://arxiv.org/abs/2406.04334) (NeurIPS 2024)
4. **SigLIP-2 Paper**: [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)
5. **NVIDIA Cosmos-Reason2**: [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
6. **SGLang-JAX RFC**: [sgl-project/sglang-jax#693](https://github.com/sgl-project/sglang-jax/issues/693)
7. **tpu-inference JAX Model Development**: [docs/developer_guides/jax_model_development.md](./docs/developer_guides/jax_model_development.md)

---

## 14. Appendix: Key Code Snippets

### A. Interleaved MRoPE Implementation

```python
def apply_interleaved_mrope(
    freqs: jax.Array,
    mrope_section: List[int] = [24, 20, 20],
) -> jax.Array:
    """
    Apply interleaved M-RoPE frequency arrangement.

    Args:
        freqs: MRoPE frequencies from T, H, W positions. Shape: (3, bs, seq, dim/2)
        mrope_section: How frequencies are partitioned [t_dim, h_dim, w_dim]

    Returns:
        Interleaved frequencies. Shape: (bs, seq, dim/2)
    """
    t, h, w = mrope_section  # [24, 20, 20]

    # Start with T frequencies as base
    result = freqs[0].copy()

    # Interleave H frequencies at positions [1, 4, 7, 10, ...]
    h_indices = jnp.arange(1, h * 3, 3)
    result = result.at[..., h_indices].set(freqs[1][..., h_indices])

    # Interleave W frequencies at positions [2, 5, 8, 11, ...]
    w_indices = jnp.arange(2, w * 3, 3)
    result = result.at[..., w_indices].set(freqs[2][..., w_indices])

    return result
```

### B. DeepStack Feature Injection

```python
def _inject_visual_features(
    hidden_states: jax.Array,      # (seq_len, hidden_size)
    visual_pos_mask: jax.Array,    # (seq_len,) boolean
    visual_embeds: jax.Array,      # (num_visual_tokens, hidden_size)
) -> jax.Array:
    """Add DeepStack visual features at masked positions."""
    # Get indices where visual tokens are located
    indices = jnp.nonzero(visual_pos_mask, size=visual_embeds.shape[0])[0]

    # Add (not replace) visual features to hidden states
    return hidden_states.at[indices].add(visual_embeds)
```

### C. Bilinear Position Embedding Interpolation

```python
def fast_pos_embed_interpolate(
    pos_embed: nnx.Embed,
    grid_thw: Tuple[Tuple[int, int, int], ...],
    base_grid_h: int = 48,
    base_grid_w: int = 48,
) -> jax.Array:
    """Bilinear interpolation for learned positional embeddings."""
    for t, h, w in grid_thw:
        # Create linearly spaced indices
        h_idxs = jnp.linspace(0, base_grid_h - 1, h)
        w_idxs = jnp.linspace(0, base_grid_w - 1, w)

        # Compute floor/ceil for bilinear interpolation
        h_floor, w_floor = h_idxs.astype(jnp.int32), w_idxs.astype(jnp.int32)
        h_ceil = (h_floor + 1).clip(max=base_grid_h - 1)
        w_ceil = (w_floor + 1).clip(max=base_grid_w - 1)

        # Compute interpolation weights
        dh, dw = h_idxs - h_floor, w_idxs - w_floor

        # Gather 4 corners and apply weighted sum
        # ... (see full implementation in qwen3_vl.py:1250-1347)
```

---

*This RFC is open for community feedback and will be updated based on discussion.*
