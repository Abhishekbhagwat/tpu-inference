import functools
import math
from functools import partial
from typing import List, Literal, NamedTuple, Optional, Tuple, TypedDict, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import (
    attention,
    sharded_flash_attention,
    # TODO: Text attention should use ragged pagedattention
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
)
from tpu_inference.models.jax.utils.weight_utils import (
    get_default_maps,
    load_hf_weights,
)

# TODO: Potential code consolidation with qwen3.py
# The following Qwen3-VL components are architecturally similar to Qwen3:
# - Qwen3VLTextMLP ~ Qwen3MLP (identical SwiGLU structure)
# - Qwen3VLTextDecoderLayer ~ Qwen3DecoderLayer (same pre-norm transformer block)
# - RMSNorm: Now uses nnx.RMSNorm directly (Phase 1.1 complete)
# - RoPE: Now uses shared rope_interface.apply_rope with interleaved_mrope (Phase 1.2 complete)
# The main differences are:
# - Interleaved MRoPE (3D positions) vs standard RoPE
# - DeepStack injection in decoder layers
# - Q/K normalization in attention (qk_norm)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

DEFAULT_BLOCK_K_MAJOR = 128


# =============================================================================
# Phase 4: MRoPE Frequency Caching
# =============================================================================


@functools.lru_cache(maxsize=32)
def _get_cached_inv_freq(dim: int, rope_theta: float) -> Tuple[float, ...]:
    """Cache inverse frequencies for common configurations.

    Returns a tuple (hashable) that can be converted to jax.Array when needed.
    """
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    return tuple(inv_freq.tolist())


def get_inv_freq(dim: int, rope_theta: float) -> jax.Array:
    """Get inverse frequencies, using cache for common configurations."""
    cached = _get_cached_inv_freq(dim, rope_theta)
    return jnp.array(cached, dtype=jnp.float32)


class _Qwen3VLConfigAdapter:
    def __init__(self, config):
        self._config = config
        self._text_config = getattr(config, "text_config", None)

    def __getattr__(self, name):
        try:
            return getattr(self._config, name)
        except AttributeError:
            pass
        if self._text_config is not None:
            try:
                return getattr(self._text_config, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self._config).__name__}' object has no attribute '{name}'"
        )


class _ModelConfigAdapter:
    def __init__(self, model_config):
        self._model_config = model_config
        self._hf_config_adapter = _Qwen3VLConfigAdapter(model_config.hf_config)

    @property
    def hf_config(self):
        return self._hf_config_adapter

    def __getattr__(self, name):
        return getattr(self._model_config, name)


def _infer_pos_embed_grid_hw(num_position_embeddings: int) -> Tuple[int, int]:
    """Infer a (grid_h, grid_w) pair from a flattened 2D embedding table."""
    root = int(math.sqrt(num_position_embeddings))
    for grid_h in range(root, 0, -1):
        if num_position_embeddings % grid_h == 0:
            return grid_h, num_position_embeddings // grid_h
    return num_position_embeddings, 1


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers). Only tokens with the same id can attend to each other.

    Attributes:
        q: segment ids along the Q sequence.
        kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


class Qwen3VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: jax.Array
    image_grid_thw: Tuple[Tuple[int, int, int], ...]


class Qwen3VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: jax.Array
    image_grid_thw: Tuple[Tuple[int, int, int], ...]


Qwen3VLImageInputs = Union[Qwen3VLImagePixelInputs,
                            Qwen3VLImageEmbeddingInputs]


def generate_segment_ids_from_grid_thw(
    grid_thw: Tuple[Tuple[int, int, int], ...],
) -> jax.Array:
    """Generate segment IDs from grid dimensions for variable-length attention.

    Each frame gets a unique segment ID (starting from 1). For images (t=1),
    this means one segment ID per image. For videos (t>1), each temporal frame
    gets its own segment ID, preventing cross-frame attention. Temporal order
    is preserved through separate tokens that annotate frame positions.

    Args:
        grid_thw: Tuple of (T, H, W) for each image/video

    Returns:
        segment_ids: (total_tokens,) array with segment IDs per frame
    """
    segments = []
    seg_id = 1 # start for 1 to make padding 0

    for (t, h, w) in grid_thw:
        frame_size = h * w
        for _ in range(t):
            segments.append(jnp.full((frame_size,), seg_id, dtype=jnp.int32))
            seg_id += 1

    return jnp.concatenate(segments, axis=0) if segments else jnp.zeros((0,), dtype=jnp.int32)


def pad_segment_ids_for_attention(
    segment_ids: jax.Array,
    padded_seq_len: int,
) -> SegmentIds:
    """Pad segment IDs and format for flash attention.

    Args:
        segment_ids: (seq_len,) segment IDs for valid tokens
        padded_seq_len: Padded sequence length (multiple of block size)

    Returns:
        SegmentIds for flash attention with padding tokens set to 0
    """
    seq_len = segment_ids.shape[0]
    padded_ids = jnp.zeros(padded_seq_len, dtype=jnp.int32)
    padded_ids = padded_ids.at[:seq_len].set(segment_ids)
    padded_ids = padded_ids.reshape(1, -1)
    return SegmentIds(q=padded_ids, kv=padded_ids)


def compute_vision_counts_per_sequence(
    input_ids: jax.Array,
    attention_mask: jax.Array,
    image_token_id: int,
    video_token_id: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute the number of images and videos per sequence in a batch.

    This is a preprocessing function for MRoPE computation. It does not JIT.
    This is not used currently, but would be needed for sequence batches.

    Args:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders

    Returns:
        num_images_per_sequence: (batch_size,)
        num_videos_per_sequence: (batch_size,)
    """
    # Mask invalid positions
    masked_ids = jnp.where(attention_mask == 1, input_ids, -1)

    # Count per sequence
    num_images = jnp.sum(masked_ids == image_token_id, axis=1)
    num_videos = jnp.sum(masked_ids == video_token_id, axis=1)

    return num_images, num_videos


def get_mrope_input_positions(
    input_tokens: List[int],
    image_grid_thw: Optional[List[Tuple[int, int, int]]],
    video_grid_thw: Optional[List[Tuple[int, int, int]]],
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> Tuple[jax.Array, int]:
    """Compute MRoPE 3D position IDs for a single sequence.

    This function computes position IDs for text and vision tokens.
    Text tokens get sequential positions, while vision tokens get
    3D positions based on their temporal, height, and width coordinates.

    Args:
        input_tokens: List of token IDs for the sequence
        image_grid_thw: List of (T, H, W) tuples for each image
        video_grid_thw: List of (T, H, W) tuples for each video
        image_token_id: Token ID for image placeholders
        video_token_id: Token ID for video placeholders
        vision_start_token_id: Token ID marking start of vision tokens
        spatial_merge_size: Spatial merge size from vision config

    Returns:
        llm_positions: (3, seq_len) position IDs for [T, H, W]
        mrope_position_delta: Delta for rope calculation
    """
    # Use the provided grid_thw lengths as authoritative counts.
    # Note: For Qwen3VL, each temporal frame may have its own vision_start marker while
    # the grid_thw represents the entire video as a single item with t>1.
    image_nums = len(image_grid_thw) if image_grid_thw else 0
    video_nums = len(video_grid_thw) if video_grid_thw else 0
    llm_pos_ids_list = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index, video_index = 0, 0

    for _ in range(image_nums + video_nums):
        if remain_images > 0:
            try:
                ed_image = input_tokens.index(image_token_id, st)
            except ValueError:
                ed_image = len(input_tokens) + 1
        else:
            ed_image = len(input_tokens) + 1

        if remain_videos > 0:
            try:
                ed_video = input_tokens.index(video_token_id, st)
            except ValueError:
                ed_video = len(input_tokens) + 1
        else:
            ed_video = len(input_tokens) + 1

        # Determine which modality comes first in the token sequence
        # Break early if no more tokens found (extra grids are ignored)
        if ed_image > len(input_tokens) and ed_video > len(input_tokens):
            break

        if ed_image <= ed_video and remain_images > 0:
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            ed = ed_image
        elif remain_videos > 0:
            t, h, w = video_grid_thw[video_index]
            video_index += 1
            remain_videos -= 1
            ed = ed_video
        else:
            # No more grids to process, break the loop
            break

        # t would always be 1
        llm_grid_t = t
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        text_len = ed - st

        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx
        )

        # t_index alaways zero
        num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w

        t_index = jnp.broadcast_to(
            jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
            (llm_grid_t, llm_grid_h * llm_grid_w),
        ).flatten()  # llm_grid_t=1 then [0, 0, 0, ...]

        h_index = jnp.broadcast_to(
            jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).flatten()

        w_index = jnp.broadcast_to(
            jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).flatten()

        llm_pos_ids_list.append(
            jnp.stack([t_index, h_index, w_index]) + text_len + st_idx
        )

        st = ed + num_vision_tokens

    # Trailing text
    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx
        )

    llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)

    return llm_positions, mrope_position_delta


def apply_interleaved_mrope(
    freqs: jax.Array,
    mrope_section: List[int] = [24, 20, 20],
) -> jax.Array:
    """

    :param freqs: MRoPE frequency derived from T, H, W values. (3, bs, seq, 64)
    :param mrope_section: How MRoPE would be placed.
    :return: A final interleaved MRoPE frequency. (bs, seq, 64)
    """
    t, h, w = mrope_section  # 24, 20, 20

    result = freqs[0].copy()

    h_indices = jnp.arange(1, h * 3, 3)
    result = result.at[..., h_indices].set(freqs[1][..., h_indices])

    w_indices = jnp.arange(2, w * 3, 3)  # [2, 5, 8, ..., 59]
    result = result.at[..., w_indices].set(freqs[2][..., w_indices])

    return result


class Qwen3VLTextAttention(nnx.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size //
                                         self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        # Ensure rope_scaling has mrope_section for interleaved M-RoPE
        if self.rope_scaling is None:
            self.rope_scaling = {"mrope_section": [24, 20, 20]}
        elif "mrope_section" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            self.rope_scaling["mrope_section"] = [24, 20, 20]

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rngs,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rngs,
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        q = self.q_proj(hidden_states)
        q = self.q_norm(q)

        k = self.k_proj(hidden_states)
        k = self.k_norm(k)

        # Ensure positions are treated as Qwen3-VL does: always 3D positions for MRoPE
        pos = md.input_positions
        if pos.ndim == 1:
            # Expand vanilla positions to (3, T) for text-only
            pos = jnp.broadcast_to(pos[None, :], (3, pos.shape[0]))

        # Apply interleaved M-RoPE using the shared rope_interface
        q = apply_rope(
            q, pos, self.head_dim_original,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            interleaved_mrope=True,
        )
        k = apply_rope(
            k, pos, self.head_dim_original,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            interleaved_mrope=True,
        )

        v = self.v_proj(hidden_states)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = utils.quantize_kv(k, v, self.kv_cache_quantized_dtype,
                                     k_scale, v_scale)

        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Qwen3VLTextMLP(nnx.Module):
    """SwiGLU MLP for Qwen3-VL text decoder.

    Note: This is architecturally identical to Qwen3MLP in qwen3.py.
    Future refactoring could consolidate these implementations.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs,
        hidden_act: str = "silu",
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            intermediate_size, hidden_size, use_bias=False, param_dtype=dtype, rngs=rngs
        )

        if hidden_act == "silu":
            self.act_fn = jax.nn.silu
        else:
            raise NotImplementedError(f"Activation function '{hidden_act}' not implemented")

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        down_proj = self.down_proj(gate_output * up_output)
        return down_proj


class Qwen3VLTextDecoderLayer(nnx.Module):
    def __init__(
        self,
        config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
    ):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.self_attn = Qwen3VLTextAttention(
            config=config,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
        )
        self.mlp = Qwen3VLTextMLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            rngs=rngs,
            hidden_act=getattr(config, "hidden_act", "silu"),
            dtype=dtype,
        )
        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        kv_cache, hidden_states = self.self_attn(
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            attention_metadata=attention_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return kv_cache, hidden_states


def apply_rotary_pos_emb_vision(
    x: jax.Array, rotary_pos_emb: jax.Array
) -> jax.Array:

    _, _, _, H = x.shape
    half_dim = H // 2

    # [B, T, N, H//2]
    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    # [T, H//2]
    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    # [1, T, 1, H//2]
    cos_emb = cos_emb[None, :, None, :]
    sin_emb = sin_emb[None, :, None, :]

    # [B, T, N, H//2]
    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    # [B, T, N, H]
    x_rotated = jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)

    return x_rotated


class Qwen3VLVisionRotaryEmbedding(nnx.Module):
    """Rotary position embedding for vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (
            self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)



class Qwen3VLVisionPatchEmbed(nnx.Module):
    """3D Patch Embedding for video/image input using 3D convolution."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, None, None, None, "model")
            ),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply 3D patch embedding.

        Args:
            x: Input tensor of shape (num_patches, C * T * H * W)

        Returns:
            Embedded patches of shape (num_patches, hidden_size)
        """
        L, dim = x.shape
        C = dim // (
            self.temporal_patch_size * self.patch_size * self.patch_size
        )
        # Reshape to (L, T, H, W, C) for Conv3D with channels_last
        x = x.reshape(
            L, C, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # L, T, H, W, C
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x



class Qwen3VLVisionMLP(nnx.Module):
    """SwiGLU-style MLP for vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.fc1 = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        return self.fc2(x)



class Qwen3VLVisionAttention(nnx.Module):
    """Full attention for vision encoder using sharded flash attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads, sharding_size)
        self.head_dim = hidden_size // num_heads  # Original head dim

        self.mesh = mesh

        # QKV projection
        self.qkv_proj = nnx.Linear(
            hidden_size,
            3 * hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )

        # Output projection
        self.proj = nnx.Linear(
            hidden_size,
            hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

        # Qwen3VL's Vision Transformer uses full bidirectional attention.
        # Note: vmem_limit increased to 256MB for larger image sequences
        self.flash_attention = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            vmem_limit_bytes=256 * 1024 * 1024,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> jax.Array:
        """Apply vision attention with variable-length segment masking.

        Args:
            x: Input tensor of shape (T, B, D)
            rotary_pos_emb: Rotary position embeddings
            segment_ids: Segment IDs (T,) for variable-length attention masking

        Returns:
            Output tensor of shape (T, B, D)
        """
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"

        qkv = self.qkv_proj(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Head-last reshape
        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        # Transpose to [B, T, N, H]
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Transpose to [B, N, T, H] for flash attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Pad sequence length to multiple of block size
        block_k_major = DEFAULT_BLOCK_K_MAJOR
        T_attn = q.shape[2]
        padded_T = (T_attn + block_k_major - 1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, "constant")
        k = jnp.pad(k, pad_width, "constant")
        v = jnp.pad(v, pad_width, "constant")

        # Pad segment IDs for attention (padding tokens get segment_id=0)
        padded_segment_ids = pad_segment_ids_for_attention(segment_ids, padded_T)

        output = self.flash_attention(q, k, v, padded_segment_ids)

        # Unpad and reshape: [B, N, T, H] -> [T, B, N, H] -> [T, B, D]
        output = output[:, :, :T_attn, :]
        output = jnp.transpose(output, (2, 0, 1, 3))
        output = output.reshape(T, B, D)

        output = self.proj(output)

        return output



class Qwen3VLVisionBlock(nnx.Module):
    """Transformer block for vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        mesh: Mesh,
        norm_eps: float = 1e-6,
    ):
        self.norm1 = nnx.LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )
        self.attn = Qwen3VLVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.norm2 = nnx.LayerNorm(
            hidden_size,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )
        self.mlp = Qwen3VLVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, segment_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(nnx.Module):
    """Merge spatial patches and project to language model dimension.

    This module supports both:
    - Final merger: norm before reshape (use_postshuffle_norm=False)
    - DeepStack merger: reshape before norm (use_postshuffle_norm=True)
    """

    def __init__(
        self,
        d_model: int, # in
        context_dim: int, # out
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        # Normalization dimension depends on use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nnx.LayerNorm(
            norm_dim,
            epsilon=norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

        self.linear_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            bias_init=nnx.with_partitioning(init_fn, ("model",)),
            rngs=rngs,
        )
        self.linear_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply patch merging.

        If use_postshuffle_norm is True, reshaping would happen first.
        """
        if self.use_postshuffle_norm:
            # DeepStack: reshape first, then norm
            x = x.reshape(-1, self.hidden_size)
            x = self.norm(x)
        else:
            # Final merger: norm first, then reshape
            x = self.norm(x)
            x = x.reshape(-1, self.hidden_size)

        x = self.linear_fc1(x)
        x = nnx.gelu(x) # make this configurable?
        x = self.linear_fc2(x)
        return x


class Qwen3VLVisionTransformer(nnx.Module):
    """Vision Transformer for Qwen3VL with DeepStack support.

    Vision encoder sharding strategy:
    - patch_embed.proj.kernel: P(None, None, None, None, "model")
    - blocks.*.attn.qkv_proj.kernel: P(None, "model")
    - blocks.*.attn.proj.kernel: P("model", None)
    - blocks.*.mlp.fc1.kernel: P(None, "model")
    - blocks.*.mlp.fc2.kernel: P("model", None)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rngs: nnx.Rngs,
        mesh: Mesh,
        norm_eps: float = 1e-6,
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vision_config = hf_config.vision_config
        dtype = model_config.dtype

        self.config = vision_config
        self.dtype = dtype

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        # analyze full attn block indexes' purpose by visiting flash attention impl.
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            rngs=rngs,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
            dtype=dtype,
        )

        # Learned PE, 48 x 48 H W
        num_position_embeddings = getattr(
            vision_config, "num_position_embeddings", 2304
        )
        self.pos_embed = nnx.Embed(
            num_embeddings=num_position_embeddings,
            features=self.hidden_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rngs,
        )
        # NOTE: Learned PE can be rectangular (H != W); infer a base (H, W) grid.
        image_size = getattr(vision_config, "image_size", None)
        pos_embed_grid_h = pos_embed_grid_w = None
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            h0, w0 = int(image_size[0]), int(image_size[1])
            if h0 > 0 and w0 > 0 and h0 * w0 == num_position_embeddings:
                pos_embed_grid_h, pos_embed_grid_w = h0, w0
            else:
                h1, w1 = h0 // patch_size, w0 // patch_size
                if h1 > 0 and w1 > 0 and h1 * w1 == num_position_embeddings:
                    pos_embed_grid_h, pos_embed_grid_w = h1, w1
        elif isinstance(image_size, int):
            s0 = int(image_size)
            if s0 > 0 and s0 * s0 == num_position_embeddings:
                pos_embed_grid_h = pos_embed_grid_w = s0
            else:
                s1 = s0 // patch_size
                if s1 > 0 and s1 * s1 == num_position_embeddings:
                    pos_embed_grid_h = pos_embed_grid_w = s1
        if pos_embed_grid_h is None:
            if image_size is not None:
                logger.warning(
                    "Couldn't derive pos_embed grid from vision_config.image_size="
                    f"{image_size!r}; inferring from num_position_embeddings={num_position_embeddings}."
                )
            pos_embed_grid_h, pos_embed_grid_w = _infer_pos_embed_grid_hw(
                num_position_embeddings
            )
        self.pos_embed_grid_h = pos_embed_grid_h
        self.pos_embed_grid_w = pos_embed_grid_w

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        intermediate_size = vision_config.intermediate_size
        self.blocks = [
            Qwen3VLVisionBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
                norm_eps=norm_eps,
            )
            for _ in range(vision_config.depth)
        ]

        # Final merger settings
        # Qwen3VLConfig uses text_config for text model hidden_size
        text_config = getattr(hf_config, "text_config", hf_config)
        out_hidden_size = getattr(vision_config, "out_hidden_size", text_config.hidden_size)

        self.merger = Qwen3VLVisionPatchMerger(
            d_model=out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            use_postshuffle_norm=False,
            norm_eps=norm_eps,
        )

        # DeepStack configuration
        self.deepstack_visual_indexes = getattr(
            vision_config, "deepstack_visual_indexes", [8, 16, 24]
        )
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(
                d_model=out_hidden_size,
                context_dim=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
                dtype=dtype,
                rngs=rngs,
                use_postshuffle_norm=True,  # DeepStack uses postshuffle norm
                norm_eps=norm_eps,
            )
            for _ in range(len(self.deepstack_visual_indexes))
        ]

        additional_config = getattr(vllm_config, "additional_config", None) or {}
        self.enable_dynamic_image_sizes = additional_config.get(
            "enable_dynamic_image_sizes", False
        )

    def rotary_pos_emb_thw(
        self, t: int, h: int, w: int
    ) -> jax.Array:
        """Compute rotary position embeddings for a grid of patches.

        Args:
            t: Temporal dimension
            h: Height dimension (in patches)
            w: Width dimension (in patches)

        Returns:
            Rotary embeddings of shape (t * merged_h * merged_w, spatial_merge_unit, dim)
        """
        merge_size = self.spatial_merge_size

        hpos_ids, wpos_ids = jnp.indices((h, w))
        hpos_ids = (
            hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids = (
            wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            .transpose(0, 2, 1, 3)
            .flatten()
        )
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)

        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

        return rotary_pos_emb

    def fast_pos_embed_interpolate(
        self, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> jax.Array:
        """Bilinear interpolation for learned positional embeddings.

        Args:
            grid_thw: Tuple of (T, H, W) for each image/video

        Returns:
            Position embeddings of shape (total_patches, hidden_size)
        """
        merge_size = self.spatial_merge_size

        idx_list = [jnp.empty((0,), dtype=jnp.int32) for _ in range(4)]
        weight_list = [jnp.empty((0,), dtype=jnp.float32) for _ in range(4)]

        grid_ts = [g[0] for g in grid_thw]
        grid_hs = [g[1] for g in grid_thw]
        grid_ws = [g[2] for g in grid_thw]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # Create linearly spaced indices for interpolation
            h_idxs = jnp.linspace(0, self.pos_embed_grid_h - 1, h)
            w_idxs = jnp.linspace(0, self.pos_embed_grid_w - 1, w)

            # Floor and ceil indices for bilinear interpolation
            h_idxs_floor = h_idxs.astype(jnp.int32)
            w_idxs_floor = w_idxs.astype(jnp.int32)
            h_idxs_ceil = (h_idxs.astype(jnp.int32) + 1).clip(
                max=self.pos_embed_grid_h - 1
            )
            w_idxs_ceil = (w_idxs.astype(jnp.int32) + 1).clip(
                max=self.pos_embed_grid_w - 1
            )

            # Interpolation weights
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            # Base indices for 2D grid (flattened)
            base_h = h_idxs_floor * self.pos_embed_grid_w
            base_h_ceil = h_idxs_ceil * self.pos_embed_grid_w

            # Compute 4 corner indices for bilinear interpolation
            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).reshape(-1),  # top-left
                (base_h[:, None] + w_idxs_ceil[None, :]).reshape(-1),   # top-right
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).reshape(-1),  # bottom-left
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).reshape(-1),   # bottom-right
            ]

            # Compute weights for bilinear interpolation
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),  # top-left
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),        # top-right
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),        # bottom-left
                (dh[:, None] * dw[None, :]).reshape(-1),              # bottom-right
            ]

            for i in range(4):
                idx_list[i] = jnp.concatenate([idx_list[i], indices[i]], axis=0)
                weight_list[i] = jnp.concatenate([weight_list[i], weights[i]], axis=0)

        # Convert to JAX arrays
        idx_tensor = jnp.stack(idx_list, axis=0)  # int32, shape (4, N)
        weight_tensor = jnp.stack(weight_list, axis=0)  # float32, shape (4, N)

        # Lookup embeddings and apply bilinear interpolation
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split embeddings for each image/video
        split_sizes = [h * w for h, w in zip(grid_hs, grid_ws)]

        patch_pos_embeds_list = []
        offset = 0
        for size in split_sizes:
            patch_pos_embeds_list.append(patch_pos_embeds[offset: offset + size])
            offset += size

        # Rearrange embeddings to match patch ordering (with merge blocks)
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws):
            # Repeat for temporal dimension
            pos_embed = jnp.tile(pos_embed, (t, 1))

            h_merged = h // merge_size
            w_merged = w // merge_size
            pos_embed = pos_embed.reshape(
                t, h_merged, merge_size, w_merged, merge_size, -1
            )
            pos_embed = jnp.transpose(pos_embed, (0, 1, 3, 2, 4, 5))
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])

            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = jnp.concatenate(patch_pos_embeds_permute, axis=0)
        return patch_pos_embeds

    def encode(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> Tuple[jax.Array, List[jax.Array]]:
        hidden_states = self.patch_embed(x)

        # Reshape for merge block ordering
        seq_len = x.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )

        # Add batch dimension: (seq, merge_unit, dim) -> (seq * merge_unit, 1, dim)
        hidden_states = hidden_states.reshape(seq_len, 1, -1)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                segment_ids=segment_ids,
            )

            # Collect DeepStack features at specified layers
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                # Squeeze batch dim for merger: (seq, 1, dim) -> (seq, dim)
                deepstack_feat = self.deepstack_merger_list[idx](
                    hidden_states.squeeze(1)
                )
                deepstack_features.append(deepstack_feat)

        # Squeeze batch dim and apply final merger
        hidden_states = self.merger(hidden_states.squeeze(1))

        return hidden_states, deepstack_features

    @partial(jax.jit, static_argnames=("grid_thw",))
    def encode_jit(
        self, x: jax.Array, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """JIT-compiled encoding with static grid dimensions."""
        # Learned PE interpolate
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        hidden_states = self.patch_embed(x)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb_list = []
        for t, h, w in grid_thw:
            rotary_pos_emb_list.append(self.rotary_pos_emb_thw(t, h, w))
        rotary_pos_emb = jnp.concatenate(rotary_pos_emb_list, axis=0)
        rotary_pos_emb = rotary_pos_emb.reshape(-1, rotary_pos_emb.shape[-1])

        # pre-merge patch segment ids
        segment_ids = generate_segment_ids_from_grid_thw(
            grid_thw
        )
        assert segment_ids.shape[0] == hidden_states.shape[0], (
            "segment_ids must match the patch sequence length. "
            f"Got {segment_ids.shape[0]=} vs {hidden_states.shape[0]=}.")

        # Reshape for transformer
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states.reshape(seq_len, 1, -1)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                segment_ids=segment_ids,
            )

            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feat = self.deepstack_merger_list[idx](
                    hidden_states.squeeze(1)
                )
                deepstack_features.append(deepstack_feat)

        hidden_states = self.merger(hidden_states.squeeze(1))

        return hidden_states, deepstack_features

    def __call__(
        self, x: jax.Array, grid_thw: Tuple[Tuple[int, int, int], ...]
    ) -> Tuple[jax.Array, List[jax.Array]]:
        """Forward pass for vision encoder.

        Args:
            x: Pixel values of shape (num_patches, C * T * ps * ps)
            grid_thw: Tuple of (T, H, W) for each image/video

        Returns:
            hidden_states: Final merged features (total_tokens, out_hidden_size)
            deepstack_features: List of intermediate features for DeepStack
        """
        return self.encode_jit(x, grid_thw)



class Qwen3VLModel(nnx.Module):
    """Text model for Qwen3VL with MRoPE and DeepStack support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = text_config.rms_norm_eps
        hidden_size = text_config.hidden_size

        self.hidden_size = hidden_size

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )

        # Decoder layers with MRoPE support and KV cache attention
        self.layers = [
            Qwen3VLTextDecoderLayer(
                config=text_config,
                dtype=dtype,
                rngs=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            )
            for _ in range(text_config.num_hidden_layers)
        ]

        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            rngs=rng,
        )

        if hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )

        self.tie_word_embeddings = hf_config.tie_word_embeddings

    def _inject_visual_features(
        self,
        hidden_states: jax.Array,
        visual_pos_mask: jax.Array,
        visual_embeds: jax.Array,
    ) -> jax.Array:
        """Add DeepStack visual features at masked positions.

        DeepStack is a multi-layer visual feature injection technique that injects
        visual embeddings into the language model at multiple decoder layers, not
        just at the input embedding layer. This enables richer cross-modal fusion
        by allowing visual information to influence intermediate representations.

        This method is called after each early decoder layer (controlled by the
        number of DeepStack layers configured in the vision encoder). The visual
        features from each DeepStack layer are added to the hidden states at
        positions corresponding to visual tokens.

        Implementation Notes:
            This optimized version uses direct indexing with jnp.nonzero + scatter-add
            instead of the previous cumsum + gather approach. The scatter-based method
            is more efficient because:
            1. It avoids creating padded arrays with dummy rows
            2. It uses fewer intermediate tensors
            3. The .at[].add() operation is optimized for sparse updates

        Args:
            hidden_states: Language model hidden states with shape
                (seq_len, hidden_size) for unbatched or
                (batch, seq_len, hidden_size) for batched input.
            visual_pos_mask: Boolean mask indicating positions where visual tokens
                are located. Shape matches hidden_states without the last dimension.
            visual_embeds: Visual features to inject from the current DeepStack layer.
                Shape: (num_visual_tokens, hidden_size). The number of visual tokens
                must match the number of True values in visual_pos_mask.

        Returns:
            Updated hidden_states with visual features added at masked positions.
            Same shape as input hidden_states.
        """
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_mask = visual_pos_mask.reshape(-1).astype(jnp.bool_)

        # Get indices where visual tokens should be injected
        # Use size parameter to make it JIT-compatible
        num_visual = visual_embeds.shape[0]

        # Use -1 as fill_value to identify padding indices
        indices = jnp.nonzero(flat_mask, size=num_visual, fill_value=-1)[0]

        # Ensure visual_embeds has matching dtype
        visual_embeds = visual_embeds.astype(flat_hidden.dtype)

        # Create mask for valid indices (not -1 padding)
        valid_mask = (indices >= 0).astype(flat_hidden.dtype)[:, None]

        # Zero out embeddings for invalid/padded indices
        masked_embeds = visual_embeds * valid_mask

        # Clamp indices to valid range for scatter (padding indices become 0)
        safe_indices = jnp.maximum(indices, 0)

        # Use scatter-add for efficient update
        updated = flat_hidden.at[safe_indices].add(masked_embeds)

        return updated.reshape(hidden_states.shape)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        visual_pos_mask: Optional[jax.Array] = None,
        deepstack_visual_embeds: Optional[List[jax.Array]] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        """Forward pass with KV cache, MRoPE, and DeepStack support."""
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache=kv_cache,
                hidden_states=x,
                attention_metadata=attention_metadata,
            )
            kv_caches[i] = kv_cache

            if (
                deepstack_visual_embeds is not None
                and i < len(deepstack_visual_embeds)
                and visual_pos_mask is not None
            ):
                x = self._inject_visual_features(
                    x, visual_pos_mask, deepstack_visual_embeds[i]
                )

        x = self.norm(x)

        return kv_caches, x

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.lm_head.value)
        return logits



class Qwen3VLForConditionalGeneration(nnx.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ):
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        config = vllm_config.model_config.hf_config
        self.config = config
        text_config = getattr(config, "text_config", config)

        self.visual = Qwen3VLVisionTransformer(
            vllm_config=vllm_config,
            rngs=self.rng,
            mesh=mesh,
            norm_eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

        self.language_model = Qwen3VLModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    def get_input_embeddings(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array],
    ) -> jax.Array:
        """Get input embeddings with multimodal content merged.

        Args:
            input_ids: Input token IDs
            multimodal_embeddings: Flattened multimodal embeddings

        Returns:
            Input embeddings with multimodal content merged
        """
        inputs_embeds = self.language_model.embed(input_ids)

        if multimodal_embeddings is not None and multimodal_embeddings.shape[0] != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.image_token_id, self.video_token_id],
            )

        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        multimodal_embeddings: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute input embeddings and merge multimodal embeddings if present."""
        return self.get_input_embeddings(input_ids, multimodal_embeddings)

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> jax.Array:
        if isinstance(mm_input, list):
            arrays_to_concat = [jnp.asarray(item) for item in mm_input]
            return jnp.concatenate(arrays_to_concat, axis=0)

        if hasattr(mm_input, 'ndim'):
            array_input = jnp.asarray(mm_input)
            if array_input.ndim == 2:
                return array_input
            if array_input.ndim == 3:
                return array_input.reshape(-1, array_input.shape[-1])

        raise ValueError(f"Incorrect type of {name}. "
                         f"Got type: {type(mm_input)}")

    def _normalize_grid_thw(
            self, grid_thw: object) -> Tuple[Tuple[int, int, int], ...]:
        if grid_thw is None:
            return ()
        if isinstance(grid_thw, (list, tuple)):
            if len(grid_thw) == 0:
                return ()
            if len(grid_thw) == 3 and all(
                    isinstance(v, (int, np.integer)) for v in grid_thw):
                return (tuple(int(v) for v in grid_thw), )
            if isinstance(grid_thw[0], (list, tuple)):
                return tuple(tuple(int(v) for v in row) for row in grid_thw)
        if hasattr(grid_thw, 'ndim'):
            array_input = np.asarray(grid_thw)
            if array_input.size == 0:
                return ()
            if array_input.ndim == 1 and array_input.shape[0] == 3:
                return (tuple(int(v) for v in array_input.tolist()), )
            if array_input.ndim == 2 and array_input.shape[1] == 3:
                return tuple(
                    tuple(int(v) for v in row)
                    for row in array_input.tolist())
        raise ValueError("Incorrect type of grid_thw. "
                         f"Got type: {type(grid_thw)}")

    def _parse_and_validate_image_input(
            self, image_grid_thw: Tuple[Tuple[int, int, int], ...],
            **kwargs: object) -> Optional[Qwen3VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            pixel_values = kwargs.pop("pixel_values_videos", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "pixel values")

            if not isinstance(pixel_values, jax.Array):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen3VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw)

        # NOTE: Not supporting image embeddings precomputed. Matches Qwen2.5VL.
        # if image_embeds is not None:
        #     image_embeds = self._validate_and_reshape_mm_tensor(
        #         image_embeds, "image embeds")
        #     if not isinstance(image_embeds, jax.Array):
        #         raise ValueError("Incorrect type of image embeddings. "
        #                          f"Got type: {type(image_embeds)}")
        #     return Qwen3VLImageEmbeddingInputs(
        #         type="image_embeds",
        #         image_embeds=image_embeds,
        #         image_grid_thw=image_grid_thw)

    def _parse_and_validate_multimodal_inputs(self,
                                              image_grid_thw: Tuple[Tuple[int,
                                                                          int,
                                                                          int],
                                                                    ...],
                                              **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "pixel_values_videos",
                             "image_embeds"
                             ) and "image" not in mm_input_by_modality:
                mm_input_by_modality[
                    "image"] = self._parse_and_validate_image_input(
                        image_grid_thw, **kwargs)
        return mm_input_by_modality

    def _process_image_input(
            self, image_input: Qwen3VLImageInputs
    ) -> tuple[tuple[jax.Array, ...],
               Optional[list[list[jax.Array]]]]:
        grid_thw = image_input["image_grid_thw"]
        if not grid_thw:
            return (), None

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].astype(self.visual.dtype)
            deepstack_embeds = None
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw)

        sizes = np.array([
            t * (h // self.spatial_merge_size) * (w // self.spatial_merge_size)
            for t, h, w in grid_thw
        ])

        if sizes.size == 0:
            return (), None
        if sizes.size == 1:
            image_splits = (image_embeds, )
            deepstack_by_item = None
            if deepstack_embeds:
                deepstack_by_item = [
                    [layer_embeds for layer_embeds in deepstack_embeds]
                ]
            return image_splits, deepstack_by_item

        split_indices = np.cumsum(sizes)[:-1]
        image_splits = tuple(jnp.split(image_embeds, split_indices))
        deepstack_by_item = None
        if deepstack_embeds:
            layer_splits = [
                tuple(jnp.split(layer_embeds, split_indices))
                for layer_embeds in deepstack_embeds
            ]
            deepstack_by_item = []
            for item_idx in range(len(image_splits)):
                deepstack_by_item.append(
                    [layer_split[item_idx] for layer_split in layer_splits]
                )
        return image_splits, deepstack_by_item

    def embed_multimodal(
        self,
        image_grid_thw: Tuple[Tuple[int, int, int], ...],
        **kwargs,
    ) -> dict:
        """Get multimodal embeddings from pixel values.

        This method is called by the serving infrastructure (multimodal_manager).
        DeepStack embeddings are returned alongside visual embeddings for caching.

        Args:
            image_grid_thw: Grid dimensions (T, H, W) for each image
            **kwargs: Contains 'pixel_values' for vision encoder

        Returns:
            A dict with:
              - "embeds": Tuple of embeddings, one per image (Qwen 2.5 VL format)
              - "deepstack": Optional list of per-image DeepStack embeddings
        """
        image_grid_thw = self._normalize_grid_thw(image_grid_thw)
        if not image_grid_thw:
            image_grid_thw = self._normalize_grid_thw(
                kwargs.get("video_grid_thw", None))

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            image_grid_thw, **kwargs)
        if not mm_input_by_modality:
            return {}
        if not image_grid_thw:
            return {}

        multimodal_embeddings: tuple[jax.Array, ...] = ()
        deepstack_outputs = None
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_splits, deepstack_by_item = self._process_image_input(
                    multimodal_input)
                multimodal_embeddings += image_splits
                if deepstack_by_item is not None:
                    if deepstack_outputs is None:
                        deepstack_outputs = []
                    deepstack_outputs.extend(deepstack_by_item)

        return {"embeds": multimodal_embeddings, "deepstack": deepstack_outputs}

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        visual_pos_mask = None
        deepstack_embeds = None
        if args:
            candidate = args[-1]
            if isinstance(candidate, (list, tuple)):
                deepstack_embeds = candidate

        if deepstack_embeds and input_ids is not None:
            visual_pos_mask = (input_ids == self.image_token_id) | (
                input_ids == self.video_token_id
            )

        kv_caches, hidden_states = self.language_model(
            kv_caches=kv_caches,
            input_ids=input_ids,
            attention_metadata=attention_metadata,
            inputs_embeds=inputs_embeds,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_embeds,
        )

        return kv_caches, hidden_states, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.language_model.compute_logits(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: List[int],
        hf_config=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths=None,
        use_audio_in_video: bool = False,
    ) -> Tuple[jax.Array, int]:
        """Compute MRoPE 3D position IDs for input sequence.

        This is a wrapper around the module-level get_mrope_input_positions function
        that uses the model's configuration.

        Args:
            input_tokens: List of token IDs for the sequence
            hf_config: Optional HF config (defaults to self.config)
            image_grid_thw: List of (T, H, W) tuples for each image
            video_grid_thw: List of (T, H, W) tuples for each video
            context_len: Context length for slicing positions
            seq_len: Sequence length for slicing positions

        Returns:
            llm_positions: (3, sliced_seq_len) position IDs for [T, H, W]
            mrope_position_delta: Delta for rope calculation
        """
        del second_per_grid_ts, audio_feature_lengths, use_audio_in_video

        if hf_config is None:
            hf_config = self.config

        llm_positions, mrope_position_delta = get_mrope_input_positions(
            input_tokens=input_tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
            vision_start_token_id=getattr(hf_config, "vision_start_token_id",
                                          self.vision_start_token_id),
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )

        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    def precompile_vision_encoder(
        self,
        run_compilation_fn,
    ) -> None:
        """Precompile vision encoder for warmup.

        Args:
            run_compilation_fn: Function to run compilation with signature
                (name, fn, *args, **kwargs)
        """
        vc = self.config.vision_config
        patch_input_dim = (
            vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
        )

        # Default warmup shapes for common image resolutions
        default_warmup_shapes = [
            (336, 336),    # Base resolution
            (448, 448),    # 1.33x
            (672, 672),    # 2x
            (896, 896),    # 2.67x (Qwen3-VL default max)
            (504, 336),    # 3:2 aspect ratio
            (336, 504),    # 2:3 aspect ratio
        ]

        # Get user-provided shapes from config
        user_shapes = []
        if warmup_config := self.vllm_config.additional_config.get(
            "vision_warmup_config"
        ):
            user_shapes = warmup_config.get("image_shapes", [])

        # Merge default shapes with user-provided shapes, avoiding duplicates
        seen_shapes = set()
        image_shapes = []
        for shape in default_warmup_shapes + user_shapes:
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                shape_tuple = tuple(shape)
                if shape_tuple not in seen_shapes:
                    seen_shapes.add(shape_tuple)
                    image_shapes.append(list(shape))

        factor = vc.patch_size * vc.spatial_merge_size
        for input_hw in image_shapes:
            if not isinstance(input_hw, list) or len(input_hw) != 2:
                logger.warning(f"Skipping invalid shape {input_hw}.")
                continue
            h_input, w_input = input_hw
            h_processed = round(h_input / factor) * factor
            w_processed = round(w_input / factor) * factor
            t, h, w = 1, h_processed // vc.patch_size, w_processed // vc.patch_size
            grid_thw = (t, h, w)
            num_patches = t * h * w

            dummy_pixel_values = jnp.ones(
                (num_patches, patch_input_dim),
                self.vllm_config.model_config.dtype,
            )
            dummy_grid_thw = (grid_thw,)

            run_compilation_fn(
                "vision_encoder",
                self.visual.encode_jit,
                dummy_pixel_values,
                dummy_grid_thw,
                image_shape=input_hw,
            )

    def load_weights(self, rng_key: jax.Array) -> None:
        self.rng = nnx.Rngs(rng_key)

        mappings = {
            "model.language_model.embed_tokens": "language_model.embed.embedding",
            "model.language_model.layers.*.input_layernorm": "language_model.layers.*.input_layernorm.scale",
            "model.language_model.layers.*.mlp.down_proj": "language_model.layers.*.mlp.down_proj.kernel",
            "model.language_model.layers.*.mlp.gate_proj": "language_model.layers.*.mlp.gate_proj.kernel",
            "model.language_model.layers.*.mlp.up_proj": "language_model.layers.*.mlp.up_proj.kernel",
            "model.language_model.layers.*.post_attention_layernorm": "language_model.layers.*.post_attention_layernorm.scale",
            "model.language_model.layers.*.self_attn.k_proj": "language_model.layers.*.self_attn.k_proj.kernel",
            "model.language_model.layers.*.self_attn.o_proj": "language_model.layers.*.self_attn.o_proj.kernel",
            "model.language_model.layers.*.self_attn.q_proj": "language_model.layers.*.self_attn.q_proj.kernel",
            "model.language_model.layers.*.self_attn.v_proj": "language_model.layers.*.self_attn.v_proj.kernel",
            "model.language_model.layers.*.self_attn.q_norm": "language_model.layers.*.self_attn.q_norm.scale",
            "model.language_model.layers.*.self_attn.k_norm": "language_model.layers.*.self_attn.k_norm.scale",
            "model.language_model.norm": "language_model.norm.scale",
            "model.visual.patch_embed.proj": "visual.patch_embed.proj.kernel",
            "model.visual.patch_embed.proj.bias": "visual.patch_embed.proj.bias",
            "model.visual.pos_embed": "visual.pos_embed.embedding",
            "model.visual.blocks.*.attn.qkv": "visual.blocks.*.attn.qkv_proj.kernel",
            "model.visual.blocks.*.attn.qkv.bias": "visual.blocks.*.attn.qkv_proj.bias",
            "model.visual.blocks.*.attn.proj": "visual.blocks.*.attn.proj.kernel",
            "model.visual.blocks.*.attn.proj.bias": "visual.blocks.*.attn.proj.bias",
            "model.visual.blocks.*.mlp.linear_fc1": "visual.blocks.*.mlp.fc1.kernel",
            "model.visual.blocks.*.mlp.linear_fc1.bias": "visual.blocks.*.mlp.fc1.bias",
            "model.visual.blocks.*.mlp.linear_fc2": "visual.blocks.*.mlp.fc2.kernel",
            "model.visual.blocks.*.mlp.linear_fc2.bias": "visual.blocks.*.mlp.fc2.bias",
            "model.visual.blocks.*.norm1": "visual.blocks.*.norm1.scale",
            "model.visual.blocks.*.norm1.bias": "visual.blocks.*.norm1.bias",
            "model.visual.blocks.*.norm2": "visual.blocks.*.norm2.scale",
            "model.visual.blocks.*.norm2.bias": "visual.blocks.*.norm2.bias",
            "model.visual.merger.norm": "visual.merger.norm.scale",
            "model.visual.merger.norm.bias": "visual.merger.norm.bias",
            "model.visual.merger.linear_fc1": "visual.merger.linear_fc1.kernel",
            "model.visual.merger.linear_fc1.bias": "visual.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2": "visual.merger.linear_fc2.kernel",
            "model.visual.merger.linear_fc2.bias": "visual.merger.linear_fc2.bias",
        }

        hf_config = self.vllm_config.model_config.hf_config
        if not hf_config.tie_word_embeddings:
            mappings["lm_head"] = "language_model.lm_head"

        # Add deepstack_merger_list mappings dynamically based on config
        # weight_utils.py only handles "layers" and "blocks" wildcards,
        # so we need explicit mappings for each deepstack merger index
        vision_config = hf_config.vision_config
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
        for i in range(len(deepstack_indexes)):
            mappings[f"model.visual.deepstack_merger_list.{i}.norm"] = f"visual.deepstack_merger_list.{i}.norm.scale"
            mappings[f"model.visual.deepstack_merger_list.{i}.norm.bias"] = f"visual.deepstack_merger_list.{i}.norm.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1"] = f"visual.deepstack_merger_list.{i}.linear_fc1.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc1.bias"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2"] = f"visual.deepstack_merger_list.{i}.linear_fc2.kernel"
            mappings[f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"] = f"visual.deepstack_merger_list.{i}.linear_fc2.bias"

        adapted_model_config = _ModelConfigAdapter(self.vllm_config.model_config)
        metadata_map = get_default_maps(
            adapted_model_config, self.mesh, mappings
        )
        load_hf_weights(
            vllm_config=self.vllm_config,
            model=self,
            metadata_map=metadata_map,
            mesh=self.mesh,
        )
