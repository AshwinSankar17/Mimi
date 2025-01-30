# coding=utf-8
# Copyright 2024 Kyutai, and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mimi model."""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from einops import rearrange, repeat

from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_mimi import MimiConfig

from accelerate.utils import broadcast as broadcast_tensors

import torchaudio as ta


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def _compute_entropy(usage: torch.Tensor) -> torch.Tensor:
    # Usage is some unnormalized distribution.
    proba = usage / usage.sum()
    p_log_p = torch.where(
        proba == 0, zero_scalar(usage.device), proba * torch.log(proba)
    )
    return -p_log_p.sum()


def _is_distributed() -> bool:
    # Checks if we need to use distributed routines.
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def _run_kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    # Kmeans algorithm used to initialize the codebooks.
    dim = samples.shape[-1]
    means = _sample_vectors(samples, num_clusters)
    bins = None

    for _ in range(num_iters):
        dists = torch.cdist(samples[None], means[None], p=2)[0]
        buckets = dists.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins.clamp_(min=1)

        new_means = torch.zeros_like(means)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means /= bins[..., None]
        resampled = _sample_vectors(samples, num_clusters)
        means = torch.where(zero_mask[..., None], resampled, new_means)

    assert bins is not None
    return means, bins


def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]


def semantic_distillation_loss_fn(feature, target_feature):
    """
    Compute cosine similarity loss for knowledge distillation
    
    Args:
        feature: B, T, D - Student model features
        target_feature: B, T, D - Teacher model features
    """
    n = min(feature.size(1), target_feature.size(1))
    # Directly use cosine similarity without sigmoid
    cos_sim = nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], dim=1)
    
    # Use 1 - cosine similarity to create a loss (minimizing distance)
    distill_loss = (1 - cos_sim).mean()
    
    return distill_loss

# General docstring
_CONFIG_FOR_DOC = "MimiConfig"


@dataclass
class MimiOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Mimi.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
        decoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    audio_codes: torch.LongTensor = None
    audio_values: torch.FloatTensor = None
    encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None
    decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None


@dataclass
class MimiEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    audio_codes: torch.LongTensor = None
    encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None


@dataclass
class MimiDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Mimi.
        decoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
    """

    audio_values: torch.FloatTensor = None
    decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None


class MimiConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode=None,
        bias: bool = True,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "MimiConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )

        kernel_size = self.conv.kernel_size[0]
        stride = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        dilation = self.conv.dilation[0]

        # Effective kernel size with dilations.
        kernel_size = torch.tensor((kernel_size - 1) * dilation + 1, dtype=torch.int64)

        self.register_buffer("stride", stride, persistent=False)
        self.register_buffer("kernel_size", kernel_size, persistent=False)
        self.register_buffer("padding_total", torch.tensor(kernel_size - stride, dtype=torch.int64), persistent=False)

        # Asymmetric padding required for odd strides
        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._get_extra_padding_for_conv1d
    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total

        return ideal_length - length

    @staticmethod
    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._pad1d
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if not mode == "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, hidden_states):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode)
        else:
            hidden_states = self._pad1d(
                hidden_states, (self.padding_left, self.padding_right + extra_padding), mode=self.pad_mode
            )

        hidden_states = self.conv(hidden_states)
        return hidden_states


class MimiConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias=True,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # Asymmetric padding required for odd strides
            self.padding_right = padding_total // 2

        self.padding_left = padding_total - self.padding_right

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        # unpad
        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., self.padding_left : end]
        return hidden_states


# Copied from transformers.models.encodec.modeling_encodec.EncodecResnetBlock with Encodec->Mimi,EnCodec->Mimi
class MimiResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by Mimi.
    """

    def __init__(self, config: MimiConfig, dim: int, dilations: List[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [MimiConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)

        if config.use_conv_shortcut:
            self.shortcut = MimiConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden_states):
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


class MimiEncoder(nn.Module):
    """SEANet encoder as used by Mimi."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        model = [MimiConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [MimiResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            # Add downsampling layers
            model += [nn.ELU()]
            model += [MimiConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [nn.ELU()]
        model += [MimiConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]

        self.layers = nn.ModuleList(model)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEncoder.forward
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MimiLayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.
    """

    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Mimi
class MimiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MimiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    # Copied from transformers.models.clip.modeling_clip.CLIPMLP.forward
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    **_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.scaling

    if mask is not None:  # no matter the length, we just slice it
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.config.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    target_dtype: torch.dtype = torch.float16,
    **_kwargs,
) -> Tuple[torch.Tensor, None]:
    if mask is not None:
        seq_len = mask.shape[1]
        query = query[:, :, :seq_len]
        value = value[:, :, :seq_len]

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor rotary embedding
    query_states = query.transpose(1, 2)
    key_states = key.transpose(1, 2)
    value_states = value.transpose(1, 2)

    dropout_rate = module.config.attention_dropout if module.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        mask,
        seq_len,
        dropout=dropout_rate,
        softmax_scale=module.scaling,
        is_causal=module.config.is_causal,
        sliding_window=module.sliding_window,
        use_top_left_mask=module.config._flash_attn_uses_top_left_mask,
    )

    return attn_output, None


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    output_attentions: bool = False,
    **_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # def causal_mask(b, h, q_idx, kv_idx):
    #     return q_idx >= kv_idx
    
    def sliding_window_causal_bias(score, b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        windowed_mask = (
            q_idx - kv_idx <= module.sliding_window
        )  # We dont need to check the right side of the sliding window since we are applying the causal mask

        return torch.where(causal_mask & windowed_mask, score, -float("inf"))

    attn_output = flex_attention(
        query,
        key,
        value,
        score_mod=sliding_window_causal_bias,
        enable_gqa=True,
        scale=module.scaling,
        return_lse=output_attentions,
    )
    if not output_attentions:
        attn_weights = None
    else:
        attn_output, attn_weights = attn_output

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    **_kwargs,
) -> Tuple[torch.Tensor, None]:
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = mask
    if mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query.device.type == "cuda" and causal_mask is not None:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and query.shape[1] > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=module.config.attention_dropout if module.training else 0.0,
        is_causal=is_causal,
        scale=module.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None

MIMI_ATTENTION_FUNCTION = {
    "flex_attention": flex_attention_forward,
    "eager": eager_attention_forward,
    "flash_attention_2": flash_attention_forward,
    "sdpa": sdpa_attention_forward,
}
# Copied from transformers.models.gemma.modeling_gemma.GemmaAttention with Gemma->Mimi
class MimiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MimiConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = 1 / math.sqrt(config.head_dim)
        # if not hasattr(self.config, "num_key_value_groups"):
        #     setattr(self.config, "num_key_value_groups", self.num_key_value_groups)
        # if not hasattr(self.config, "training"):
        #     setattr(self.config, "training", self.training)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = MimiRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.sliding_window = config.sliding_window  # Ignore copy

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if output_attentions and self.config._attn_implementation in ["sdpa", "flash_attention_2"]:
            logger.warning_once("Setting `attention_type` to `flex_attention` because `output_attentions=True`")
            attention_type = "flex_attention"
        else:
            attention_type = self.config._attn_implementation

        attn_output, attn_weights = MIMI_ATTENTION_FUNCTION[attention_type](
            self, query_states, key_states, value_states, attention_mask, output_attentions=output_attentions
        )

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.gemma.modeling_gemma.GemmaFlashAttention2 with Gemma->Mimi
class MimiFlashAttention2(MimiAttention):
    """
    Mimi flash attention module. This module inherits from `MimiAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.config._attn_implementation = "flash_attention_2"
        logger.warning_once(
            "The `MimiFlashAttention2` class is deprecated in favor of simply modifying the `config._attn_implementation`"
            "attribute of the `MimiAttention` class! It will be removed in v4.48"
        )

# Copied from transformers.models.gemma.modeling_gemma.GemmaSdpaAttention with Gemma->Mimi
class MimiSdpaAttention(MimiAttention):
    """
    Mimi attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MimiAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config._attn_implementation = "sdpa"
        logger.warning_once(
            "The `MimiSdpaAttention` class is deprecated in favor of simply modifying the `config._attn_implementation`"
            "attribute of the `MimiAttention` class! It will be removed in v4.48"
        )


class MimiTransformerLayer(nn.Module):
    def __init__(self, config: MimiConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MimiAttention(config=config, layer_idx=layer_idx)
        # self.self_attn = MIMI_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MimiTransformerModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MimiTransformerLayer`]

    Args:
        config: MimiConfig
    """

    def __init__(self, config: MimiConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [MimiTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation

        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded representation that will be contextualized by the model
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
                `past_key_values`).

                If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
                and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
                information on the default strategy.

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
                returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                Two formats are allowed:
                - a [`~cache_utils.Cache`] instance;
                - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
                cache format.

                The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
                legacy cache format will be returned.

                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
                have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
                of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache and not isinstance(past_key_values, Cache):
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = self._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.phi3.modeling_phi3.Phi3Model._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by nn.functional.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.mistral.modeling_mistral.MistralModel._prepare_4d_causal_attention_mask_with_cache_position with Mistral->Mimi
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: MimiConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`MimiConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class MimiDecoder(nn.Module):
    """SEANet decoder as used by Mimi."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [MimiConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add upsampling layers
            model += [nn.ELU()]
            model += [
                MimiConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [MimiResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        model += [MimiConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecDecoder.forward
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MimiSemanticPooling(nn.Module):
    """
    Adapter and Embedding Resampler for Semantic Distillation into SplitRVQ
    """

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.adapter = nn.Conv1d(config.semantic_feature_dim, config.hidden_size, 1)
        self.pooling = nn.AvgPool1d(kernel_size=config.pooling_kernel_size, stride=config.pooling_stride)
    
    def forward(self, semantic_features):
        B, T, C = semantic_features.shape
        out = self.adapter(semantic_features.transpose(1, 2))
        out = self.pooling(out)
        return out

class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: MimiConfig, epsilon: float = 1e-5):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)
        
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer("_embed", None, persistent=False)
        self.register_buffer("embed_sum", embed)
        # self._embed = None
        self.epsilon = epsilon

        self.decay = config.decay # Add to config
        self.threshold_usage_ratio = config.threshold_usage_ratio  # Add to config
        self.replaced_usage_ratio = config.replaced_usage_ratio # Add to config
        self.check_unused_every = config.check_unused_every
        self._next_unused_check = config.check_unused_every
        self._cached_initialized = False

    @property
    def embed(self) -> torch.Tensor:
        if self._embed is None:
            embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self.register_buffer("_embed", embed, persistent=False)
            return embed
        return self._embed
    
    @property
    def initialized(self) -> bool:
        """Cached version of self._initialized,
        This assumes that once the module is initialized, it will never go back to the uninitialized state."""
        if not self._cached_initialized:
            self._cached_initialized = self._initialized.item()
        return self._cached_initialized
    
    def _init_embedding(self, data: torch.Tensor) -> None:
        # Initialize the codebook, e.g. using kmeans.
        if self.initialized:
            return

        rank = 0
        if _is_distributed():
            rank = torch.distributed.get_rank()
            # First gathering shapes in case not all GPUs have the same effective batch size.
            # then gathering the actual content.
            if rank == 0:
                other_shapes: List[torch.Size] = [None] * torch.distributed.get_world_size()  # type: ignore
                torch.distributed.gather_object(data.shape, other_shapes)
                other_data: List[torch.Tensor] = [
                    torch.empty(shape, device=data.device, dtype=data.dtype) for shape in other_shapes]
                torch.distributed.gather(data, other_data)
                data = torch.cat(other_data, dim=0)
            else:
                torch.distributed.gather_object(data.shape)
                torch.distributed.gather(data)
        if rank == 0:
            embedding, cluster_usage = _run_kmeans(data, self.codebook_size)
            self.embed_sum.data.copy_(embedding * cluster_usage[:, None])
            self.cluster_usage.data.copy_(cluster_usage)
            self._initialized.data.fill_(1)
        # Make sure all buffers across workers are in sync after initialization
        self._broadcast_buffers()
    
    def _broadcast_buffers(self) -> None:
        if _is_distributed():
            for buffer in self.buffers():
                torch.distributed.broadcast(buffer, 0)
    
    def _replace_expired_codes(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        # Replaces expired centroids, as indicated by `mask` (a true value indicate the code needs to be replaced).
        # The new codes are sampled from the batch `samples`.
        new_vectors = _sample_vectors(samples, self.codebook_size)
        replace_cluster_usage = (
            self.replaced_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        )
        self.embed_sum[:] = torch.where(
            mask[:, None], replace_cluster_usage * new_vectors, self.embed_sum
        )
        self.cluster_usage[:] = torch.where(
            mask, replace_cluster_usage, self.cluster_usage
        )

    def _check_expired_codes(self, batch_samples: torch.Tensor) -> torch.Tensor:
        # Checks whether some centroids are under utilized, and replace them if necessary.
        if not self.initialized:
            return torch.tensor(0.0, device=batch_samples.device)

        self._next_unused_check -= 1
        if self._next_unused_check > 0:
            return torch.tensor(0.0, device=batch_samples.device)
        # we don't check every iteration to avoid having too many sync points.
        self._next_unused_check = self.check_unused_every
        threshold_cluster_usage = self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size
        expired_codes = self.cluster_usage < threshold_cluster_usage

        assert batch_samples.dim() == 2
        self._replace_expired_codes(batch_samples, mask=expired_codes)
        self._broadcast_buffers()

        return expired_codes.float().mean()

    def quantize(self, hidden_states):
        # Projects each vector in `hidden_states` over the nearest centroid and return its index.
        # `hidden_states` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        dists = torch.cdist(hidden_states[None], self.embed[None], p=2)[0]
        embed_ind = dists.argmin(dim=-1)
        return embed_ind

    def dequantize(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize

    def forward(self, hidden_states, initialize=True):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        if self.training and initialize:
            self._init_embedding(hidden_states.detach())
        # quantize
        flat_embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = flat_embed_ind.view(*shape[:-1])
        quantize = self.dequantize(embed_ind)

        metrics = {}

        if self.training:
            expired = self._check_expired_codes(hidden_states)
            metrics['rvq_expired'] = expired
            cluster_usage = torch.zeros_like(self.cluster_usage)
            cluster_usage.scatter_add_(
                0, flat_embed_ind, torch.ones_like(flat_embed_ind, dtype=cluster_usage.dtype)
            )
            _ema_inplace(self.cluster_usage, cluster_usage, self.decay)

            if self.initialized:
                metrics['rvq_entropy'] = _compute_entropy(self.cluster_usage) / math.log(self.codebook_size)
            
            embed_sum = torch.zeros_like(self.embed_sum).to(hidden_states.dtype)
            # embed_sum.scatter_add_(0, repeat(flat_embed_ind, "n -> n d", d=self.codebook_dim), hidden_states)
            
            embed_sum.scatter_add_(0, flat_embed_ind.unsqueeze(-1).repeat(1, self.codebook_dim), hidden_states)
            _ema_inplace(self.embed_sum, embed_sum, self.decay)
            self.register_buffer("_embed", None)
        
        return quantize, embed_ind, metrics

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.encode
    def encode(self, hidden_states):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.decode
    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize


# Copied from transformers.models.encodec.modeling_encodec.EncodecVectorQuantization with Encodec->Mimi
class MimiVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)
    
    @property
    def embedding(self):
        return self.codebook.embedding

    @property
    def initialized(self):
        return self.codebook.initialized
    
    def forward(self, hidden_states, initialize=True):
        hidden_states = hidden_states.permute(0, 2, 1)
        quantize, embed_ind, metrics = self.codebook(hidden_states, initialize)

        if self.training:
            quantize = hidden_states + (quantize - hidden_states).detach()
            loss = nn.functional.mse_loss(hidden_states, quantize.detach())
        else:
            loss = zero_scalar(hidden_states.device)
        
        quantize = quantize.permute(0, 2, 1)

        return quantize, embed_ind, loss, metrics

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class MimiResidualVectorQuantization(nn.Module):

    """Residual Vector Quantizer."""

    def __init__(self, config: MimiConfig, num_quantizers: int = None, codebook_offset: int = 0):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = num_quantizers
        self.codebook_offset = codebook_offset
        self.layers = nn.ModuleList([MimiVectorQuantization(config) for _ in range(self.num_quantizers)])
    
    def forward(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=embeddings.device)

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers
        previous_layer_is_initialized = True

        residual = embeddings
        all_losses = []
        all_codes = []
        all_metrics = {}
        for ix, layer in enumerate(self.layers[:num_quantizers]):
            quantized, codes, loss, metrics = layer(
                residual, initialize=previous_layer_is_initialized
            )
            if self.training:
                previous_layer_is_initialized = layer.initialized

            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            
            all_codes.append(codes)
            all_losses.append(loss)

            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k] += v / num_quantizers
                else:
                    all_metrics[k] = v / num_quantizers
                all_metrics[k + f"_{ix + self.codebook_offset}"] = v

        if self.training:
            quantized_out = embeddings + (quantized_out - embeddings).detach()

        out_losses, out_indices = map(torch.stack, (all_losses, all_codes))
        return quantized_out, out_indices, out_losses, all_metrics

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes of shape [B, K, T] to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        # codes = codes.transpose(0, 1)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized

        return quantized_out



class MimiResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(
        self,
        config: MimiConfig, 
        num_quantizers: int = None, 
        codebook_offset: int = 0,
        q_dropout: bool = True,
    ):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = num_quantizers
        self.codebook_offset = codebook_offset

        ## Add to Config
        self.q_dropout = q_dropout
        self.no_quantization_rate = config.no_quantization_rate
        self.rng_dropout = random.Random(1234)

        self.input_proj = None
        self.output_proj = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = torch.nn.Conv1d(
                config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False
            )
            self.output_proj = torch.nn.Conv1d(
                config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False
            )
        
        self.vq = MimiResidualVectorQuantization(config, self.num_quantizers, self.codebook_offset)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = self.num_quantizers
        
        if self.training and self.q_dropout:
            num_quantizers = self.rng_dropout.randint(1, num_quantizers)
        
        quantized, codes, commit_loss, metrics = self.vq(embeddings, num_quantizers=num_quantizers)
        B, *_ = quantized.shape
        
        if self.training and self.no_quantization_rate > 0:
            mask = (torch.rand(B, 1, 1, device=x.device) <= self.no_quantization_rate).float()
            quantized = embeddings * mask + (1 - mask) * quantized

        if self.output_proj is not None:
            quantized = self.output_proj(quantized)
        
        codes = codes.transpose(0, 1)
        return quantized, codes, commit_loss.mean(), metrics

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

        out_indices = self.vq.encode(embeddings, num_quantizers)
        out_indices = out_indices.transpose(0, 1)

        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes of shape [B, K, T] to the quantized representation."""
        
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)

        if self.output_proj:
            quantized = self.output_proj(quantized)

        return quantized


class MimiSplitResidualVectorQuantizer(nn.Module):
    """Split Residual Vector Quantizer."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.max_num_quantizers = config.num_quantizers

        self.semantic_teacher = MimiSemanticPooling(config)

        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_quantizers - config.num_semantic_quantizers

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_semantic_quantizers, 0, False
        )
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_acoustic_quantizers, 1, config.q_dropout
        )
    
    def _renorm_and_add(
        self,
        semantic_features: torch.Tensor,
        acoustic_features: torch.Tensor,
        num_quantizers_semantic: int,
        num_quantizers_acoustic: int,
    ) -> torch.Tensor:
        """Renormalizes values from `rvq_first` and `rvq_rest` and adds them.

        This allows correcting statistics that are normalized by the number of quantizers. To renormalize, we use the
        number of quantizers that are actually used, e.g. taking into account quantizer dropout.
        """
        num_quantizers = num_quantizers_semantic + num_quantizers_acoustic
        renorm_semantic_features = semantic_features * num_quantizers_semantic / num_quantizers
        renorm_acoustic_features = acoustic_features * num_quantizers_acoustic / num_quantizers
        return renorm_semantic_features + renorm_acoustic_features
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        target_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        semantic_features_res = self.semantic_residual_vector_quantizer(embeddings)
        semantic_distillation_loss = torch.tensor(0.0, device=embeddings.device)
        if self.training and target_features is not None:
            target_features = self.semantic_teacher(target_features)
            semantic_distillation_loss = semantic_distillation_loss_fn(
                semantic_features_res[0].transpose(1, 2), 
                target_features.transpose(1, 2)
            )

        if self.max_num_quantizers == self.num_semantic_quantizers:
            return semantic_features_res
        acoustic_features_res = self.acoustic_residual_vector_quantizer(embeddings)
        full_quantized_embedding = semantic_features_res[0] + acoustic_features_res[0]
        full_quantized_codes = torch.cat([semantic_features_res[1], acoustic_features_res[1]], dim=1)

        num_quantizers_semantic = semantic_features_res[1].shape[1]
        num_quantizers_acoustic = acoustic_features_res[1].shape[1]

        full_quantized_penalty = self._renorm_and_add(
            semantic_features_res[2], acoustic_features_res[2], num_quantizers_semantic, num_quantizers_acoustic
        )

        full_quantized_metrics = semantic_features_res[3]
        for k, v in acoustic_features_res[3].items():
            if k in full_quantized_metrics:
                full_quantized_metrics[k] = self._renorm_and_add(
                    full_quantized_metrics[k], v, num_quantizers_semantic, num_quantizers_acoustic
                )
            else:
                full_quantized_metrics[k] = v
        
        return full_quantized_embedding, full_quantized_codes, (full_quantized_penalty, semantic_distillation_loss), full_quantized_metrics

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """

        num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.max_num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
            )

        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
            )

        # codes is [K, B, T], with T frames, K nb of codebooks.
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)

        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""

        # The first num_semantic_quantizers codebooks are decoded using the semantic RVQ
        quantized_out = self.semantic_residual_vector_quantizer.decode(codes[:, : self.num_semantic_quantizers])

        # The rest of the codebooks are decoded using the acoustic RVQ
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized_out += self.acoustic_residual_vector_quantizer.decode(codes[:, self.num_semantic_quantizers :])
        return quantized_out


class MimiPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MimiConfig
    base_model_prefix = "mimi"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    # Copied from transformers.models.encodec.modeling_encodec.EncodecPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)


MIMI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MimiConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


MIMI_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*.
        num_quantizers (`int`, *optional*):
            Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
        decoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The Mimi neural audio codec model.",
    MIMI_START_DOCSTRING,
)
class MimiModel(MimiPreTrainedModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformerModel(config)

        self.downsample = None
        self.upsample = None
        if config.frame_rate != config.encodec_frame_rate:
            self.downsample = MimiConv1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
            )

            self.upsample = MimiConvTranspose1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                groups=config.upsample_groups,
            )

        self.decoder_transformer = MimiTransformerModel(config)
        self.decoder = MimiDecoder(config)

        self.quantizer = MimiSplitResidualVectorQuantizer(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        num_quantizers: int,
        padding_mask: int,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. The padding mask is required to compute the correct scale.
        """
        embeddings = self.encoder(input_values)
        encoder_outputs = self.encoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = encoder_outputs.get("past_key_values")
        elif len(encoder_outputs) > 1:
            past_key_values = encoder_outputs[1]
        embeddings = encoder_outputs[0].transpose(1, 2)
        embeddings = self.downsample(embeddings)

        codes = self.quantizer.encode(embeddings, num_quantizers)
        codes = codes.transpose(0, 1)
        return codes, past_key_values

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        num_quantizers: Optional[float] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], MimiEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            num_quantizers (`int`, *optional*):
                Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
            encoder_past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
                This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                The model will output the same cache format that is fed as input.

                If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
                have their past key value states given to this model).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `codebook` of shape `[batch_size, num_codebooks, frames]`, the discrete encoded codes for the input audio waveform.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        num_quantizers = self.config.num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.config.num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.config.num_quantizers}, but is currently {num_quantizers}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames, encoder_past_key_values = self._encode_frame(
            input_values,
            num_quantizers,
            padding_mask.bool(),
            past_key_values=encoder_past_key_values,
            return_dict=return_dict,
        )

        if not return_dict:
            return (
                encoded_frames,
                encoder_past_key_values,
            )

        return MimiEncoderOutput(encoded_frames, encoder_past_key_values)

    def _decode_frame(
        self,
        codes: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        embeddings = self.quantizer.decode(codes)

        embeddings = self.upsample(embeddings)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(embeddings)
        return outputs, past_key_values

    def decode(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MimiDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            decoder_past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
                This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                The model will output the same cache format that is fed as input.

                If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
                have their past key value states given to this model).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_values, decoder_past_key_values = self._decode_frame(
            audio_codes, past_key_values=decoder_past_key_values, return_dict=return_dict
        )

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (
                audio_values,
                decoder_past_key_values,
            )
        return MimiDecoderOutput(audio_values, decoder_past_key_values)

    @add_start_docstrings_to_model_forward(MIMI_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MimiOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], MimiOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, MimiModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "kyutai/mimi"
        >>> model = MimiModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        
        embeddings = self.encoder(input_values)
        encoder_outputs = self.encoder_transformer(embeddings.transpose(1, 2))
        embeddings = encoder_outputs[0].transpose(1, 2)
        embeddings = self.downsample(embeddings)

        quantized = self.quantizer(embeddings, target_features=target_embeddings)

        embeddings = self.upsample(quantized[0])
        decoder_outputs = self.decoder_transformer(embeddings.transpose(1, 2))
        embeddings = decoder_outputs[0].transpose(1, 2)
        audio_values = self.decoder(embeddings)

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        return audio_values, quantized


class MimiModelForTraining(nn.Module):
    def __init__(self, model, ssl_model_name_or_path):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(ssl_model_name_or_path)
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(ssl_model_name_or_path)
        self.model = model
        self.freeze_semantic_model()

    def freeze_semantic_model(self):
        for param in self.semantic_model.parameters():
            param.requires_grad = False

    def forward(self, input_values, padding_mask):
        semantic_input_values = gather_unpadded_tensors(input_values, padding_mask)
        semantic_input_values = list(
            map(
                lambda audio: ta.functional.resample(
                    audio, 
                    self.model.config.sampling_rate, 
                    self.processor.sampling_rate
                ).cpu().numpy(), 
                semantic_input_values
            )
        )
        semantic_inputs = self.processor(
            semantic_input_values, 
            sampling_rate=self.processor.sampling_rate, 
            return_tensors="pt"
        ).to(device=input_values.device)

        semantic_target_features = self.semantic_model(**semantic_inputs).last_hidden_state

        return self.model(input_values, padding_mask, target_embeddings=semantic_target_features)

def gather_unpadded_tensors(tensor, padding_mask):
    """
    Gathers a list of unpadded tensors from the input tensor and padding mask.

    Args:
        tensor (torch.Tensor): The input tensor of shape `(batch_size, seq_len, feature_dim)`.
        padding_mask (torch.Tensor): The padding mask of shape `(batch_size, seq_len)` 
                                      with `1` indicating valid tokens and `0` indicating padding.

    Returns:
        list[torch.Tensor]: A list of unpadded tensors, one for each batch element.
    """
    # Compute lengths of unpadded sequences
    lengths = padding_mask.sum(dim=1)
    
    unpadded_tensors = []
    for i, length in enumerate(lengths):
        # Slice up to the valid length
        unpadded_tensors.append(tensor[i, :length])
    
    return unpadded_tensors