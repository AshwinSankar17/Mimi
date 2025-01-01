from abc import ABC, abstractmethod

import typing as tp

import torchaudio
import torch
from torch import nn
from einops import rearrange

from transformers.modeling_utils import PreTrainedModel

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
MultiDiscriminatorOutputType = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]

def apply_parametrization_norm(module: nn.Module, norm: str = 'none'):
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        return module

def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs):
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class MultiDiscriminator(ABC, nn.Module):
    """Base implementation for discriminators composed of sub-discriminators acting at different scales.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        ...

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        """Number of discriminators.
        """
        ...


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.

    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_fft (int): Size of FFT for each scale.
        hop_length (int): Length of hop between STFT windows for each scale.
        kernel_size (tuple of int): Inner Conv2d kernel sizes.
        stride (tuple of int): Inner Conv2d strides.
        dilations (list of int): Inner Conv2d dilation on the time dimension.
        win_length (int): Window size for each scale.
        normalized (bool): Whether to normalize by magnitude after stft.
        norm (str): Normalization method.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters.
    """
    def __init__(self, config, layer_idx,):
        super().__init__()
        assert len(config.kernel_size) == 2
        assert len(config.stride) == 2
        self.filters = config.filters
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.n_fft = config.n_ffts[layer_idx]
        self.hop_length = config.hop_lengths[layer_idx]
        self.win_length = config.win_lengths[layer_idx]
        self.normalized = config.normalized
        self.activation = getattr(torch.nn, config.activation)(**config.activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=self.normalized, center=False, pad_mode=None, power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=config.kernel_size, padding=get_2d_padding(config.kernel_size))
        )
        in_chs = min(config.filters_scale * self.filters, config.max_filters)
        for i, dilation in enumerate(config.dilations):
            out_chs = min((config.filters_scale ** (i + 1)) * self.filters, config.max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=config.kernel_size, stride=config.stride,
                                         dilation=(dilation, 1), padding=get_2d_padding(config.kernel_size, (dilation, 1)),
                                         norm=config.norm))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(config.kernel_size[0], config.kernel_size[0]),
                                     padding=get_2d_padding((config.kernel_size[0], config.kernel_size[0])),
                                     norm=config.norm))
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(config.kernel_size[0], config.kernel_size[0]),
                                    padding=get_2d_padding((config.kernel_size[0], config.kernel_size[0])),
                                    norm=config.norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(PretrainedModel):
    """Multi-Scale STFT (MS-STFT) discriminator.

    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        sep_channels (bool): Separate channels to distinct samples for stereo support.
        n_ffts (Sequence[int]): Size of FFT for each scale.
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale.
        win_lengths (Sequence[int]): Window size for each scale.
        **kwargs: Additional args for STFTDiscriminator.
    """
    def __init__(self, config):
        super().__init__(config)
        assert len(config.n_ffts) == len(config.hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(config, i)
            # DiscriminatorSTFT(config.filters, in_channels=config.in_channels, out_channels=config.out_channels,
            #                   n_fft=config.n_ffts[i], win_length=config.win_lengths[i], hop_length=config.hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])

    @property
    def num_discriminators(self):
        return len(self.discriminators)

    def _separate_channels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        return x.view(-1, 1, T)

    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
    
    # def from_pretrained(self, model_name_or_path):
    #     state_dict = torch.load(model_name_or_path, map_location="cpu")["state_dict"]
    #     self.load_state_dict(state_dict, strict=False)