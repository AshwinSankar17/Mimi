"""Discriminator model configuration"""

import math
import numpy as np
import typing as tp

from dataclasses import dataclass, field

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class DiscriminatorConfig(PretrainedConfig):
    model_type="discriminator"
    def __init__(
        self,
        filters=32,
        in_channels=1,
        out_channels=1,
        n_ffts=None,
        hop_lengths=None,
        win_lengths=None,
        max_filters=1024,
        filters_scale=1,
        kernel_size=(3, 9),
        dilations=None,
        stride=(1, 2),
        normalized=True,
        norm='weight_norm',
        activation='LeakyReLU',
        activation_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_ffts = n_ffts or [1024, 2048, 512]
        self.hop_lengths = hop_lengths or [256, 512, 128]
        self.win_lengths = win_lengths or [1024, 2048, 512]
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4]
        self.stride = stride
        self.normalized = normalized
        self.norm = norm
        self.activation = activation
        self.activation_params = activation_params or {'negative_slope': 0.2}
