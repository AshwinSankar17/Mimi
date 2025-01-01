"""Discriminator model configuration"""

import math

import numpy as np

from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

@dataclass
class DiscriminatorConfig(PretrainedConfig):
    filters: int = 32
    in_channels: int = 1
    out_channels: int = 1
    out_channels: int = 1
    n_ffts: tp.List[int] = [1024, 2048, 512]
    hop_lengths: tp.List[int] = [256, 512, 128]
    win_lengths: tp.List[int] = [1024, 2048, 512]
    max_filters: int = 1024
    filters_scale: int = 1
    kernel_size: tp.Tuple[int, int] = (3, 9)
    dilations: tp.List = [1, 2, 4]
    stride: tp.Tuple[int, int] = (1, 2)
    normalized: bool = True
    norm: str = 'weight_norm'
    activation: str = 'LeakyReLU' 
    activation_params: dict = {'negative_slope': 0.2}


    model_type: str = "discriminator"
