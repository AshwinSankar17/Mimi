import logging
import os
import re
import sys
import time
from multiprocess import set_start_method
from datetime import timedelta
import inspect
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

from transformers import HfArgumentParser
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry

from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin
from accelerate.utils.memory import release_memory

from src.configuration_mimi import MimiConfig
from src.modeling_mimi import MimiModel

from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)

from training.data import load_multiple_datasets

logger = logging.getLogger(__name__)