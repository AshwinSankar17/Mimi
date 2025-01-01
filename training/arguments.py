from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    discriminator_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for discriminator"}
    )
    attn_implementation: str = field(
        default="flex",
        metadata={"help": "Attention implementation used. One of `eager`, `sdpa`, `flash_attention_2`, `flex`"},
    )
    ssl_model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    sampling_rate: int = field(
        default=24_000,
        metadata={
            "help": ("The sampling rate at which the codec should be trained")
        }
    )
    ssl_sampling_rate: int = field(
        default=16_000,
        metadata={
            "help": ("The sampling rate at which the codec should be trained")
        }
    )
    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load and combine "
            " librispeech and common voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol."
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": ("The name of the training data set split to use (via the datasets library). Defaults to 'train'")
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in the training data. Load and combine "
            "multiple datasets by separating dataset samples by a '+' symbol."
        },
    )
    eval_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training dataset name if unspecified."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the training dataset config name if unspecified"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )
    target_audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the target audio data. Defaults to 'audio'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_duration_in_seconds: float = field(
        default=7.0,
        metadata={
            "help": (
                "Clip audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`."
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.5, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to load the datasets in streaming mode."
            )
        },
    )
    add_audio_samples_to_wandb: bool = field(
        default=False,
        metadata={"help": "If set and if `wandb` in args.report_to, will add generated audio samples to wandb logs."},
    )
    id_column_name: str = field(default=None, metadata={"help": "id column name."})
    wandb_project: str = field(
        default="parler-speech",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_run_name: str = field(
        default=None,
        metadata={
            "help": "If specified, the name of the run. If not specified, wandb will give a random name to this run."
        },
    )


@dataclass
class MimiCodecTrainingArguments(Seq2SeqTrainingArguments):
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    per_device_batch_size: int = field(
        default=96,
        metadata={"help": ("Specify the batch size of the audio encoding pre-processing steps.")},
    )
    train_dataloader_num_workers: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "Number of subprocesses to use for evaluation data loading (PyTorch only). 0 means that the data will be loaded in the main process."
            )
        },
    )
    eval_dataloader_num_workers: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "Number of subprocesses to use for evaluation data loading (PyTorch only). 0 means that the data will be loaded in the main process."
            )
        },
    )

@dataclass
class DiscriminatorArguments:
    num_filters: int = field(
        default=32,
        metadata={
            "help": "Number of filters in convolutions for MultiScaleSTFTDiscriminator"
        }
    )
    in_channels: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of input channels to MultiScaleSTFTDiscriminator"
        }
    )
    out_channels: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of output channels to MultiScaleSTFTDiscriminator"
        }
    )
    n_ffts: Optional[List[int]] = field(
        default_factory=lambda: [1024, 2048, 512],
        metadata={
            "help": "Size of FFT for each scale in MultiScaleSTFTDiscriminator"
        }
    )
    hop_lengths: Optional[List[int]] = field(
        default_factory=lambda: [256, 512, 128],
        metadata={
            "help": "Length of hops STFT windows for each scale in MultiScaleSTFTDiscriminator"
        }
    )
    window_lengths: Optional[List[int]] = field(
        default_factory=lambda: [1024, 2048, 512],
        metadata={
            "help": "Window size for each scale in MultiScaleSTFTDiscriminator"
        }
    )
    max_filters: int = field(
        default=1024,
        metadata={
            "help": "Maximum number of filters for the discriminator"
        }
    )
    filters_scale: int = field(
        default=1,
        metadata={
            "help": "Scale factor for the filters"
        }
    )
    kernel_size: Tuple[int, int] = field(
        default_factory=lambda: (3, 9),
        metadata={
            "help": "Kernel size for convolutions"
        }
    )
    dilations: List[int] = field(
        default_factory=lambda: [1, 2, 4],
        metadata={
            "help": "Dilation rates for convolutions"
        }
    )
    stride: Tuple[int, int] = field(
        default_factory=lambda: (1, 2),
        metadata={
            "help": "Stride for convolutions"
        }
    )
    normalized: bool = field(
        default=True,
        metadata={
            "help": "Whether to normalize the input"
        }
    )
    norm: str = field(
        default='weight_norm',
        metadata={
            "help": "Normalization technique to use"
        }
    )
    activation: str = field(
        default='LeakyReLU',
        metadata={
            "help": "Activation function to use"
        }
    )
    activation_params: dict = field(
        default_factory=lambda: {'negative_slope': 0.2},
        metadata={
            "help": "Parameters for the activation function"
        }
    )