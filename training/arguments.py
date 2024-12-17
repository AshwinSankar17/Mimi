from dataclasses import dataclass, field
from typing import Optional, List

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation used. One of `eager`, `sdpa`, `flash_attention_2`"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

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
        default=35.0,
        metadata={
            "help": (
                "Clip audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`."
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training."
                " In this training script, `save_to_disk` must be set to the path in which the dataset should be saved. "
            )
        },
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
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
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
    save_to_disk: str = field(
        default=None,
        metadata={
            "help": "If set, will save the dataset to this path if this is an empyt folder. If not empty, will load the datasets from it."
        },
    )


@dataclass
class MimiCodecTrainingArguments(Seq2SeqTrainingArguments):
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    per_device_batch_size: int = field(
        default=8,
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
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for evaluation data loading (PyTorch only). 0 means that the data will be loaded in the main process."
            )
        },
    )