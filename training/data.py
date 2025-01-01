import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import torch
import librosa
import datasets
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from datasets import Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset

from semantic_features import mask_from_lens, W2V2BertFeature


@dataclass
class DataCollatorMimiCodecWithPadding:
    audio_column_name: str
    max_audio_length: float = 7.0
    sampling_rate: int = 24_000
    semantic_feature_model: torch.nn.Module
    # ssl_sampling_rate: int = 16_000

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Check if sampling rate matches
        assert self.sampling_rate == batch[0][self.audio_column_name]["sampling_rate"], (
            "Given sampling rate and audio column sampling rate do not match"
        )
        ssl_sampling_rate = semantic_feature_model.sampling_rate
        
        # Calculate max audio length in samples
        max_audio_pad_length = int(self.max_audio_length * self.sampling_rate)
        
        # Extract and truncate audio arrays
        audios = [torch.from_numpy(feature[self.audio_column_name]["array"][:max_audio_pad_length]) for feature in batch]
        len_audio = torch.tensor([len(audio) for audio in audios], dtype=torch.int32)
        
        # Calculate max audio length in samples
        max_ssl_pad_length = int(self.max_audio_length * self.ssl_sampling_rate)
        # Resample audio for SSL models
        audios_ssl = [
            torch.from_numpy(librosa.resample(
                feature, 
                orig_sr=self.sampling_rate, 
                target_sr=self.ssl_sampling_rate
            ))
            for feature in audios
        ]

        # ssl_len_audio = torch.tensor([len(audio) for audio in audios_ssl], dtype=torch.int32)
        
        # Pad audios and audios_ssl
        def pad_sequences(sequences, max_length, pad_value=0.0):
            padded = torch.full((len(sequences), max_length), pad_value, dtype=torch.float32)
            for i, seq in enumerate(sequences):
                length = min(len(seq), max_length)
                padded[i, :length] = torch.tensor(seq[:length], dtype=torch.float32)
            return padded

        # Apply padding
        audios_padded = pad_sequences(audios, max_audio_pad_length)
        audios_padding_mask = mask_from_lens(len_audio)
        ssl_embeddings, ssl_padding_mask = self.semantic_feature_model(audios_ssl)
        
        # Return as dictionary
        return {
            "input_values": audios_padded,
            "input_padding_mask": audios_padding_mask,
            "ssl_embeddings": ssl_embeddings,
            "ssl_padding_mask": ssl_padding_mask,
        }


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    dataset_samples=None,
    default_split="train",
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict


def load_multiple_datasets(
    accelerator: Accelerator,
    dataset_names: Union[List, str],
    dataset_config_names: Union[List, str],
    splits: Optional[Union[List, str]] = None,
    label_column_names: Optional[List] = None,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = False,
    seed: Optional[int] = None,
    id_column_name: Optional[str] = None,
    columns_to_keep: Optional[Set[str]] = None,
    prompt_column_name: Optional[str] = None,
    sampling_rate: Optional[int] = None,
    audio_column_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, dataset_config_names, splits, label_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(dataset_names_dict, desc="Combining datasets..."):
        with accelerator.local_main_process_first():
            dataset = load_dataset(
                dataset_dict["name"],
                dataset_dict["config"],
                split=dataset_dict["split"],
                streaming=streaming,
                **kwargs,
            )
            dataset_features = dataset.features.keys()

            if sampling_rate is not None and audio_column_name is not None:
                # resample target audio
                dataset = dataset.cast_column(audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate))
                dataset_features = dataset.features.keys()

            if columns_to_keep is not None:
                dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        with accelerator.local_main_process_first():
            interleaved_dataset = concatenate_datasets(all_datasets)

    return interleaved_dataset