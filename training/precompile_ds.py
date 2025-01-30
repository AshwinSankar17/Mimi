import os
import torch
import datasets
import argparse
import librosa
from transformers import AutoModel, AutoFeatureExtractor

from multiprocessing import set_start_method

set_start_method("spawn")

def parse_args():
    parser = argparse.ArgumentParser(description="Load and preprocess Hugging Face datasets")
    parser.add_argument('--datasets', type=str, required=True, help="Hugging Face dataset strings separated by '+'")
    parser.add_argument('--num_proc', type=int, default=1, help="Number of processes to use")
    parser.add_argument('--subset', type=str, default='default', help="Subset of the dataset to load")
    parser.add_argument('--split', type=str, default='train', help="Split of the dataset to load")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_names = args.datasets.split('+')
    subset_names = args.subset.split('+')
    split_names = args.split.split('+')
    assert len(dataset_names) == len(subset_names) == len(split_names), "Length of dataset names, subset names, and split names must be the same"

    audio_encoder = AutoModel.from_pretrained("facebook/w2v-bert-2.0", torch_dtype=torch.bfloat16)
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    print("Loaded audio encoder and feature extractor")
    for param in audio_encoder.parameters():
        param.requires_grad = False
    audio_encoder.eval()

    def apply_audio_preprocessing(batch, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        audio_encoder.to(device)

        audios_list = [sample['array'] for sample in batch['audio']]
        inputs = feature_extractor(
            audios_list,
            sampling_rate=feature_extractor.sampling_rate, 
            padding="longest",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad() and torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = audio_encoder(**inputs)['last_hidden_state']

        lengths = inputs['attention_mask'].sum(dim=1).cpu().tolist()  # List of lengths for each batch element

        # Step 2: Slice embeddings and convert to CPU tensors
        embeddings_list = [
            embeddings[i, :lengths[i], :].float().cpu()  # Slice valid token embeddings and move to CPU
            for i in range(len(lengths))
        ]
        batch['ssl_embeddings'] = embeddings_list
        return batch

    # all_datasets = []
    for dataset_name, subset_name, split_name in zip(dataset_names, subset_names, split_names):
        if os.path.exists(f"precompiled_dataset/{dataset_name.replace('/', '_')}_{subset_name}/{split_name}"):
            print(f"Dataset {dataset_name} Subset {subset_name} Split {split_name} already processed, skipping")
            continue
        print(f"Processing Dataset {dataset_name} Subset {subset_name} Split {split_name}")
        dataset = datasets.load_dataset(dataset_name, subset_name, split=split_name, num_proc=args.num_proc)
        dataset = dataset.select_columns(['audio'])
        dataset = dataset.cast_column('audio', datasets.Audio(sampling_rate=24000))
        ds = dataset.cast_column('audio', datasets.Audio(sampling_rate=16000))
        
        dataset = dataset.map(
            lambda row, idx: {'_idx': idx}, 
            with_indices=True,
            num_proc=args.num_proc
        )
        ds = ds.map(
            lambda row, idx: {'_idx': idx}, 
            with_indices=True,
            num_proc=args.num_proc
        )

        ds = ds.map(
            apply_audio_preprocessing,
            batched=True,
            batch_size=32,
            remove_columns=['audio'],
            with_rank=True,
            num_proc=torch.cuda.device_count()
        )

        dataset = datasets.concatenate_datasets([dataset, ds], axis=1).sort('_idx').remove_columns(['_idx'])

        dataset.save_to_disk(f"precompiled_dataset/{dataset_name.replace('/', '_')}_{subset_name}_{split_name}/train", max_shard_size="1GB")
        print(f"Processed dataset: {dataset_name}")
    #     all_datasets.append(dataset)
    
    # concatenated_dataset = datasets.concatenate_datasets(all_datasets)
    # print("All datasets concatenated successfully")

if __name__ == "__main__":
    main()