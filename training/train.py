import contextlib
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
from collections.abc import Iterable


import torch
from torch.utils.data import DataLoader

import datasets
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

from transformers import HfArgumentParser
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry

from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin
from accelerate.utils.memory import release_memory

from src import MimiConfig, MimiModel

from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)

from training.semantic_features import W2V2BertFeature, IndicConformerFeature
from training.data import load_multiple_datasets, DataCollatorMimiCodecWithPadding
from training.arguments import ModelArguments, DataArguments, MimiCodecTrainingArguments, DiscriminatorArguments

from training.discriminators import MultiScaleSTFTDiscriminator
from training.losses import AdversarialLoss, get_adv_criterion, get_fake_criterion, get_real_criterion, FeatureMatchingLoss, MelSpectrogramL1Loss

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser([ModelArguments, DataArguments, MimiCodecTrainingArguments])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32
    
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120)), DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_batch_size": training_args.per_device_batch_size,
            "global_batch_size": training_args.per_device_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "freeze_text_encoder": model_args.freeze_text_encoder,
            "max_duration_in_seconds": data_args.max_duration_in_seconds,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": model_args.temperature,
        },
        init_kwargs={"wandb": {"name": data_args.wandb_run_name}} if data_args.wandb_run_name else {},
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    num_workers = data_args.preprocessing_num_workers

    # 2. Now, let's load the dataset
    raw_datasets = DatasetDict()

    columns_to_keep = {
        "target_audio_column_name": data_args.target_audio_column_name,
        "prompt_column_name": data_args.prompt_column_name,
    }
    if data_args.description_column_name is not None:
        columns_to_keep["description_column_name"] = data_args.description_column_name

    if training_args.do_train:
        raw_datasets["train"] = load_multiple_datasets(
            accelerator,
            data_args.train_dataset_name,
            data_args.train_dataset_config_name,
            metadata_dataset_names=data_args.train_metadata_dataset_name,
            splits=data_args.train_split_name,
            dataset_samples=data_args.train_dataset_samples,
            seed=training_args.seed,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            id_column_name=data_args.id_column_name,
            columns_to_keep=columns_to_keep.values(),
            prompt_column_name=data_args.prompt_column_name,
            audio_column_name=data_args.target_audio_column_name,
            sampling_rate=data_args.sampling_rate,
            logger=logger,
            streaming=data_args.streaming, #TODO(SG): optionally enable streaming mode
        )

        for key in columns_to_keep:
            if columns_to_keep[key] not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--{key} '{columns_to_keep[key]}' not found in dataset '{data_args.train_dataset_name}'."
                    f" Make sure to set `--{key}` to the correct audio column - one of"
                    f" {', '.join(raw_datasets['train'].column_names)}."
                )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    
    if training_args.do_eval:
        raw_datasets["eval"] = load_multiple_datasets(
            accelerator,
            data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
            data_args.eval_dataset_config_name
            if data_args.eval_dataset_config_name
            else data_args.train_dataset_config_name,
            metadata_dataset_names=data_args.eval_metadata_dataset_name,
            splits=data_args.eval_split_name,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            id_column_name=data_args.id_column_name,
            columns_to_keep=columns_to_keep.values(),
            prompt_column_name=data_args.prompt_column_name,
            audio_column_name=data_args.target_audio_column_name,
            sampling_rate=data_args.sampling_rate,
            logger=logger,
            streaming=data_args.streaming, #TODO(SG): optionally enable streaming mode
        )

        if data_args.max_eval_samples is not None:
            with accelerator.local_main_process_first():
                raw_datasets["eval"] = (
                    raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
                )

    # 3. Next, let's load the config.
    config = MimiConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    model = MimiModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        _attn_implementation=model_args._attn_implementation
    )

    discriminator_config = DiscriminatorConfig.from_pretrained(
        model_args.discriminator_model_name_or_path
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    discriminator = MultiScaleSTFTDiscriminator.from_pretrained(
        model_args.discriminator_model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=discriminator_config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    adversarial_loss = AdversarialLoss(
        discriminator, 
        get_adv_criterion("hinge"),
        get_real_criterion("hinge"),
        get_fake_criterion("hinge"),
        FeatureMatchingLoss()
    )

    mel_loss = MelSpectrogramL1Loss(data_args.sampling_rate)

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Test all gather - used for warmout and avoiding timeout
    logger.debug(str(accelerator.process_index), main_process_only=False, in_order=True)
    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()

    with accelerator.local_main_process_first():
        # filter description that is shorter than max_text_length
        raw_datasets = raw_datasets.filter(
            lambda x: len(x["array"]) > data_args.min_duration_in_seconds,
            num_proc=num_workers,
            input_columns=[target_audio_column_name],
        )

    if "w2v-bert2.0" in model_args.ssl_model_name_or_path:
        ssl_model = W2V2BertFeature(model_args.ssl_model_name_or_path)
    elif "MahaDhwani_pretrained_conformer" in model_args.ssl_model_name_or_path:
        ssl_model = IndicConformerFeature(model_args.ssl_model_name_or_path)

    if training_args.torch_compile:
        ssl_model = accelerator.prepare_model(ssl_model, evaluation_mode=True)

    # Define Training Schedule
    # Store some constants
    per_device_batch_size = int(training_args.per_device_batch_size)
    train_batch_size = per_device_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(raw_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info(f"eval_steps is not set, evaluating at the end of each epoch")
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # autocast_kwargs = AutocastKwargs(enabled=(mixed_precision in ("fp16", "bf16")))

    # Define optimizer, LR scheduler, collator
    generator_optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        fused=True
    )

    discriminator_optimizer = torch.optim.AdamW(
        params=discriminator.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        fused=True
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    generator_lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=generator_optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps) * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )
    
    discriminator_lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=generator_optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps) * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    # Instantiate custom data collator
    data_collator = DataCollatorMimiCodecWithPadding(
        data_args.target_audio_column_name,
        max_audio_length=data_args.max_duration_in_seconds,
        sampling_rate=data_args.sampling_rate,
        semantic_feature_model=ssl_model
    )

    # Prepare everything with accelerate
    (
        model, 
        discriminator, 
        generator_lr_scheduler, 
        discriminator_optimizer, 
        generator_lr_scheduler, 
        discriminator_lr_scheduler
    ) = accelerator.prepare(
        model, 
        discriminator, 
        generator_lr_scheduler, 
        discriminator_optimizer, 
        generator_lr_scheduler, 
        discriminator_lr_scheduler
    )

    num_examples = total_train_steps * train_batch_size * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info("  Instantaneous batch size per device =" f" {per_device_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            api = HfApi(token=training_args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            with accelerator.local_main_process_first():
                raw_datasets["train"] = raw_datasets["train"].shuffle(training_args.seed)

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            with accelerator.local_main_process_first():
                raw_datasets["train"] = raw_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    def train_step_discriminator(batch, accelerator, generator_samples=None):
        with accelerator.autocast():
            if generator_samples is None:
                generator_samples = model(**batch)
            
            disc_loss = adversarial_loss.discriminator_fwd(generator_samples, batch['input_values'])

        return disc_loss
    
    def train_step_generator(batch, accelerator):
        with accelerator.autocast():
            generator_samples, quantized = model(**batch)

            mel_l1_loss = mel_loss(audio_batch, batch['input_values'])
            gen_loss, fm_loss = adversarial_loss.generator_fwd(generator_samples, batch['input_values'])

            gen_loss *= 4.0
            fm_loss *= 4.0
        
        return gen_loss, fm_loss, mel_l1_loss, generator_samples

    def _reduce(metrics):
        for key, value in metrics.items():
            if isinstance(value, Ierable):
                metrics[key] = sum(value) / len(value)
        return metrics

    @torch.no_grad()
    def eval_step(batch, accelerator):
        with accelerator.autocast():
            audio_batch, quantized = model(**batch)

            mel_l1_loss = mel_loss(audio_batch, batch['input_values'])

            # discriminator pass
            disc_loss = adversarial_loss.discriminator_fwd(audio_batch, batch['input_values'])

            # generator pass
            gen_loss, fm_loss = adversarial_loss.generator_fwd(audio_batch, batch['input_values'])

            gen_loss = 4.0 * gen_loss
            fm_loss = 4.0 * fm_loss

        return {
            "disc_loss": disc_loss,
            "gen_loss": gen_loss,
            "fm_loss": fm_loss,
            "mel_l1_loss": mel_l1_loss
        }

    model.train()
    discriminator.train()

    total_batched_samples = resume_step if resume_step is not None else 0
    for epoch in range(epochs_trained, num_epochs):
        with accelerator.local_main_process_first():
            raw_datasets["train"] = raw_datasets["train"].shuffle(training_args.seed)
        sampler = DistributedSampler(raw_datasets["train"])
        train_dataloader = DataLoader(
            raw_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_batch_size,
            sampler=sampler,
            shuffle=True,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            logger.info(f"  Skip first {resume_step} batches")
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None
            accelerator.wait_for_everyone()
        
        train_iterator = iter(train_dataloader)
        num_steps_in_epoch = len(train_dataloader)
        remainder = num_steps_in_epoch % gradient_accumulation_steps
        remainder = remainder if remainder != 0 else gradient_accumulation_steps
        total_updates = math.ceil(num_steps_in_epoch / gradient_accumulation_steps)
        
        update_step = -1
        for _ in range(total_updates):
            update_step += 1
            
            # preload the total batch per step
            batch_samples = []
            num_batches_in_step = gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(train_iterator)]

            # warm-up discriminator
            if update_step == 1:
                discriminator_optimizer.zero_grad()
                for i, batch in enumerate(batch_samples):
                    ctx = model.no_sync if (i < len(batch_samples) - 1 and accelerator.num_processes > 1) else contextlib.nullcontext
                    with ctx():
                        disc_loss = train_step_discriminator(batch, accelerator)
                        disc_loss = disc_loss / gradient_accumulation_steps
                        accelerator.backward(disc_loss)
                discriminator_optimizer.step()

            losses = {
                "gen_loss": [],
                "fm_loss": [],
                "mel_loss": [],
                "loss": [],
                "disc_loss": []
            }
            for i, batch in enumerate(batch_samples):
                total_batched_samples += 1
                ctx = model.no_sync if (i < len(batch_samples) - 1 and accelerator.num_processes > 1) else contextlib.nullcontext

                with ctx():
                    ## Generator
                    gen_loss, fm_loss, mel_l1_loss, generated_batch = train_step_generator(batch, accelerator)
                    loss = gen_loss + fm_loss
                    accelerator.backward(loss)

                    ## Discriminator
                    disc_loss = train_step_discriminator(batch, accelerator, generated_batch)
                    accelerator.backward(disc_loss)

                    # tracking
                    losses["gen_loss"].append(gen_loss.detach().item())
                    losses["fm_loss"].append(fm_loss.detach().item())
                    losses["mel_loss"].append(mel_loss.detach().item())
                    losses["loss"].append(loss.detach().item())
                    losses["disc_loss"].append(disc_loss.detach().item())

            discriminator_optimizer.step()
            discriminator_lr_scheduler.step()
            discriminator_optimizer.zero_grad()


            grad_norm = accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            generator_optimizer.step()
            generator_lr_scheduler.step()
            generator_optimizer.zero_grad()

            # The accelerator has performed an optimization step behind the scenes
            steps_trained_progress_bar.update(1)
            cur_step += 1
            
            if cur_step % training_args.logging_steps == 0:
                steps_trained_progress_bar.write(
                    f"Step... ({cur_step} / {total_train_steps} | Loss:"
                    f" {losses['loss']}, Learning Rate:"
                    f" {lr_scheduler.get_last_lr()[0]})"
                )
                losses["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                log_metric(
                    accelerator,
                    metrics=losses,
                    learning_rate=lr_scheduler.get_last_lr()[0],
                    train_time=train_time + time.time() - train_start,
                    step=cur_step,
                    epoch=epoch,
                    prefix="train",
                )

            # save checkpoint and weights after each save_steps and at the end of training
            if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                # safe_serialization=False to avoid shared tensors saving issue (TODO(YL): it's a temporary fix)
                # https://github.com/huggingface/transformers/issues/27293#issuecomment-1872560074
                accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    rotate_checkpoints(
                        training_args.save_total_limit, output_dir=training_args.output_dir, logger=logger
                    )

                    if cur_step == total_train_steps:
                        # un-wrap student model for save
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(training_args.output_dir)

                    if training_args.push_to_hub:
                        api.upload_folder(
                            repo_id=repo_id,
                            folder_path=training_args.output_dir,
                            commit_message=f"Saving train state of step {cur_step}",
                            run_as_future=True,
                        )
                accelerator.wait_for_everyone()
            
            if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                train_time += time.time() - train_start
                # ======================== Evaluating ==============================
                model.eval()
                discriminator.eval()
                eval_metrics = []
                eval_preds = []
                eval_descriptions = []
                eval_prompts = []
                eval_start = time.time()

                # release training input batch
                batch = release_memory(batch)

                validation_dataloader = DataLoader(
                    raw_datasets["eval"],
                    collate_fn=data_collator,
                    batch_size=per_device_eval_batch_size,
                    drop_last=False,
                    num_workers=training_args.eval_dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory,
                )
                validation_dataloader = accelerator.prepare(validation_dataloader)

                for batch in tqdm(
                    validation_dataloader,
                    desc=f"Evaluating - Inference ...",
                    position=2,
                    disable=not accelerator.is_local_main_process,
                ):
                    # Model forward
                    eval_metric = eval_step(batch, accelerator, autocast_kwargs)
                    eval_metric = accelerator.gather_for_metrics(eval_metric)
                    eval_metric = {key: val.unsqueeze(0) if val.ndim == 0 else val for (key,val) in eval_metric.items()}
                    eval_metrics.append(eval_metric)
                
                eval_time = time.time() - eval_start
                eval_metrics = {
                    key: torch.mean(torch.cat([d[key] for d in eval_metrics])).to("cpu") for key in eval_metrics[0]
                }

                # Print metrics and update progress bar
                if accelerator.is_local_main_process:
                    steps_trained_progress_bar.write(
                        f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                        f" {metrics_desc})"
                    )

                log_metric(
                    accelerator,
                    metrics=eval_metrics,
                    train_time=eval_time,
                    step=cur_step,
                    epoch=epoch,
                    prefix="eval",
                )

                # release eval batch and relax metrics
                eval_metrics, eval_preds, eval_descriptions, eval_prompts, batch, eval_metric = release_memory(
                    eval_metrics, eval_preds, eval_descriptions, eval_prompts, batch, eval_metric
                )

                model.train()
                discriminator.train()

                 train_start = time.time()

            # break condition
            if cur_step == total_train_steps:
                continue_training = False
                break

        if not continue_training:
            break

    accelerator.end_training()

if __name__ == "__main__":
    main()