import gc
import os
import sys
import csv
import yaml
import json
import time
import datetime
import numpy as np
import psutil
import torch
import glob
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from bitsandbytes.optim import AdamW8bit

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def dir_setup(output_dir, train_config):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
        
    with open(output_dir + "/train_log.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch","step","loss","ppl","time"])
    with open(output_dir + "/train_config.yaml", "w") as yamlfile:
        yaml.dump(train_config, yamlfile, default_flow_style=False)

def log_memory_usage(accelerator, step):
    if accelerator.is_local_main_process:
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        cpu_memory = psutil.Process().memory_info().rss / 1024**3
        
        accelerator.print(f"Step {step}")
        accelerator.print(f"GPU Memory: {gpu_memory:.2f}GB")
        accelerator.print(f"CPU Memory: {cpu_memory:.2f}GB")
        
        # メモリ統計をリセット
        torch.cuda.reset_peak_memory_stats()

def main():
    # load config file
    train_config = read_yaml_file("train_config.yaml")

    # common config
    lr = float(train_config["parameter_config"]["common"]["lr"])
    num_epochs = int(train_config["parameter_config"]["common"]["epoch"])
    batch_size = int(train_config["parameter_config"]["common"]["batch_size"])
    cutoff_len = int(train_config["parameter_config"]["common"]["cutoff_len"])
    output_dir = train_config["directory_config"]["output"]
    logging_steps = train_config["logging_steps"]

    # make output dir and log files
    dir_setup(output_dir, train_config)

    # lora hyperparams
    lora = True
    lora_r = int(train_config["parameter_config"]["lora"]["r"])
    lora_alpha = int(train_config["parameter_config"]["lora"]["alpha"])
    lora_dropout = float(train_config["parameter_config"]["lora"]["drop_out"])
    lora_target_modules = train_config["parameter_config"]["lora"]["target_modules"]
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=True
    )

    # DeepSpeed ZeROの設定
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=train_config["parameter_config"]["common"]["gradient_accumulation_steps"],
        gradient_clipping=1.0,
        zero_stage=2,  # ZeRO-2を使用
        offload_optimizer_device="cpu",  # オプティマイザ状態をCPUにオフロード
        offload_param_device="cpu"  # パラメータをCPUにオフロード
    )

    # Modified accelerator configuration
    accelerator = Accelerator(
        gradient_accumulation_steps=train_config["parameter_config"]["common"].get("gradient_accumulation_steps", 1),
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=deepspeed_plugin,
        device_placement=True
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        train_config["directory_config"]["model"],
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_token_type_ids=False
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def collate_fn(examples):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for example in examples:
            for key in batch.keys():
                batch[key].append(example[key])
        
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

    # Load and preprocess dataset with memory optimization
    dataset = load_dataset('json', data_files=train_config["directory_config"]["dataset"])['train']
    
    if train_config["sample"] < 1:
        dataset = dataset.train_test_split(train_size=train_config["sample"])
        dataset = dataset["train"]

    if accelerator.is_local_main_process:
        print(f"Dataset size: {len(dataset)}")

    # Tokenize dataset with memory optimization
    def preprocess_function(examples):
        return tokenize(examples["text"])

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            preprocess_function,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            load_from_cache_file=False,
            batched=True,
            batch_size=100
        )
    
    # 最適化されたDataLoader
    train_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,  # 並列データローディング
        persistent_workers=True  # ワーカーを維持
    )

    # Modified model loading with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        train_config["directory_config"]["model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_flash_attention_2=True,
        max_memory=train_config["parameter_config"]["common"]["max_memory"]
    )

    if lora:
        model = get_peft_model(model, peft_config)
        if accelerator.is_local_main_process:
            model.print_trainable_parameters()

    # Enable gradient checkpointing
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
        model._set_static_graph()

    # Enable torch.compile if available
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # Disable cache
    model.config.use_cache = False

    # 8-bit AdamWオプティマイザを使用
    optimizer = AdamW8bit(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # prepare for distributed training
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    # Training loop with memory optimization
    if accelerator.is_local_main_process:
        with open(output_dir+"/train_log.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([0,0,0,0,datetime.datetime.now()])

    total_step = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        step_loss_avg = 0
        step_loss = 0
        pbar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        pbar.set_description(f'[Epoch {epoch}/{num_epochs}]')
        
        for step, batch in enumerate(pbar):
            # メモリ管理の改善
            if step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                log_memory_usage(accelerator, step)

            with accelerator.accumulate(model):
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix({"loss": loss.item()})
            
            if accelerator.is_local_main_process:
                if step % logging_steps == 0 and total_step != 0:
                    step_loss_avg = step_loss/logging_steps
                    step_ppl = torch.exp(step_loss_avg)
                    with open(output_dir+"/train_log.csv", 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([epoch, total_step, round(step_loss_avg.item(),3), 
                                       round(step_ppl.item(),3), datetime.datetime.now()])
                    step_loss = 0
            
            total_step += 1

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        # save checkpoint with memory optimization
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Clear cache before saving
        torch.cuda.empty_cache()
        gc.collect()
        
        unwrapped_model.save_pretrained(
            output_dir+f"/checkpoint-{epoch}",
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        if lora:
            accelerator.save_state(output_dir=output_dir+f"/checkpoint-{epoch}")

if __name__ == "__main__":
    main()