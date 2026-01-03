"""
Fine-tune Qwen 3 0.6B using LoRA for crossword clue generation.
Uses PyTorch with MPS (Apple Silicon) backend.
"""

import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    torch_dtype: str = "float32"  # "float32", "float16", "bfloat16"

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lora_bias: str = "none"  # "none", "all", "lora_only"

    # Data
    train_data_path: str = "data/train_final.parquet"
    val_data_path: str = "data/val_final.parquet"
    max_length: int = 256
    cache_dir: str = "tokenized_cache"
    tokenize_batch_size: int = 1000

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # DataLoader
    num_workers: int = 0  # MPS doesn't support multiprocessing well
    pin_memory: bool = False

    # Logging & Checkpointing
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    output_dir: str = "checkpoints"

    # Device
    device: str = "mps"

class CrosswordDataset(Dataset):
    """Dataset for crossword clue generation with disk-cached tokenization."""

    PROMPT_TEMPLATE = """Generate a cryptic crossword clue for the following:
Answer: {answer}
Enumeration: {enumeration}

Clue:"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
        cache_dir: str = "tokenized_cache",
        tokenize_batch_size: int = 1000,
    ):
        self.max_length = max_length
        self.tokenize_batch_size = tokenize_batch_size

        # Determine cache path based on data file and max_length
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        cache_path = os.path.join(cache_dir, f"{data_name}_ml{max_length}.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading tokenized cache from {cache_path}")
            cache = torch.load(cache_path)
            self.input_ids = cache["input_ids"]
            self.attention_mask = cache["attention_mask"]
            self.labels = cache["labels"]
        else:
            logger.info(f"Cache not found. Tokenizing {data_path}...")
            df = pd.read_parquet(data_path)
            self.input_ids, self.attention_mask, self.labels = self._tokenize_dataset(
                df, tokenizer
            )
            # Save cache
            os.makedirs(cache_dir, exist_ok=True)
            torch.save({
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "labels": self.labels,
            }, cache_path)
            logger.info(f"Saved tokenized cache to {cache_path}")

        logger.info(f"Dataset ready. Shape: {self.input_ids.shape}")

    def _tokenize_dataset(self, df, tokenizer):
        """Tokenize entire dataset in batches."""
        prompts = [
            self.PROMPT_TEMPLATE.format(answer=row["answer"], enumeration=row["enumeration"])
            for _, row in df.iterrows()
        ]
        full_texts = [
            prompt + " " + str(clue) + tokenizer.eos_token
            for prompt, clue in zip(prompts, df["clue"])
        ]

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for i in tqdm(range(0, len(full_texts), self.tokenize_batch_size), desc="Tokenizing"):
            batch_prompts = prompts[i : i + self.tokenize_batch_size]
            batch_texts = full_texts[i : i + self.tokenize_batch_size]

            # Batch tokenize full texts
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Batch tokenize prompts to get lengths
            prompt_encodings = tokenizer(
                batch_prompts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            # Create labels with prompt masking
            labels = encodings["input_ids"].clone()
            for j, prompt_len in enumerate(prompt_lengths):
                labels[j, :prompt_len] = -100
            labels[encodings["attention_mask"] == 0] = -100

            all_input_ids.append(encodings["input_ids"])
            all_attention_mask.append(encodings["attention_mask"])
            all_labels.append(labels)

        return (
            torch.cat(all_input_ids, dim=0),
            torch.cat(all_attention_mask, dim=0),
            torch.cat(all_labels, dim=0),
        )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def evaluate(model, val_dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Count non-masked tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {"val_loss": avg_loss, "val_perplexity": perplexity}


def save_checkpoint(model, tokenizer, optimizer, scheduler, step, config, metrics):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save LoRA weights only
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save training state
    state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def train(config: TrainingConfig):
    """Main training function."""

    # Set up device
    if config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) backend")
    elif config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA backend")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU backend")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from {config.model_name}")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype_map[config.torch_dtype],
        trust_remote_code=True,
    )

    # Configure LoRA
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias=config.lora_bias,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = CrosswordDataset(
        config.train_data_path,
        tokenizer,
        config.max_length,
        config.cache_dir,
        config.tokenize_batch_size,
    )
    val_dataset = CrosswordDataset(
        config.val_data_path,
        tokenizer,
        config.max_length,
        config.cache_dir,
        config.tokenize_batch_size,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    logger.info(f"Total training steps: {total_training_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Set up scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training metrics
    metrics_log = []
    global_step = 0
    total_loss = 0
    best_val_loss = float("inf")

    logger.info("Starting training")
    model.train()

    for epoch in range(config.num_epochs):
        epoch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        )

        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / config.gradient_accumulation_steps
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log training metrics
                if global_step % config.log_every_n_steps == 0:
                    avg_loss = total_loss / config.log_every_n_steps
                    current_lr = scheduler.get_last_lr()[0]

                    metrics = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "train_perplexity": math.exp(avg_loss) if avg_loss < 100 else float("inf"),
                        "learning_rate": current_lr,
                    }

                    logger.info(
                        f"Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {metrics['train_perplexity']:.2f} | "
                        f"LR: {current_lr:.2e}"
                    )

                    metrics_log.append(metrics)
                    total_loss = 0

                # Evaluate on validation set
                if global_step % config.eval_every_n_steps == 0:
                    val_metrics = evaluate(model, val_dataloader, device)

                    logger.info(
                        f"Validation | "
                        f"Loss: {val_metrics['val_loss']:.4f} | "
                        f"PPL: {val_metrics['val_perplexity']:.2f}"
                    )

                    # Add to metrics log
                    metrics_log[-1].update(val_metrics)

                    # Save best model
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        best_checkpoint_dir = os.path.join(config.output_dir, "best")
                        os.makedirs(best_checkpoint_dir, exist_ok=True)
                        model.save_pretrained(best_checkpoint_dir)
                        tokenizer.save_pretrained(best_checkpoint_dir)
                        logger.info(f"New best model saved to {best_checkpoint_dir}")

                # Save checkpoint
                if global_step % config.save_every_n_steps == 0:
                    save_checkpoint(
                        model, tokenizer, optimizer, scheduler,
                        global_step, config, metrics_log[-1]
                    )

                # Update progress bar
                epoch_iterator.set_postfix(
                    loss=f"{outputs.loss.item():.4f}",
                    step=global_step,
                )

    # Final evaluation
    logger.info("Running final evaluation")
    final_metrics = evaluate(model, val_dataloader, device)
    logger.info(
        f"Final Validation | "
        f"Loss: {final_metrics['val_loss']:.4f} | "
        f"PPL: {final_metrics['val_perplexity']:.2f}"
    )

    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")

    # Save metrics log
    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics_log


def generate_sample(
    model,
    tokenizer,
    answer,
    enumeration,
    device,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate a sample clue for testing."""
    prompt = CrosswordDataset.PROMPT_TEMPLATE.format(
        answer=answer,
        enumeration=enumeration,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clue = generated[len(prompt):].strip()

    return clue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen 3 0.6B with LoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_data", type=str, default="data/train_final.parquet")
    parser.add_argument("--val_data", type=str, default="data/val_final.parquet")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        log_every_n_steps=args.log_every,
        eval_every_n_steps=args.eval_every,
        save_every_n_steps=args.save_every,
        device=args.device,
    )

    train(config)
