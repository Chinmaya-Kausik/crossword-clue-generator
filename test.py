"""
Quick test of the training pipeline on a small subset.
"""

import os
import torch
import pandas as pd

# Create small test datasets
print("Creating small test datasets...")
train_df = pd.read_parquet("data/train_final.parquet").head(1000)
val_df = pd.read_parquet("data/val_final.parquet").head(200)

os.makedirs("data/test_subset", exist_ok=True)
train_df.to_parquet("data/test_subset/train_small.parquet")
val_df.to_parquet("data/test_subset/val_small.parquet")
print(f"Created train subset: {len(train_df)} examples")
print(f"Created val subset: {len(val_df)} examples")

# Run training with minimal settings
from train_lora import TrainingConfig, train

config = TrainingConfig(
    model_name="Qwen/Qwen3-0.6B",
    train_data_path="data/test_subset/train_small.parquet",
    val_data_path="data/test_subset/val_small.parquet",
    output_dir="checkpoints_test",
    batch_size=2,
    gradient_accumulation_steps=2,
    num_epochs=1,
    max_length=128,
    log_every_n_steps=10,
    eval_every_n_steps=50,
    save_every_n_steps=100,
    device="mps",
)

print("\nStarting test training run...")
train(config)
print("\nTest complete!")
