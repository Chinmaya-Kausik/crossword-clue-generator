#!/bin/bash

# Detect device
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    BATCH_SIZE=16
    GRAD_ACCUM=2
else
    DEVICE="mps"
    BATCH_SIZE=4
    GRAD_ACCUM=8
fi

python train_lora.py \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --max_length 128 \
    2>&1 | tee training.log
