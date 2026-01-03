#!/bin/bash
export PYENV_VERSION=py313
cd /Users/ckausik/Documents/GitHub/crossword-clue-generator
caffeinate -i python train_lora.py --max_length 128 2>&1 | tee training.log
