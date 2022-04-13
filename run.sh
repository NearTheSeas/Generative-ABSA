#!/usr/bin/env bash

python main.py --task aste \
            --dataset 14lap \
            --model_name_or_path t5-base \
            --paradigm prompt \
            --n_gpu 1 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 4 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 \
            --seed 20