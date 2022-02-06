#!/usr/bin/env bash

python main.py --task aste \
            --dataset laptop14 \
            --model_name_or_path t5-base \
            --paradigm annotation \
            --n_gpu 1 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 4 \
            --learning_rate 3e-4 \
            --num_train_epochs 5 \
            --seed 35