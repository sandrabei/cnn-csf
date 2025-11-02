#!/bin/bash
# Batch run 5 new finetune experiments

echo "Starting 5 new finetune experiments..."

# v12_mse experiment
echo "=== Starting v12_mse finetune experiment ==="
python3 train.py --finetune --train_list 135_tune.list --epochs 100 \
                   --batch_size 20 --exp_name v12_mse \
                   --resume outputs/v12_mse/best_model/best_model.pth \
                   --loss mse

# v16_gce_sigma1 experiment
echo "=== Starting v16_gce_sigma1 finetune experiment ==="
python3 train.py --finetune --train_list 135_tune.list --epochs 100 \
                   --batch_size 20 --exp_name v16_gce_sigma1 \
                   --resume outputs/v16_gce_sigma1/best_model/best_model.pth \
                   --loss gce --sigma 1.0 --gce_q 0.4

# v17_gce_sigma0.5 experiment
echo "=== Starting v17_gce_sigma0.5 finetune experiment ==="
python3 train.py --finetune --train_list 135_tune.list --epochs 100 \
                   --batch_size 20 --exp_name v17_gce_sigma0.5 \
                   --resume outputs/v17_gce_sigma0.5/best_model/best_model.pth \
                   --loss gce --sigma 0.5 --gce_q 0.4

# v18_gce_q0.2 experiment
echo "=== Starting v18_gce_q0.2 finetune experiment ==="
python3 train.py --finetune --train_list 135_tune.list --epochs 100 \
                   --batch_size 20 --exp_name v18_gce_q0.2 \
                   --resume outputs/v18_gce_q0.2/best_model/best_model.pth \
                   --loss gce --sigma 2.0 --gce_q 0.2

# v19_gce_q0.6 experiment
echo "=== Starting v19_gce_q0.6 finetune experiment ==="
python3 train.py --finetune --train_list 135_tune.list --epochs 100 \
                   --batch_size 20 --exp_name v19_gce_q0.6 \
                   --resume outputs/v19_gce_q0.6/best_model/best_model.pth \
                   --loss gce --sigma 2.0 --gce_q 0.6

echo "All finetune experiments completed!"