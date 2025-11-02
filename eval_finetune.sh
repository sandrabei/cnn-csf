#!/bin/bash
# Batch evaluation for 5 new finetune experiments

echo "Starting batch evaluation for 5 new finetune experiments..."

# v12_mse evaluation
echo "=== Evaluating v12_mse ==="
python3 eval.py v12_mse --batch_eval --data_list 135_other.list --point_num 4 --loss_type mse

# v16_gce_sigma1 evaluation
echo "=== Evaluating v16_gce_sigma1 ==="
python3 eval.py v16_gce_sigma1 --batch_eval --data_list 135_other.list --point_num 4 --loss_type gce --gce_q 0.4 --sigma 1.0

# v17_gce_sigma0.5 evaluation
echo "=== Evaluating v17_gce_sigma0.5 ==="
python3 eval.py v17_gce_sigma0.5 --batch_eval --data_list 135_other.list --point_num 4 --loss_type gce --gce_q 0.4 --sigma 0.5

# v18_gce_q0.2 evaluation
echo "=== Evaluating v18_gce_q0.2 ==="
python3 eval.py v18_gce_q0.2 --batch_eval --data_list 135_other.list --point_num 4 --loss_type gce --gce_q 0.2 --sigma 2.0

# v19_gce_q0.6 evaluation
echo "=== Evaluating v19_gce_q0.6 ==="
python3 eval.py v19_gce_q0.6 --batch_eval --data_list 135_other.list --point_num 4 --loss_type gce --gce_q 0.6 --sigma 2.0

echo "All evaluations completed!"
echo "Result files saved in *_batch_evaluation_results.csv under each experiment directory"