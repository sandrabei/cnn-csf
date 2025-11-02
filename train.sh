python3 train.py --exp_name v11 --loss gce --sigma 2.0 --gce_q 0.4
python3 train.py --exp_name v12_mse --loss mse
python3 train.py --exp_name v13_mae --loss mae
python3 train.py --exp_name v14_bce --loss bce
python3 train.py --exp_name v15_focal --loss focal
python3 train.py --exp_name v16_gce_sigma1 --loss gce --sigma 1.0
python3 train.py --exp_name v17_gce_sigma0.5 --loss gce --sigma 0.5
python3 train.py --exp_name v18_gce_q0.2 --loss gce --gce_q 0.2
python3 train.py --exp_name v19_gce_q0.6 --loss gce --gce_q 0.6
