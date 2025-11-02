# 3D Medical Image Processing Tool

A deep learning framework for 3D medical image processing and point detection using U-Net architecture.

## Data Preprocessing

Use `preprocess_data.py` to process raw 3D medical image data:

```bash
python3 preprocess_data.py raw_data data
```

- `raw_data/`: Directory containing raw 3D medical image datasets
- `data/`: Directory containing processed data

## Training and Evaluation

See the following shell scripts for detailed usage:

- `train.sh`: Model training
- `eval.sh`: Model evaluation
- `train_finetune.sh`: Fine-tuning experiments
- `eval_finetune.sh`: Fine-tuning evaluation

## Requirements

```bash
pip install numpy matplotlib Pillow scipy torch torchvision scikit-learn tqdm
```
