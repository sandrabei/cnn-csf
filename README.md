# CNN U-Net for CSF

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

## Models

`test_data/best_model.pth` is the MAE model used in the paper. It is the recommended model for direct use or further finetuning.

## Use as a Library

The files located inside `cnn_csf` subdir are "library" source files. While the "py" files located inside root dir are original experimental codes used for the paper. 

Install the package:

```bash
pip install -e .
```

Run inference:

```python
from cnn_csf import inference
import numpy as np

# Input: (2, 64, 64) array with EPI and T1 channels
input_data = np.stack([epi_data, t1_data], axis=0)
heatmap = inference(input_data, checkpoint_path='model.pth')
```

Fine-tune on custom data:

```python
from cnn_csf import finetune

history = finetune(
    train_list_path='train.list',
    checkpoint_path='pretrained.pth',
    output_dir='outputs',
    epochs=50
)
```

See `examples/` for complete runnable demos.

## Requirements

```bash
pip install numpy matplotlib Pillow scipy torch torchvision scikit-learn tqdm
```
