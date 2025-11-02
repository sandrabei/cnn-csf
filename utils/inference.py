"""Inference and result saving utilities"""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


def extract_peaks(heatmap, threshold=0.5, min_distance=3):
    """Extract peak points from heatmap"""
    heatmap = heatmap.squeeze()
    peaks = []
    
    # Find local maxima
    max_pool = nn.MaxPool2d(min_distance*2+1, stride=1, padding=min_distance)
    maxima = (heatmap == max_pool(heatmap.unsqueeze(0).unsqueeze(0))).squeeze()
    
    # Filter by threshold
    mask = (heatmap > threshold) & maxima
    
    # Get coordinates
    if mask.any():
        y_coords, x_coords = torch.where(mask)
        for x, y in zip(x_coords, y_coords):
            peaks.append((x.item(), y.item()))
    
    return peaks


def dump_inference_results(predictions, targets, output_dir, epoch, dataset_type, sample_ids=None):
    """
    Dump inference results with new directory structure: {dataset_type}/epoch{epoch}/{sample_id}.txt
    Also generates discrete version {sample_id}_post.txt with top 4 values as 1, others as 0
    
    Args:
        predictions: Tensor of predictions [batch_size, 1, H, W]
        targets: Tensor of ground truth [batch_size, 1, H, W]
        output_dir: Base directory to save results
        epoch: Current epoch number
        dataset_type: 'train' or 'val'
        sample_ids: List of original sample IDs (e.g., ['01000002', '01000003', ...])
    """
    if sample_ids is None:
        # Fallback to numeric IDs if no sample IDs provided
        sample_ids = [f"{i:08d}" for i in range(predictions.size(0))]
    
    # Create directory structure: {output_dir}/{dataset_type}/epoch{epoch}/
    epoch_dir = Path(output_dir) / dataset_type / f"epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = predictions.size(0)
    
    # Process each sample separately
    for batch_idx in range(batch_size):
        if batch_idx >= len(sample_ids):
            break
            
        sample_id = sample_ids[batch_idx]
        pred = predictions[batch_idx].squeeze()
        target = targets[batch_idx].squeeze()
        
        # Extract predictions (values from heatmap)
        pred_coords = []
        
        for y in range(pred.size(0)):
            for x in range(pred.size(1)):
                pred_coords.append((x, y, pred[y, x].item()))
        
        # Save prediction file with new naming format: {sample_id}.txt
        pred_file = epoch_dir / f"{sample_id}.txt"
        with open(pred_file, 'w') as f:
            for x, y, val in pred_coords:
                f.write(f"{x} {y} {val:.6f}\n")
        
        # Generate discrete version with top 4 values as 1, others as 0
        post_file = epoch_dir / f"{sample_id}_post.txt"
        
        # Sort by value to find top 4
        sorted_coords = sorted(pred_coords, key=lambda item: item[2], reverse=True)
        top_4_threshold = sorted_coords[3][2] if len(sorted_coords) >= 4 else sorted_coords[-1][2] if sorted_coords else 0
        
        # Write discrete version
        with open(post_file, 'w') as f:
            for x, y, val in pred_coords:
                discrete_val = 1 if val >= top_4_threshold else 0
                f.write(f"{x} {y} {discrete_val}\n")


def run_final_inference(model, val_dataset, device, inference_dir, final_epoch):
    """Run final validation set inference"""
    print("Running final inference on entire validation set...")
    
    # Set batch size to 1 for detailed per-sample results
    final_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                                   num_workers=0, pin_memory=False)
    
    model.eval()
    final_predictions = []
    final_targets = []
    final_sample_ids = []
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(tqdm(final_val_loader, desc="Final validation inference")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            final_predictions.append(outputs.cpu())
            final_targets.append(targets.cpu())
            
            # Get sample ID from dataset
            data_path = val_dataset.data_list.iloc[idx]['output']
            sample_id = Path(data_path).parent.name
            final_sample_ids.append(sample_id)
    
    # Concatenate all predictions and targets
    if final_predictions:
        all_predictions = torch.cat(final_predictions, dim=0)
        all_targets = torch.cat(final_targets, dim=0)
        
        # Save final validation results
        final_output_dir = inference_dir / 'final'
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        dump_inference_results(all_predictions, all_targets, str(final_output_dir), 
                             final_epoch, "val", final_sample_ids)
        
        print(f"Final validation inference completed for {len(final_sample_ids)} samples")
        return len(final_sample_ids)
    return 0