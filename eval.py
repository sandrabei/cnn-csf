#!/usr/bin/env python3
"""
Evaluation script - Use best model for inference and evaluation on validation set
"""

import argparse
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os

from utils.config import setup_device, setup_model_and_optimizer, setup_loss_function
from utils.inference import dump_inference_results
from dataset import MedicalImageDataset
import csv
from datetime import datetime


def load_ground_truth(gt_file):
    """Load ground truth point coordinates"""
    points = []
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = int(parts[0]), int(parts[1])
                        # Strictly process floating point values
                        try:
                            value = float(parts[2])
                            # Strict judgment: <=0.01 treated as 0, >=0.99 treated as 1, other values error
                            if value <= 0.01:
                                value = 0
                            elif value >= 0.99:
                                value = 1
                            else:
                                raise ValueError(f"Value {value} not in valid range [0,1]")
                            value = int(value)
                        except ValueError as e:
                            if "not in valid range" in str(e):
                                raise e
                            continue
                        if value == 1:
                            points.append((x, y))
    except Exception as e:
        print(f"Error loading {gt_file}: {e}")
    
    return set(points)


def calculate_iou(pred_points, gt_points):
    """Calculate IoU (Intersection/Union)"""
    if not pred_points and not gt_points:
        return 1.0
    if not pred_points or not gt_points:
        return 0.0
    
    intersection = len(pred_points & gt_points)
    union = len(pred_points | gt_points)
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(heatmap, threshold=0.1, kernel_size=3):
    """
    Non-maximum suppression
    Args:
        heatmap: 2D numpy array or tensor
        threshold: Suppression threshold, points below this value will be ignored
        kernel_size: Suppression window size
    Returns:
        Suppressed numpy array
    """
    import torch
    
    # Convert to numpy array for processing
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.squeeze().cpu().numpy()
    else:
        heatmap_np = heatmap.squeeze()
    
    h, w = heatmap_np.shape
    suppressed = heatmap_np.copy()
    
    # Ensure kernel_size is odd
    kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    radius = kernel_size // 2
    
    for y in range(h):
        for x in range(w):
            if heatmap_np[y, x] < threshold:
                suppressed[y, x] = 0
                continue
                
            # Check if it's a local maximum
            is_max = True
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if heatmap_np[ny, nx] > heatmap_np[y, x]:
                            is_max = False
                            break
                if not is_max:
                    break
            
            if not is_max:
                suppressed[y, x] = 0
                
    return suppressed


def extract_top_k_points(heatmap, k, use_nms=False, nms_threshold=0.1, nms_kernel_size=3):
    """Extract top-k point coordinates from heatmap, optionally using NMS"""
    heatmap = heatmap.squeeze()
    
    if use_nms:
        heatmap = non_max_suppression(heatmap, nms_threshold, nms_kernel_size)
    
    h, w = heatmap.shape
    
    # Get all coordinates and values
    coords = []
    for y in range(h):
        for x in range(w):
            coords.append((x, y, heatmap[y, x]))

    # Sort by value, take top-k
    coords.sort(key=lambda x: x[2], reverse=True)
    top_k_coords = coords[:k]
    
    return set((x, y) for x, y, _ in top_k_coords)


def calculate_iou_for_results(output_dir, val_dataset, sample_ids, point_num, use_nms, nms_threshold, nms_kernel_size):
    """Calculate IoU for given inference results"""
    print("Calculating IoU...")
    iou_scores = []

    for idx, sample_id in enumerate(tqdm(sample_ids, desc="Calculating IoU")):
        # Load inference results
        pred_file = output_dir / f"{sample_id}.txt"

        # Load predicted heatmap
        heatmap = np.zeros((64, 64))
        with open(pred_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = int(parts[0]), int(parts[1])
                        val = float(parts[2])
                        if 0 <= x < 64 and 0 <= y < 64:
                            heatmap[y, x] = val

        # Extract predicted points
        pred_points = extract_top_k_points(heatmap, point_num,
                                         use_nms=use_nms,
                                         nms_threshold=nms_threshold,
                                         nms_kernel_size=nms_kernel_size)

        # Load ground truth points
        data_path = val_dataset.data_list.iloc[idx]['output']
        gt_file = Path(data_path)
        gt_points = load_ground_truth(gt_file)

        # Calculate IoU
        iou = calculate_iou(pred_points, gt_points)
        iou_scores.append(iou)

        # Print individual sample info
        # print(f"Sample {sample_id}: {len(pred_points)} predicted points, {len(gt_points)} GT points, IoU: {iou:.3f}")

    # Calculate average IoU
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    return avg_iou


def process_single_checkpoint(args):
    """Function to process single checkpoint, used for multiprocessing"""
    (checkpoint_path, epoch, output_base_dir, val_dataset_path, sample_ids,
     point_num, use_nms, nms_threshold, nms_kernel_size, loss_type, gce_q, sigma, device_id) = args

    # Set up device
    device = setup_device(device_id)

    # Set up loss function
    loss_kwargs = {'focal_alpha': 0.25, 'focal_gamma': 2.0}
    if loss_type.lower() == 'gce':
        loss_kwargs['gce_q'] = gce_q
    criterion = setup_loss_function(loss_type, **loss_kwargs)

    # Load dataset
    val_dataset = MedicalImageDataset(val_dataset_path, sigma=sigma, transform=None, device=str(device), cache_dir='data_cache')

    # Set up single evaluation output directory
    epoch_name = str(epoch)
    epoch_output_dir = output_base_dir / f"epoch_{epoch_name}"
    epoch_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Inference and calculate loss
        model, _ = setup_model_and_optimizer(device, 1e-3)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        all_predictions = []
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_samples += 1
                all_predictions.append(outputs.cpu())

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

        # Save inference results
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            for batch_idx in range(len(sample_ids)):
                sample_id = sample_ids[batch_idx]
                pred = all_predictions[batch_idx].squeeze()

                # Save original results
                pred_coords = [(x, y, pred[y, x].item()) for y in range(pred.size(0)) for x in range(pred.size(1))]
                pred_file = epoch_output_dir / f"{sample_id}.txt"
                with open(pred_file, 'w') as f:
                    for x, y, val in pred_coords:
                        f.write(f"{x} {y} {val:.6f}\n")

                # Save discretized results
                post_file = epoch_output_dir / f"{sample_id}_post.txt"
                sorted_coords = sorted(pred_coords, key=lambda item: item[2], reverse=True)
                top_k_threshold = sorted_coords[point_num-1][2] if len(sorted_coords) >= point_num else 0
                with open(post_file, 'w') as f:
                    for x, y, val in pred_coords:
                        f.write(f"{x} {y} {1 if val >= top_k_threshold else 0}\n")

        # Calculate IoU
        avg_iou = calculate_iou_for_results(epoch_output_dir, val_dataset, sample_ids, point_num, use_nms, nms_threshold, nms_kernel_size)

        return {
            'epoch': epoch,
            'loss': avg_loss,
            'iou': avg_iou,
            'num_samples': num_samples,
            'success': True
        }
    except Exception as e:
        print(f"Error processing Epoch {epoch}: {str(e)}")
        return {
            'epoch': epoch,
            'loss': float('inf'),
            'iou': 0.0,
            'num_samples': 0,
            'success': False,
            'error': str(e)
        }


def run_evaluation(exp_name, point_num=4, data_list="val.list", use_nms=False,
                   nms_threshold=0.1, nms_kernel_size=3, use_best=True, force=False,
                   batch_eval=False, loss_type='focal', gce_q=0.4, sigma=2.0, num_workers=8):
    """Run evaluation"""
    # Determine experiment directory based on batch_eval
    if batch_eval:
        exp_dir = Path("outputs_finetune") / exp_name
    else:
        exp_dir = Path("outputs") / exp_name
    
    # Set up dataset
    data_list_name = Path(data_list).stem  # Extract base name from filename
    print(f"Loading {data_list_name} set...")
    val_dataset = MedicalImageDataset(data_list, device=torch.device('cpu'))  # First load data using CPU
    
    # Create output directory (includes model type and NMS information)
    model_suffix = "best" if use_best else "last"
    nms_suffix = f"_nms{nms_threshold:.1f}" if use_nms else ""
    output_dir = exp_dir / "inference_results" / data_list_name / f"{model_suffix}_{point_num}{nms_suffix}"
    
    # Check if inference results already exist
    all_files_exist = True
    sample_ids = []

    # First collect all sample IDs
    for idx in range(len(val_dataset)):
        data_path = val_dataset.data_list.iloc[idx]['output']
        sample_id = Path(data_path).parent.name
        sample_ids.append(sample_id)

    # Then check which files exist (unless force mode)
    missing_files = []
    if not force:
        for sample_id in sample_ids:
            pred_file = output_dir / f"{sample_id}.txt"
            post_file = output_dir / f"{sample_id}_post.txt"

            if not pred_file.exists() or not post_file.exists():
                all_files_exist = False
                missing_files.append(sample_id)
    else:
        all_files_exist = False
        missing_files = sample_ids.copy()

    # If all inference result files exist and not force mode, load directly
    if all_files_exist and len(sample_ids) > 0 and not force:
        print(f"Detected existing inference results, skipping inference step...")
        print(f"Results directory: {output_dir}")
    else:
        # Need to re-run inference
        if batch_eval:
            # Batch evaluation mode: evaluate all checkpoints
            checkpoint_dir = exp_dir / "checkpoints"
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

            if not checkpoint_files:
                print(f"Error: No checkpoint files found in {checkpoint_dir}")
                return

            print(f"Found {len(checkpoint_files)} checkpoint files, starting batch evaluation...")

            # Set up device and loss function
            device = setup_device(None)
            loss_kwargs = {'focal_alpha': 0.25, 'focal_gamma': 2.0}
            if loss_type.lower() == 'gce':
                loss_kwargs['gce_q'] = gce_q
            criterion = setup_loss_function(loss_type, **loss_kwargs)

            # Evaluation results list
            evaluation_results = []

            # Get num_workers parameter
            num_workers = min(num_workers, len(checkpoint_files))  # Use at most specified number of processes
            print(f"Using {num_workers} processes for parallel evaluation...")

            # Prepare multiprocessing parameters
            args_list = []
            for checkpoint_path in checkpoint_files:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                epoch = checkpoint.get('epoch', 0) + 1
                epoch_name = checkpoint_path.stem.replace("checkpoint_epoch_", "")

                args_list.append((
                    checkpoint_path, epoch, output_dir.parent, data_list,
                    sample_ids, point_num, use_nms, nms_threshold,
                    nms_kernel_size, loss_type, gce_q, sigma, None  # device_id will be set in subprocess
                ))

            # Use multiprocessing processing
            with mp.Pool(processes=num_workers) as pool:
                results = []
                for result in tqdm(pool.imap_unordered(process_single_checkpoint, args_list),
                                 total=len(args_list), desc="Batch evaluation"):
                    results.append(result)
                    if result['success']:
                        print(f"Epoch {result['epoch']}: Loss={result['loss']:.6f}, IoU={result['iou']:.4f}")
                    else:
                        print(f"Epoch {result['epoch']} processing failed: {result.get('error', 'Unknown error')}")

            # Collect successful results
            evaluation_results = [r for r in results if r['success']]

            # Output summary
            evaluation_results.sort(key=lambda x: x['epoch'])
            print(f"\n=== Batch Evaluation Summary ===")
            print(f"Number of evaluated checkpoints: {len(evaluation_results)}")
            print(f"Discretization points: {point_num}")
            print(f"Loss function: {loss_type}")

            if evaluation_results:
                best_loss_result = min(evaluation_results, key=lambda x: x['loss'])
                best_iou_result = max(evaluation_results, key=lambda x: x['iou'])
                print(f"\nBest Loss (Epoch {best_loss_result['epoch']}): {best_loss_result['loss']:.6f}")
                print(f"Best IoU (Epoch {best_iou_result['epoch']}): {best_iou_result['iou']:.6f}")

                print(f"\n=== All Epoch Results ===")
                print("Epoch | Loss      | IoU       ")
                print("-" * 30)
                for result in evaluation_results:
                    print(f"{result['epoch']:5d} | {result['loss']:.6f} | {result['iou']:.6f}")

                # Save results to CSV file
                csv_filename = exp_dir / f"{exp_name}_batch_evaluation_results.csv"
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['epoch', 'loss', 'iou', 'num_samples']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for result in evaluation_results:
                        # Only write specified fields, filter out other fields like 'success'
                        filtered_result = {key: result[key] for key in fieldnames if key in result}
                        writer.writerow(filtered_result)

                print(f"\nEvaluation results saved to CSV file: {csv_filename}")

            return
        else:
            # Single model evaluation mode
            if use_best:
                model_path = exp_dir / "best_model" / "best_model.pth"
                model_type = "best"
            else:
                model_path = exp_dir / "checkpoints" / "latest_checkpoint.pth"
                model_type = "latest"
        
        if not model_path.exists():
            print(f"Error: {model_type} model file does not exist: {model_path}")
            return
        
        # Set up device
        device = setup_device(None)
        
        # Load model
        model, _ = setup_model_and_optimizer(device, 1e-3)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Process checkpoint format (check if it's full checkpoint or just model weights)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Create data loader
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using {model_type} model for inference, output directory: {output_dir}")
        if use_nms:
            print(f"NMS enabled: threshold={nms_threshold}, kernel_size={nms_kernel_size}")
        
        # Collect all predictions
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Inference")):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Save inference results
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            batch_size = all_predictions.size(0)
            
            for batch_idx in range(batch_size):
                if batch_idx >= len(sample_ids):
                    break
                    
                sample_id = sample_ids[batch_idx]
                pred = all_predictions[batch_idx].squeeze()
                
                # Extract all coordinates and values
                pred_coords = []
                for y in range(pred.size(0)):
                    for x in range(pred.size(1)):
                        pred_coords.append((x, y, pred[y, x].item()))
                
                # Save original inference results
                pred_file = output_dir / f"{sample_id}.txt"
                with open(pred_file, 'w') as f:
                    for x, y, val in pred_coords:
                        f.write(f"{x} {y} {val:.6f}\n")
                
                # Save discretized results (top-k)
                post_file = output_dir / f"{sample_id}_post.txt"
                
                # Sort by value, take top-k
                sorted_coords = sorted(pred_coords, key=lambda item: item[2], reverse=True)
                top_k_threshold = sorted_coords[point_num-1][2] if len(sorted_coords) >= point_num else (sorted_coords[-1][2] if sorted_coords else 0)
                
                # Write discretized results
                with open(post_file, 'w') as f:
                    for x, y, val in pred_coords:
                        discrete_val = 1 if val >= top_k_threshold else 0
                        f.write(f"{x} {y} {discrete_val}\n")
    
    # Calculate IoU (calculate regardless of whether inference was re-run)
    avg_iou = calculate_iou_for_results(output_dir, val_dataset, sample_ids, point_num, use_nms, nms_threshold, nms_kernel_size)

    if avg_iou is not None:
        print(f"\n=== Evaluation Results ===")
        print(f"Experiment: {exp_name}")
        print(f"Model type: {'best' if use_best else 'latest'}")
        print(f"Discretization points: {point_num}")
        if use_nms:
            print(f"NMS: enabled (threshold={nms_threshold}, kernel_size={nms_kernel_size})")
        else:
            print(f"NMS: disabled")
        print(f"{data_list_name} samples: {len(sample_ids)}")
        print(f"Average IoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
        print(f"Results saved to: {output_dir}")
        
        return avg_iou
    else:
        print("Error: No samples were successfully processed")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='Evaluate dataset using best model')
    parser.add_argument('exp_name', help='Experiment name (e.g. gce)')
    parser.add_argument('--point_num', type=int, default=4, choices=[3, 4],
                       help='Top-k points for discretization, default 4')
    parser.add_argument('--data_list', type=str, default='val.list',
                       help='Data list file to use, default val.list')
    parser.add_argument('--use_nms', action='store_true',
                       help='Enable non-maximum suppression (NMS), disabled by default')
    parser.add_argument('--nms_threshold', type=float, default=0.1,
                       help='NMS threshold, default 0.1')
    parser.add_argument('--nms_kernel_size', type=int, default=3,
                       help='NMS kernel size, default 3')
    parser.add_argument('--model', type=str, choices=['best', 'last'], default='best',
                       help='Model type to use: best (best model) or last (latest model), default best')
    parser.add_argument('--force', action='store_true',
                       help='Force re-inference and overwrite existing results, no overwrite by default')
    parser.add_argument('--batch_eval', action='store_true',
                       help='Batch evaluation mode: evaluate all checkpoints, default single model evaluation')
    parser.add_argument('--loss_type', type=str, default='focal',
                       help='Loss function type (used in batch evaluation), default focal')
    parser.add_argument('--gce_q', type=float, default=0.4,
                       help='GCE loss q parameter, default 0.4')
    parser.add_argument('--sigma', type=float, default=2.0,
                       help='Gaussian smoothing sigma parameter, default 2.0')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of processes for batch evaluation, default 8')

    args = parser.parse_args()

    run_evaluation(args.exp_name, args.point_num, args.data_list,
                   use_nms=args.use_nms,
                   nms_threshold=args.nms_threshold,
                   nms_kernel_size=args.nms_kernel_size,
                   use_best=(args.model == 'best'),
                   force=args.force,
                   batch_eval=args.batch_eval,
                   loss_type=args.loss_type,
                   gce_q=args.gce_q,
                   sigma=args.sigma,
                   num_workers=args.num_workers)


if __name__ == '__main__':
    main()
