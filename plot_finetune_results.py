#!/usr/bin/env python3
"""
Plot finetune training results with Loss and IoU curves - Refactored version
Supports command line arguments and general configuration
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import argparse
from typing import Dict, List, Tuple, Optional


class ExperimentConfig:
    """Experiment configuration class for storing experiment metadata"""
    def __init__(self, name: str, loss_label: str, color: str = None,
                 fallback_data: Optional[Tuple[List[int], List[float], List[float]]] = None):
        self.name = name
        self.loss_label = loss_label
        self.color = color
        self.fallback_data = fallback_data


def load_evaluation_data_from_csv(exp_name: str) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[float]]]:
    """Read data from CSV evaluation result files"""
    possible_files = [
        f'outputs_finetune/{exp_name}/{exp_name}_batch_evaluation_results.csv',
        f'{exp_name}_batch_evaluation_results.csv',
        f'outputs_finetune/{exp_name}/batch_evaluation_results.csv',
    ]

    for filename in possible_files:
        if os.path.exists(filename):
            print(f"Reading data from CSV file: {filename}")
            epochs = []
            losses = []
            ious = []

            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    losses.append(float(row['loss']))
                    ious.append(float(row['iou']))

            if epochs:
                # Ensure data is sorted by epoch
                sorted_data = sorted(zip(epochs, losses, ious))
                epochs, losses, ious = zip(*sorted_data)
                return list(epochs), list(losses), list(ious)

    print(f"No CSV evaluation data file found for {exp_name}")
    return None, None, None


def plot_single_experiment(epochs: List[int], losses: List[float], ious: List[float],
                          config: ExperimentConfig, output_dir: str = ".") -> Tuple[float, float]:
    """Plot results for a single experiment"""

    # Find best points and key points
    best_loss_epoch = np.argmin(losses) + 1
    best_loss_value = min(losses)
    best_iou_epoch = np.argmax(ious) + 1
    best_iou_value = max(ious)

    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Finetune Training Results ({config.name}): Loss and IoU Over {len(epochs)} Epochs',
                fontsize=16, fontweight='bold')

    # Loss curve
    color = config.color if config.color else 'blue'
    ax1.plot(epochs, losses, '-', linewidth=2, label=config.loss_label, color=color)
    ax1.scatter([best_loss_epoch], [best_loss_value], color='red', s=100, zorder=5)
    ax1.annotate(f'Best Loss: {best_loss_value:.6f} (Epoch {best_loss_epoch})',
                 xy=(best_loss_epoch, best_loss_value),
                 xytext=(best_loss_epoch+10, best_loss_value*1.5),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')  # Use log scale because loss varies greatly

    # IoU curve
    ax2.plot(epochs, ious, '-', linewidth=2, label=f'IoU ({config.name})', color=color)
    ax2.scatter([best_iou_epoch], [best_iou_value], color='red', s=100, zorder=5)
    ax2.annotate(f'Best IoU: {best_iou_value:.4f} (Epoch {best_iou_epoch})',
                 xy=(best_iou_epoch, best_iou_value),
                 xytext=(best_iou_epoch+10, best_iou_value*0.8),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('IoU on Validation Set (135_other.list)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add key stage markers
    max_epoch = max(epochs)
    ax2.axvspan(1, max(10, max_epoch*0.1), alpha=0.1, color='blue', label='Early Stage')
    ax2.axvspan(max(10, max_epoch*0.1), max(40, max_epoch*0.4), alpha=0.1, color='green', label='Peak Performance')
    ax2.axvspan(max(40, max_epoch*0.4), max_epoch, alpha=0.1, color='red', label='Overfitting Stage')

    # Adjust layout
    plt.tight_layout()

    # Save image
    filename = f'{output_dir}/finetune_results_{config.name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as {filename}")

    # Display key statistics
    print(f"\n=== Finetune Results Summary ({config.name}) ===")
    print(f"Best Loss: {best_loss_value:.6f} at Epoch {best_loss_epoch}")
    print(f"Best IoU: {best_iou_value:.4f} ({best_iou_value*100:.2f}%) at Epoch {best_iou_epoch}")
    print(f"Initial Loss: {losses[0]:.6f}, Final Loss: {losses[-1]:.6f}")
    print(f"Initial IoU: {ious[0]:.4f} ({ious[0]*100:.2f}%), Final IoU: {ious[-1]:.4f} ({ious[-1]*100:.2f}%)")

    if losses[0] > 0:
        loss_improvement = ((losses[0] - losses[-1]) / losses[0] * 100)
        print(f"Loss Improvement: {loss_improvement:.2f}%")

    if ious[0] > 0:
        iou_improvement = ((ious[-1] - ious[0]) / ious[0] * 100)
        print(f"IoU Improvement: {iou_improvement:.2f}%")

    plt.close()
    return best_loss_value, best_iou_value


def plot_comparison(all_results: Dict[str, Dict], output_dir: str = "."):
    """Plot comparison of multiple experiments"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Finetune Experiments Comparison', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i, (exp_name, result) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        epochs = result['epochs']
        losses = result['losses']
        ious = result['ious']
        loss_label = result['config'].loss_label

        # Loss curve comparison
        ax1.plot(epochs, losses, '-', linewidth=2, label=f'{exp_name} ({loss_label})', color=color)

        # IoU curve comparison
        ax2.plot(epochs, ious, '-', linewidth=2, label=f'{exp_name} ({loss_label})', color=color)

    # Loss chart settings
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')

    # IoU chart settings
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('IoU Comparison on Validation Set (135_other.list)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save comparison chart
    filename = f'{output_dir}/finetune_results_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved as {filename}")
    plt.close()

    # Generate comparison table
    print(f"\n=== Experiment Comparison Summary ===")
    print(f"{'Experiment':<15} {'Best Loss':<12} {'Best IoU':<12} {'Loss Improve':<12} {'IoU Improve':<12}")
    print("-" * 65)

    for exp_name, result in all_results.items():
        losses = result['losses']
        ious = result['ious']
        best_loss = min(losses)
        best_iou = max(ious)

        loss_improvement = ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0
        iou_improvement = ((ious[-1] - ious[0]) / ious[0] * 100) if ious[0] > 0 else 0

        print(f"{exp_name:<15} {best_loss:<12.6f} {best_iou:<12.4f} {loss_improvement:<12.2f}% {iou_improvement:<12.2f}%")


def get_default_experiments() -> List[ExperimentConfig]:
    """Get default experiment configuration list"""

    return [
        ExperimentConfig("v15_focal", "Training Loss (Focal)", "blue"),
        ExperimentConfig("v14_bce", "Training Loss (BCE)", "red"),
        ExperimentConfig("v11", "Training Loss (GCE)", "green"),
        ExperimentConfig("v13_mae", "Training Loss (MAE)", "orange"),
        ExperimentConfig("v12_mse", "Training Loss (MSE)", "purple"),
        ExperimentConfig("v16_gce_sigma1", "Training Loss (GCE, σ=1.0)", "brown"),
        ExperimentConfig("v17_gce_sigma0.5", "Training Loss (GCE, σ=0.5)", "pink"),
        ExperimentConfig("v18_gce_q0.2", "Training Loss (GCE, q=0.2)", "gray"),
        ExperimentConfig("v19_gce_q0.6", "Training Loss (GCE, q=0.6)", "olive"),
    ]


def main():
    parser = argparse.ArgumentParser(description='Plot finetune training results charts')
    parser.add_argument('--experiments', nargs='+',
                       help='List of experiment names to plot, e.g.: v15_focal v14_bce v11')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory, default is current directory')
    parser.add_argument('--comparison', action='store_true',
                       help='Generate comparison chart for all experiments')
    parser.add_argument('--list-defaults', action='store_true',
                       help='List all available default experiment configurations')

    args = parser.parse_args()

    if args.list_defaults:
        print("=== Available default experiment configurations ===")
        for config in get_default_experiments():
            print(f"- {config.name}: {config.loss_label}")
        return

    # Get experiment configuration
    default_configs = get_default_experiments()
    config_dict = {config.name: config for config in default_configs}

    # Determine experiments to process
    if args.experiments:
        experiment_names = args.experiments
        # Validate experiment names
        invalid_names = [name for name in experiment_names if name not in config_dict]
        if invalid_names:
            print(f"Invalid experiment names: {invalid_names}")
            print("Use --list-defaults to view available experiment names")
            return
    else:
        # Default: process all configured experiments
        experiment_names = list(config_dict.keys())

    print(f"Experiments to process: {experiment_names}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each experiment
    all_results = {}

    for exp_name in experiment_names:
        config = config_dict[exp_name]
        print(f"\nProcessing experiment: {exp_name}")

        # Try to load data from CSV
        epochs, losses, ious = load_evaluation_data_from_csv(exp_name)

        if epochs is None:
            if config.fallback_data:
                print(f"Using built-in data")
                epochs, losses, ious = config.fallback_data
            else:
                print(f"Skipping experiment {exp_name} (no data)")
                continue

        # Plot single experiment chart
        best_loss, best_iou = plot_single_experiment(epochs, losses, ious, config, args.output_dir)

        all_results[exp_name] = {
            'epochs': epochs,
            'losses': losses,
            'ious': ious,
            'config': config,
            'best_loss': best_loss,
            'best_iou': best_iou
        }

    # Generate comparison chart
    if args.comparison and len(all_results) > 1:
        plot_comparison(all_results, args.output_dir)


if __name__ == "__main__":
    main()