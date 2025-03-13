#!/usr/bin/env python3
"""
A script to plot training, validation, and test losses from a JSONL training output file.
Handles validation metrics that may be output at different frequencies than training metrics.
"""

import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def read_jsonl(file_path):
    """Read a jsonl file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line.strip()}")
    return data

def extract_metrics(data, use_epoch=False):
    """
    Extract training, validation, and test metrics from the data.
    
    Args:
        data: List of dictionaries from the jsonl file
        use_epoch: If True, use epoch numbers for x-axis instead of steps
    """
    train_x = []
    train_loss = []
    val_x = []
    val_loss = []
    test_x = []
    test_loss = []
    
    for entry in data:
        # Get x-axis value (epoch or step)
        if use_epoch and 'epoch' in entry:
            x_value = entry['epoch']
        elif 'step' in entry:
            x_value = entry['step']
        else:
            continue
        
        # Extract training loss
        if 'train/loss' in entry:
            train_x.append(x_value)
            train_loss.append(entry['train/loss'])
        
        # Extract validation loss
        if 'val/loss' in entry:
            val_x.append(x_value)
            val_loss.append(entry['val/loss'])
        
        # Extract test loss
        if 'test/loss' in entry:
            test_x.append(x_value)
            test_loss.append(entry['test/loss'])
    
    return {
        'train': {'x': train_x, 'loss': train_loss},
        'val': {'x': val_x, 'loss': val_loss},
        'test': {'x': test_x, 'loss': test_loss}
    }

def plot_losses(metrics, output_path=None, smooth=1, use_epoch=False, include_test=True):
    """
    Plot training, validation, and optionally test losses.
    
    Args:
        metrics: Dictionary with metrics for train, val, and test
        output_path: Path to save the figure (optional)
        smooth: Smoothing factor for the training curve (1 means no smoothing)
        use_epoch: Whether the x-axis represents epochs (for label)
        include_test: Whether to include test metrics in the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training loss with optional smoothing
    if metrics['train']['x']:
        x_values = metrics['train']['x']
        y_values = metrics['train']['loss']
        
        if smooth > 1 and len(y_values) > smooth:
            # Apply smoothing window
            smoothed_loss = []
            for i in range(len(y_values)):
                start = max(0, i - smooth // 2)
                end = min(len(y_values), i + smooth // 2 + 1)
                smoothed_loss.append(sum(y_values[start:end]) / (end - start))
            plt.plot(x_values, smoothed_loss, label='Training Loss', color='blue')
        else:
            plt.plot(x_values, y_values, label='Training Loss', color='blue')
    
    # Plot validation loss
    if metrics['val']['x']:
        plt.plot(metrics['val']['x'], metrics['val']['loss'], 
                 label='Validation Loss', color='red', marker='o')
    
    # Plot test loss
    if include_test and metrics['test']['x']:
        plt.plot(metrics['test']['x'], metrics['test']['loss'], 
                 label='Test Loss', color='green', marker='s')
    
    plt.xlabel('Epoch' if use_epoch else 'Step')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure y-axis starts from 0 or slightly below the minimum
    min_loss = float('inf')
    for dataset in metrics.values():
        if dataset['loss']:
            min_loss = min(min_loss, min(dataset['loss']))
    
    y_min = max(0, min_loss * 0.9)  # Start slightly below the minimum or at 0
    plt.ylim(bottom=y_min)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training, validation, and test losses from a JSONL file.')
    parser.add_argument('input_file', type=str, help='Path to the JSONL file')
    parser.add_argument('--output', type=str, help='Path to save the plot (optional)', default=None)
    parser.add_argument('--smooth', type=int, help='Smoothing factor for training curve (default: 1, no smoothing)', default=1)
    parser.add_argument('--use-epoch', action='store_true', help='Use epoch for x-axis instead of steps')
    parser.add_argument('--no-test', action='store_true', help='Exclude test metrics from the plot')
    args = parser.parse_args()
    
    print(f"Reading data from {args.input_file}...")
    data = read_jsonl(args.input_file)
    print(f"Extracted {len(data)} entries")
    
    metrics = extract_metrics(data, args.use_epoch)
    print(f"Found {len(metrics['train']['x'])} training loss values")
    print(f"Found {len(metrics['val']['x'])} validation loss values")
    print(f"Found {len(metrics['test']['x'])} test loss values")
    
    plot_losses(metrics, args.output, args.smooth, args.use_epoch, not args.no_test)
    print("Done!")

if __name__ == "__main__":
    main()