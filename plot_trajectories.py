#!/usr/bin/env python3
"""
Trajectory Plotter

This script plots the trajectories of the primary metric across all optimization iterations.
It visualizes both individual iteration performance and overall optimization progress.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

TEMP_DIR = "temp_0"

def load_config():
    """Load configuration from info.json."""
    # Try to load from info_final.json first, then fall back to info.json
    config_files = ["info_final.json", "info.json"]
    
    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                print(f"Loading configuration from {config_file}")
                return json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {config_file}")
            sys.exit(1)
    
    print("Error: Neither info_final.json nor info.json found")
    sys.exit(1)

def extract_trajectory_data(config):
    """Extract trajectory data for plotting, filtering out entries without metrics."""
    trajectories = config.get('previous_trajectories', [])
    primary_metric = config.get('metrics', {}).get('primary_metric', 'val_loss')
    
    if not trajectories:
        print("No trajectory data found in info.json")
        return None, None, None, None
    
    iteration_numbers = []
    final_metric_values = []
    all_epochs_data = []
    
    for traj in trajectories:
        iteration = traj.get('iteration', 0)
        metrics = traj.get('metrics', {})
        epochs = traj.get('epochs', [])
        
        # Skip entries that don't have the primary metric data
        if primary_metric not in metrics:
            print(f"Skipping iteration {iteration} - no {primary_metric} data")
            continue
            
        metric_values = metrics[primary_metric]
        
        # Skip if metric_values is empty or invalid
        if not metric_values or (isinstance(metric_values, list) and len(metric_values) == 0):
            print(f"Skipping iteration {iteration} - empty {primary_metric} data")
            continue
        
        iteration_numbers.append(iteration)
        
        # Extract primary metric values
        if isinstance(metric_values, list) and metric_values:
            final_metric_values.append(metric_values[-1])  # Final value
            all_epochs_data.append((iteration, epochs, metric_values))
        elif isinstance(metric_values, (int, float)):
            final_metric_values.append(metric_values)
            all_epochs_data.append((iteration, epochs, [metric_values]))
    
    if not iteration_numbers:
        print("No valid trajectory data found")
        return None, None, None, None
    
    print(f"Found {len(iteration_numbers)} valid iterations with {primary_metric} data")
    return iteration_numbers, final_metric_values, all_epochs_data, primary_metric


def generate_colors(n_iterations):
    """Generate distinct colors for all iterations."""
    if n_iterations <= 10:
        # Use tab10 for up to 10 iterations
        colors = plt.cm.tab10(np.linspace(0, 1, n_iterations))
    elif n_iterations <= 20:
        # Use Set3 for 11-20 iterations
        colors = plt.cm.Set3(np.linspace(0, 1, n_iterations))
    else:
        # For more than 20 iterations, use a combination of colormaps
        # or cycle through multiple colormaps
        colors = []
        colormaps = ['tab10', 'Set3', 'Paired', 'Dark2', 'Set1', 'Set2']
        
        for i in range(n_iterations):
            colormap_idx = (i // 10) % len(colormaps)
            color_idx = i % 10
            cmap = plt.cm.get_cmap(colormaps[colormap_idx])
            colors.append(cmap(color_idx / 10.0))
        
        colors = np.array(colors)
    
    return colors


def plot_trajectories(iteration_numbers, final_metric_values, all_epochs_data, primary_metric):
    """Create comprehensive trajectory plots."""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'ML Optimization Trajectories - {primary_metric}', fontsize=16, fontweight='bold')
    
    # Plot 1: Final metric value per iteration (optimization progress)
    if iteration_numbers and final_metric_values:
        ax1.plot(iteration_numbers, final_metric_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(f'Final {primary_metric}')
        ax1.set_title('Optimization Progress')
        ax1.grid(True, alpha=0.3)
        
        # Add best value annotation
        best_idx = np.argmin(final_metric_values) if 'loss' in primary_metric.lower() else np.argmax(final_metric_values)
        best_value = final_metric_values[best_idx]
        best_iter = iteration_numbers[best_idx]
        ax1.annotate(f'Best: {best_value:.4f}', 
                    xy=(best_iter, best_value), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Individual iteration trajectories (styled like your image)
    ax2.set_xlabel('Epoch/Step')
    ax2.set_ylabel(f'{primary_metric}')
    ax2.set_title('Training Trajectories by Iteration')
    ax2.grid(True, alpha=0.3)
    
    # Generate distinct colors for ALL iterations
    colors = generate_colors(len(all_epochs_data))
    
    # Plot each iteration trajectory
    plotted_count = 0
    for i, (iteration, epochs, metric_values) in enumerate(all_epochs_data):
        if len(metric_values) > 1:  # Only plot if there's trajectory data
            ax2.plot(range(len(metric_values)), metric_values, 
                    color=colors[i], label=f'Iteration {iteration}', 
                    linewidth=2, alpha=0.8)
            plotted_count += 1
    
    # Handle legend for many iterations
    if plotted_count <= 15:
        # Show legend for reasonable number of iterations
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        # For many iterations, add a note instead of crowded legend
        ax2.text(0.02, 0.98, f'Total iterations: {plotted_count}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set y-axis limits to better show the trajectories
    if all_epochs_data:
        all_values = []
        for _, _, metric_values in all_epochs_data:
            all_values.extend(metric_values)
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            if y_range > 0:  # Avoid division by zero
                ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Plot 3: Distribution of final values
    if final_metric_values:
        n_bins = min(len(final_metric_values), 20)  # Adjust bin count
        ax3.hist(final_metric_values, bins=n_bins, 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(final_metric_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(final_metric_values):.4f}')
        ax3.set_xlabel(f'Final {primary_metric}')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Final Metric Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Improvement over iterations
    if len(final_metric_values) > 1:
        improvements = []
        running_best = final_metric_values[0]
        improvements.append(0)
        
        for val in final_metric_values[1:]:
            if 'loss' in primary_metric.lower():
                improvement = max(0, running_best - val)
                running_best = min(running_best, val)
            else:
                improvement = max(0, val - running_best)
                running_best = max(running_best, val)
            improvements.append(improvement)
        
        ax4.bar(iteration_numbers, improvements, alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Improvement')
        ax4.set_title('Improvement per Iteration')
        ax4.grid(True, alpha=0.3)
    
    # Adjust layout to accommodate the legend
    plt.tight_layout()
    if plotted_count <= 15:
        plt.subplots_adjust(right=0.85)
    
    return fig

def save_hyperparameter_summary(config):
    """Save a summary of hyperparameters used in each iteration (only for iterations with metrics)."""
    trajectories = config.get('previous_trajectories', [])
    primary_metric = config.get('metrics', {}).get('primary_metric', 'val_loss')
    
    if not trajectories:
        return
    
    summary = []
    for traj in trajectories:
        iteration = traj.get('iteration', 0)
        hyperparams = traj.get('hyperparameters', {})
        metrics = traj.get('metrics', {})
        
        # Only include iterations that have metrics data
        if primary_metric not in metrics:
            continue
            
        metric_values = metrics[primary_metric]
        if not metric_values or (isinstance(metric_values, list) and len(metric_values) == 0):
            continue
        
        final_value = "N/A"
        if isinstance(metric_values, list) and metric_values:
            final_value = f"{metric_values[-1]:.4f}"
        elif isinstance(metric_values, (int, float)):
            final_value = f"{metric_values:.4f}"
        
        summary.append({
            'iteration': iteration,
            'hyperparameters': hyperparams,
            'final_metric': final_value
        })
    
    # Save to temp folder
    os.makedirs(TEMP_DIR, exist_ok=True)
    summary_filepath = os.path.join(TEMP_DIR, "hyperparameter_summary.json")
    try:
        with open(summary_filepath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Hyperparameter summary saved to {summary_filepath}")
    except Exception as e:
        print(f"Error saving hyperparameter summary: {e}")

def main():
    """Main execution function."""
    print("Loading trajectory data...")
    config = load_config()
    
    iteration_numbers, final_metric_values, all_epochs_data, primary_metric = extract_trajectory_data(config)
    
    if iteration_numbers is None:
        return
    
    print(f"Found {len(iteration_numbers)} valid iterations")
    print(f"Primary metric: {primary_metric}")
    
    # Create plots
    print("Creating trajectory plots...")
    fig = plot_trajectories(iteration_numbers, final_metric_values, all_epochs_data, primary_metric)
    
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save plot to temp folder
    plot_filename = os.path.join(TEMP_DIR, "trajectory_plots.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Trajectory plots saved to {plot_filename}")
    
    # Save hyperparameter summary
    save_hyperparameter_summary(config)
    
    # Show statistics
    if final_metric_values:
        print(f"\nOptimization Statistics:")
        print(f"Total iterations with data: {len(final_metric_values)}")
        print(f"Best {primary_metric}: {min(final_metric_values) if 'loss' in primary_metric.lower() else max(final_metric_values):.4f}")
        print(f"Mean {primary_metric}: {np.mean(final_metric_values):.4f}")
        print(f"Std {primary_metric}: {np.std(final_metric_values):.4f}")
    
    # Display plot
    plt.show()

if __name__ == "__main__":
    main()