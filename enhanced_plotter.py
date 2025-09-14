#!/usr/bin/env python3
"""
Enhanced Hyperparameter Optimization Results Plotter

This script creates comprehensive visualizations for hyperparameter optimization results,
including optimization progress, hyperparameter relationships, correlation analysis,
and performance insights across all iterations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import sys
from datetime import datetime

TEMP_DIR = "temp"

def load_config():
    """Load configuration from info.json."""
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

def get_project_name():
    """Extract project name from environment or configuration."""
    # Try to get from environment variable first
    project_name = os.environ.get("PROJECT_NAME")
    if project_name:
        return project_name
    
    # Try to extract from run.sh or use default
    try:
        with open("run.sh", "r") as f:
            content = f.read()
            if 'PROJECT_NAME=' in content:
                for line in content.split('\n'):
                    if line.strip().startswith('export PROJECT_NAME='):
                        return line.split('=', 1)[1].strip().strip('"\'')
    except FileNotFoundError:
        pass
    
    # Default fallback
    return "ML Hyperparameter Optimization"

def extract_comprehensive_data(config):
    """Extract all data needed for comprehensive analysis."""
    trajectories = config.get('previous_trajectories', [])
    primary_metric = config.get('metrics', {}).get('primary_metric', 'val_loss')
    hyperparams_config = config.get('hyperparameters', {})
    
    if not trajectories:
        print("No trajectory data found")
        return None
    
    # Create comprehensive dataset
    data = []
    for traj in trajectories:
        iteration = traj.get('iteration', 0)
        hyperparams = traj.get('hyperparameters', {})
        metrics = traj.get('metrics', {})
        epochs = traj.get('epochs', [])
        
        # Extract final metric value
        final_metric_value = None
        if primary_metric in metrics:
            metric_values = metrics[primary_metric]
            if isinstance(metric_values, list) and metric_values:
                final_metric_value = metric_values[-1]
            elif isinstance(metric_values, (int, float)):
                final_metric_value = metric_values
        
        # Create row for this iteration
        row = {
            'iteration': iteration,
            'final_metric': final_metric_value,
            'epochs': len(epochs) if epochs else 0,
            'metric_trajectory': metrics.get(primary_metric, []),
            'all_metrics': metrics
        }
        
        # Add hyperparameters
        for param_name in hyperparams_config.keys():
            row[param_name] = hyperparams.get(param_name, hyperparams_config[param_name].get('default', 0))
        
        data.append(row)
    
    return pd.DataFrame(data), primary_metric, hyperparams_config

def create_comprehensive_plots(df, primary_metric, hyperparams_config, project_name):
    """Create comprehensive hyperparameter optimization visualizations."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Add main title with project name
    fig.suptitle(f'{project_name}\nHyperparameter Optimization Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    # 1. Optimization Progress (Top Left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    if 'final_metric' in df.columns and not df['final_metric'].isna().all():
        ax1.plot(df['iteration'], df['final_metric'], 'o-', linewidth=3, markersize=8, 
                color='#2E86AB', alpha=0.8)
        
        # Add running best line
        if 'loss' in primary_metric.lower():
            running_best = df['final_metric'].cummin()
        else:
            running_best = df['final_metric'].cummax()
        ax1.plot(df['iteration'], running_best, '--', linewidth=2, 
                color='#A23B72', alpha=0.7, label='Running Best')
        
        # Highlight best point
        best_idx = df['final_metric'].idxmin() if 'loss' in primary_metric.lower() else df['final_metric'].idxmax()
        ax1.scatter(df.loc[best_idx, 'iteration'], df.loc[best_idx, 'final_metric'], 
                   s=200, color='gold', edgecolor='black', linewidth=2, zorder=5)
        
        ax1.set_xlabel('Optimization Iteration', fontsize=12)
        ax1.set_ylabel(f'{primary_metric.replace("_", " ").title()}', fontsize=12)
        ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. Training Trajectories (Top Right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    for idx, row in df.iterrows():
        if isinstance(row['metric_trajectory'], list) and len(row['metric_trajectory']) > 1:
            ax2.plot(range(len(row['metric_trajectory'])), row['metric_trajectory'], 
                    color=colors[idx], alpha=0.7, linewidth=2, 
                    label=f"Iter {row['iteration']}")
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(f'{primary_metric.replace("_", " ").title()}', fontsize=12)
    ax2.set_title('Training Trajectories by Iteration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(df) <= 10:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # 3. Hyperparameter Correlation Heatmap (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0:2])
    numeric_params = []
    for param in hyperparams_config.keys():
        if param in df.columns and df[param].dtype in ['int64', 'float64']:
            numeric_params.append(param)
    
    if numeric_params and 'final_metric' in df.columns:
        corr_data = df[numeric_params + ['final_metric']].corr()
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('Hyperparameter Correlations', fontsize=14, fontweight='bold')
    
    # 4. Hyperparameter vs Performance Scatter (Middle Right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    if numeric_params and 'final_metric' in df.columns:
        # Choose the most correlated hyperparameter
        correlations = df[numeric_params].corrwith(df['final_metric']).abs()
        if not correlations.empty and correlations.notna().any():
            best_param = correlations.idxmax()
            scatter = ax4.scatter(df[best_param], df['final_metric'], 
                                c=df['iteration'], cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black')
            ax4.set_xlabel(f'{best_param.replace("_", " ").title()}', fontsize=12)
            ax4.set_ylabel(f'{primary_metric.replace("_", " ").title()}', fontsize=12)
            ax4.set_title(f'Performance vs {best_param.replace("_", " ").title()}', 
                         fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Iteration')
    
    # 5. Performance Distribution (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0:2])
    if 'final_metric' in df.columns and not df['final_metric'].isna().all():
        ax5.hist(df['final_metric'], bins=min(len(df), 10), alpha=0.7, 
                color='skyblue', edgecolor='black', density=True)
        ax5.axvline(df['final_metric'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["final_metric"].mean():.4f}')
        ax5.axvline(df['final_metric'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {df["final_metric"].median():.4f}')
        ax5.set_xlabel(f'{primary_metric.replace("_", " ").title()}', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.set_title('Performance Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Hyperparameter Evolution (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 2:4])
    if numeric_params:
        # Normalize hyperparameters for comparison
        normalized_df = df[numeric_params].copy()
        for param in numeric_params:
            param_min = normalized_df[param].min()
            param_max = normalized_df[param].max()
            if param_max != param_min:
                normalized_df[param] = (normalized_df[param] - param_min) / (param_max - param_min)
        
        for param in numeric_params:
            ax6.plot(df['iteration'], normalized_df[param], 'o-', 
                    label=param.replace('_', ' ').title(), alpha=0.7, linewidth=2)
        
        ax6.set_xlabel('Iteration', fontsize=12)
        ax6.set_ylabel('Normalized Parameter Value', fontsize=12)
        ax6.set_title('Hyperparameter Evolution', fontsize=14, fontweight='bold')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax6.grid(True, alpha=0.3)
    
    # 7. Performance Improvement Analysis (Bottom)
    ax7 = fig.add_subplot(gs[3, 0:2])
    if 'final_metric' in df.columns and len(df) > 1:
        improvements = []
        running_best = df['final_metric'].iloc[0]
        improvements.append(0)
        
        for i in range(1, len(df)):
            current_val = df['final_metric'].iloc[i]
            if 'loss' in primary_metric.lower():
                improvement = max(0, running_best - current_val)
                running_best = min(running_best, current_val)
            else:
                improvement = max(0, current_val - running_best)
                running_best = max(running_best, current_val)
            improvements.append(improvement)
        
        bars = ax7.bar(df['iteration'], improvements, alpha=0.7, 
                      color='lightgreen', edgecolor='black')
        ax7.set_xlabel('Iteration', fontsize=12)
        ax7.set_ylabel('Improvement', fontsize=12)
        ax7.set_title('Performance Improvement per Iteration', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            if imp > 0:
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{imp:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 8. Summary Statistics (Bottom Right)
    ax8 = fig.add_subplot(gs[3, 2:4])
    ax8.axis('off')
    
    # Create summary statistics
    if 'final_metric' in df.columns and not df['final_metric'].isna().all():
        best_value = df['final_metric'].min() if 'loss' in primary_metric.lower() else df['final_metric'].max()
        best_iter = df.loc[df['final_metric'].idxmin() if 'loss' in primary_metric.lower() else df['final_metric'].idxmax(), 'iteration']
        
        stats_text = f"""
        OPTIMIZATION SUMMARY
        {'='*30}
        
        Total Iterations: {len(df)}
        Best {primary_metric}: {best_value:.4f}
        Best Iteration: {best_iter}
        
        Mean {primary_metric}: {df['final_metric'].mean():.4f}
        Std {primary_metric}: {df['final_metric'].std():.4f}
        
        Improvement Range: {(df['final_metric'].max() - df['final_metric'].min()):.4f}
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    return fig

def save_detailed_analysis(df, primary_metric, output_dir):
    """Save detailed analysis to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hyperparameter analysis
    analysis_file = os.path.join(output_dir, "detailed_analysis.json")
    
    analysis = {
        "summary": {
            "total_iterations": len(df),
            "best_metric": float(df['final_metric'].min() if 'loss' in primary_metric.lower() else df['final_metric'].max()),
            "mean_metric": float(df['final_metric'].mean()),
            "std_metric": float(df['final_metric'].std()),
            "improvement_range": float(df['final_metric'].max() - df['final_metric'].min())
        },
        "best_configuration": {},
        "correlations": {}
    }
    
    # Best configuration
    best_idx = df['final_metric'].idxmin() if 'loss' in primary_metric.lower() else df['final_metric'].idxmax()
    best_row = df.loc[best_idx]
    
    for col in df.columns:
        if col not in ['iteration', 'final_metric', 'epochs', 'metric_trajectory', 'all_metrics']:
            analysis["best_configuration"][col] = best_row[col]
    
    # Correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'final_metric' in numeric_cols:
        for col in numeric_cols:
            if col != 'final_metric':
                corr = df[col].corr(df['final_metric'])
                if not np.isnan(corr):
                    analysis["correlations"][col] = float(corr)
    
    try:
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Detailed analysis saved to {analysis_file}")
    except Exception as e:
        print(f"Error saving analysis: {e}")

def main():
    """Main execution function."""
    print("Loading optimization data...")
    config = load_config()
    project_name = get_project_name()
    
    # Extract comprehensive data
    result = extract_comprehensive_data(config)
    if result is None:
        print("No data to plot")
        return
    
    df, primary_metric, hyperparams_config = result
    
    print(f"Found {len(df)} iterations")
    print(f"Primary metric: {primary_metric}")
    print(f"Project: {project_name}")
    print(f"Hyperparameters: {list(hyperparams_config.keys())}")
    
    # Create comprehensive plots
    print("Creating enhanced visualizations...")
    fig = create_comprehensive_plots(df, primary_metric, hyperparams_config, project_name)
    
    # Save plot
    output_dir = TEMP_DIR
    plot_filename = os.path.join(output_dir, "enhanced_optimization_analysis.png")
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Enhanced plots saved to {plot_filename}")
    
    # Save detailed analysis
    save_detailed_analysis(df, primary_metric, output_dir)
    
    # Display plot
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"OPTIMIZATION ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Project: {project_name}")
    print(f"Total Iterations: {len(df)}")
    if 'final_metric' in df.columns:
        best_value = df['final_metric'].min() if 'loss' in primary_metric.lower() else df['final_metric'].max()
        print(f"Best {primary_metric}: {best_value:.4f}")
    print(f"Visualizations saved to: {plot_filename}")

if __name__ == "__main__":
    main()
