import json
import ollama
import os
import sys
import numpy as np
from typing import Dict, List, Any, Tuple

# Configuration
LLM_MODEL = os.environ.get("LLM_MODEL")
TEMP_DIR = os.environ.get("TEMP_DIR", "temp")

# Helpers to handle complex hyperparameter values (e.g., list architectures)
def _canonical_param_value(v):
    """Return a hashable canonical form for a hyperparameter value."""
    if isinstance(v, list):
        return tuple(_canonical_param_value(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_canonical_param_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _canonical_param_value(val)) for k, val in v.items()))
    return v

def _display_param_value(v):
    """Return a compact, readable representation suitable for logs/prompts."""
    try:
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        return str(v)
    except Exception:
        return str(v)

def clean_response(response):
    """Remove thinking tags from LLM response if present."""
    if "</think>" in response:
        return response.split("</think>", 1)[1].strip()
    return response

def api_call(messages):
    """Make API call to Ollama with error handling."""
    print('Prompt:', messages[1]['content'])  # Debug output to see the prompt being sent
    try:
        result = ollama.chat(model=LLM_MODEL, messages=messages)
        response = result['message']['content']
        print(f"API Response: {response}")
        return clean_response(response)
    except Exception as e:
        print(f"Error in API call: {e}")
        return "Error: Unable to analyze training trajectory."

def load_training_results():
    """Load training results with error handling."""
    filepath = os.path.join(TEMP_DIR, "results.json")
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        sys.exit(1)

def load_config():
    """Load configuration from info.json."""
    filepath = os.path.join(TEMP_DIR, "info.json")
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        sys.exit(1)

def get_metric_info(config: Dict) -> Dict[str, Any]:
    """Extract primary metric information from configuration."""
    metrics_config = config.get("metrics", {})
    primary_metric = metrics_config.get("primary_metric", "val_accuracy")
    
    # Define metric properties based on common ML metrics
    metric_properties = {
        "val_accuracy": {"higher_is_better": True, "target": 0.90, "format": ".4f"},
        "accuracy": {"higher_is_better": True, "target": 0.90, "format": ".4f"},
        "f1_score": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        "precision": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        "recall": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        "auc_roc": {"higher_is_better": True, "target": 0.90, "format": ".4f"},
        "auc": {"higher_is_better": True, "target": 0.97, "format": ".4f"},
        "loss": {"higher_is_better": False, "target": 0.1, "format": ".6f"},
        "val_loss": {"higher_is_better": False, "target": 0.1, "format": ".6f"},
        "mse": {"higher_is_better": False, "target": 0.01, "format": ".6f"},
        "rmse": {"higher_is_better": False, "target": 0.1, "format": ".6f"},
        "mae": {"higher_is_better": False, "target": 0.1, "format": ".6f"},
        "logloss": {"higher_is_better": False, "target": 0.3, "format": ".6f"},
        # Computer Vision & Object Detection Metrics
        "iou": {"higher_is_better": True, "target": 0.70, "format": ".4f"},
        "miou": {"higher_is_better": True, "target": 0.70, "format": ".4f"},
        "mean_iou": {"higher_is_better": True, "target": 0.70, "format": ".4f"},
        # NLP Metrics
        "bleu": {"higher_is_better": True, "target": 0.30, "format": ".4f"},
        "bleu_score": {"higher_is_better": True, "target": 0.30, "format": ".4f"},
        "rouge": {"higher_is_better": True, "target": 0.40, "format": ".4f"},
        "rouge_l": {"higher_is_better": True, "target": 0.40, "format": ".4f"},
        "meteor": {"higher_is_better": True, "target": 0.35, "format": ".4f"},
        # Information Retrieval Metrics
        "ndcg": {"higher_is_better": True, "target": 0.80, "format": ".4f"},
        "ndcg@10": {"higher_is_better": True, "target": 0.80, "format": ".4f"},
        "ndcg_at_10": {"higher_is_better": True, "target": 0.80, "format": ".4f"},
        "map": {"higher_is_better": True, "target": 0.75, "format": ".4f"},
        "mean_average_precision": {"higher_is_better": True, "target": 0.75, "format": ".4f"},
        # Regression Metrics
        "r2": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        "r2_score": {"higher_is_better": True, "target": 0.7, "format": ".4f"},
        "r_squared": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        # Additional Classification Metrics
        "matthews_corrcoef": {"higher_is_better": True, "target": 0.70, "format": ".4f"},
        "cohen_kappa": {"higher_is_better": True, "target": 0.70, "format": ".4f"},
        "balanced_accuracy": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        "roc_auc": {"higher_is_better": True, "target": 0.90, "format": ".4f"},
        "pr_auc": {"higher_is_better": True, "target": 0.85, "format": ".4f"},
        # Distance/Similarity Metrics (lower is better)
        "hamming_loss": {"higher_is_better": False, "target": 0.1, "format": ".4f"},
        "jaccard_loss": {"higher_is_better": False, "target": 0.2, "format": ".4f"},
        "cosine_distance": {"higher_is_better": False, "target": 0.2, "format": ".4f"},
        "euclidean_distance": {"higher_is_better": False, "target": 1.0, "format": ".4f"},
        # Perplexity (lower is better)
        "perplexity": {"higher_is_better": False, "target": 50.0, "format": ".2f"},
        # Cross-entropy (lower is better)
        "cross_entropy": {"higher_is_better": False, "target": 0.5, "format": ".6f"},
        "categorical_crossentropy": {"higher_is_better": False, "target": 0.5, "format": ".6f"},
        "binary_crossentropy": {"higher_is_better": False, "target": 0.3, "format": ".6f"},
    }
    
    # Get properties for the primary metric, default to accuracy-like behavior
    properties = metric_properties.get(primary_metric, {
        "higher_is_better": True, 
        "target": 0.85, 
        "format": ".4f"
    })
    
    return {
        "name": primary_metric,
        "description": metrics_config.get("description", f"Primary metric: {primary_metric}"),
        "higher_is_better": properties["higher_is_better"],
        "target": properties["target"],
        "format": properties["format"]
    }

def get_hyperparameter_ranges(config: Dict) -> Dict[str, Dict]:
    """Extract ALL hyperparameter ranges from configuration in info.json."""
    hyperparameters = config.get("hyperparameters", {})
    ranges = {}
    
    print(f"DEBUG: Found hyperparameters in config: {list(hyperparameters.keys())}")
    
    for param, param_config in hyperparameters.items():
        print(f"DEBUG: Processing {param}: {param_config}")
        
        if isinstance(param_config, dict):
            if param == "batch_size" and "options" in param_config:
                # Special handling for batch_size with ordinal options
                ranges[param] = {
                    "min": param_config.get("range", [16, 256])[0] if "range" in param_config else param_config.get("min", 16),
                    "max": param_config.get("range", [16, 256])[1] if "range" in param_config else param_config.get("max", 256),
                    "default": param_config.get("default", 64),
                    "type": "ordinal",
                    "options": param_config.get("options", [16, 32, 64, 128, 256])
                }
            else:
                # Standard handling for other hyperparameters
                param_range = param_config.get("range", [0.0001, 0.1])
                ranges[param] = {
                    "min": param_range[0] if isinstance(param_range, list) else param_config.get("min", 0.0001),
                    "max": param_range[1] if isinstance(param_range, list) else param_config.get("max", 0.1),
                    "default": param_config.get("default", 0.01),
                    "type": param_config.get("type", "float")
                }
        else:
            # Fallback for simple value configs
            ranges[param] = {
                "min": 0.0001,
                "max": 0.1,
                "default": param_config,
                "type": "float"
            }
    
    print(f"DEBUG: Extracted ranges: {ranges}")
    return ranges

def analyze_performance_pattern(trajectories: List[Dict], metric_info: Dict) -> Dict[str, Any]:
    """Analyze performance pattern with clear reasoning for any metric."""
    if len(trajectories) < 2:
        return {"status": "insufficient_data", "reasoning": "Need more experiments"}
    
    primary_metric = metric_info["name"]
    higher_is_better = metric_info["higher_is_better"]
    target = metric_info["target"]
    
    # Get last 5 experiments
    recent = trajectories[-5:]
    primary_values = []
    train_values = []
    
    for traj in recent:
        metrics = traj.get("metrics", {})
        
        # Get primary metric value
        primary_val = metrics.get(primary_metric, [])
        if isinstance(primary_val, list) and primary_val:
            primary_values.append(primary_val[-1])
        elif isinstance(primary_val, (int, float)):
            primary_values.append(primary_val)
        
        # Try to get training metric (if available)
        train_metric = f"train_{primary_metric}" if not primary_metric.startswith("train_") else primary_metric
        train_val = metrics.get(train_metric, [])
        if isinstance(train_val, list) and train_val:
            train_values.append(train_val[-1])
        elif isinstance(train_val, (int, float)):
            train_values.append(train_val)
    
    if len(primary_values) < 2:
        return {"status": "insufficient_data", "reasoning": f"Missing {primary_metric} data"}
    
    # Current performance
    current_primary = primary_values[-1]
    current_train = train_values[-1] if train_values else current_primary
    previous_primary = primary_values[-2]
    change = current_primary - previous_primary
    
    # Overfitting analysis (only if we have train metrics)
    if train_values:
        if higher_is_better:
            overfitting_gap = current_train - current_primary
        else:
            overfitting_gap = current_primary - current_train
        overfitting_level = "HIGH" if overfitting_gap > 0.1 else "MODERATE" if overfitting_gap > 0.05 else "LOW"
    else:
        overfitting_gap = 0
        overfitting_level = "UNKNOWN"
    
    # Stability analysis
    if len(primary_values) >= 3:
        std_dev = np.std(primary_values)
        is_stable = std_dev < 0.01
        variance_level = "stable" if is_stable else "unstable"
    else:
        variance_level = "unknown"
    
    # Performance level assessment
    if higher_is_better:
        if current_primary >= target:
            performance_level = "excellent"
        elif current_primary >= target * 0.95:
            performance_level = "good"
        elif current_primary >= target * 0.90:
            performance_level = "fair"
        else:
            performance_level = "poor"
        target_gap = max(0, target - current_primary)
    else:
        if current_primary <= target:
            performance_level = "excellent"
        elif current_primary <= target * 1.1:
            performance_level = "good"
        elif current_primary <= target * 1.2:
            performance_level = "fair"
        else:
            performance_level = "poor"
        target_gap = max(0, current_primary - target)
    
    # Trend analysis
    improvement_threshold = 0.01 if higher_is_better else -0.01
    degradation_threshold = -0.01 if higher_is_better else 0.01
    
    if (higher_is_better and change > improvement_threshold) or (not higher_is_better and change < improvement_threshold):
        trend = "improving"
        reasoning = f"{primary_metric} improved by {abs(change):.4f}"
    elif (higher_is_better and change < degradation_threshold) or (not higher_is_better and change > degradation_threshold):
        trend = "degrading"
        reasoning = f"{primary_metric} degraded by {abs(change):.4f}"
    else:
        trend = "stagnant"
        reasoning = f"{primary_metric} stuck around {current_primary:.4f}"
    
    return {
        "status": trend,
        "current_primary": current_primary,
        "current_train": current_train,
        "change": change,
        "performance_level": performance_level,
        "stability": variance_level,
        "overfitting_gap": overfitting_gap,
        "overfitting_level": overfitting_level,
        "reasoning": reasoning,
        "target_gap": target_gap
    }

def identify_hyperparameter_impact(trajectories: List[Dict], available_hyperparams: Dict, metric_info: Dict) -> Dict[str, Any]:
    """Identify which hyperparameters have the most impact based on ALL available hyperparameters.
    
    Now analyzes the COMPLETE trajectory for best/worst performance, not just recent experiments.
    """
    if len(trajectories) < 2:
        return {"impact_analysis": "insufficient_data", "hyperparameter_history": {}}
    
    primary_metric = metric_info["name"]
    
    # Analyze ALL trajectories for best/worst performance
    all_trajectories = trajectories  # Use complete history for best/worst analysis
    recent = trajectories[-5:] if len(trajectories) >= 5 else trajectories  # Keep recent for trend analysis
    
    impacts = {}
    hyperparameter_history = {}
    
    print(f"DEBUG: Analyzing impact for hyperparameters: {list(available_hyperparams.keys())}")
    print(f"DEBUG: Using ALL {len(all_trajectories)} trajectories for best/worst analysis")
    print(f"DEBUG: Using last {len(recent)} trajectories for recent trend analysis")
    
    for param_name in available_hyperparams.keys():
        # Collect data from ALL trajectories for best/worst analysis
        all_param_values = []
        all_performance_values = []
        all_trajectory_info = []
        
        # Collect data from recent trajectories for trend analysis
        recent_param_values = []
        recent_performance_values = []
        recent_trajectory_info = []
        
        # Process ALL trajectories first
        for traj in all_trajectories:
            hparams = traj.get("hyperparameters", {})
            metrics = traj.get("metrics", {})
            
            if param_name in hparams:
                param_value = hparams[param_name]
                
                # Get primary metric value
                metric_val = metrics.get(primary_metric, [])
                performance_val = None
                
                if isinstance(metric_val, list) and metric_val:
                    performance_val = metric_val[-1]
                elif isinstance(metric_val, (int, float)):
                    performance_val = metric_val
                
                if performance_val is not None:
                    all_param_values.append(param_value)
                    all_performance_values.append(performance_val)
                    all_trajectory_info.append({
                        "iteration": traj.get("iteration", "unknown"),
                        "param_value": param_value,
                        f"{primary_metric}": performance_val,
                        "all_hyperparams": hparams
                    })
        
        # Process RECENT trajectories for trend analysis
        for traj in recent:
            hparams = traj.get("hyperparameters", {})
            metrics = traj.get("metrics", {})
            
            if param_name in hparams:
                param_value = hparams[param_name]
                
                # Get primary metric value
                metric_val = metrics.get(primary_metric, [])
                performance_val = None
                
                if isinstance(metric_val, list) and metric_val:
                    performance_val = metric_val[-1]
                elif isinstance(metric_val, (int, float)):
                    performance_val = metric_val
                
                if performance_val is not None:
                    recent_param_values.append(param_value)
                    recent_performance_values.append(performance_val)
                    recent_trajectory_info.append({
                        "iteration": traj.get("iteration", "unknown"),
                        "param_value": param_value,
                        f"{primary_metric}": performance_val,
                        "all_hyperparams": hparams
                    })
        
        # Store complete history for this hyperparameter
        hyperparameter_history[param_name] = all_trajectory_info
        
        if len(all_param_values) >= 1:
            if len(all_param_values) >= 2:
                # Calculate best/worst from ALL trajectories
                if metric_info["higher_is_better"]:
                    best_idx = np.argmax(all_performance_values)
                    worst_idx = np.argmin(all_performance_values)
                else:
                    best_idx = np.argmin(all_performance_values)
                    worst_idx = np.argmax(all_performance_values)
                
                impacts[param_name] = {
                    # Best/worst from complete history
                    "best_value": all_param_values[best_idx],
                    "worst_value": all_param_values[worst_idx],
                    "best_performance": all_performance_values[best_idx],
                    "worst_performance": all_performance_values[worst_idx],
                    "best_iteration": all_trajectory_info[best_idx]["iteration"],
                    "worst_iteration": all_trajectory_info[worst_idx]["iteration"],
                    
                    # Overall statistics from complete history
                    "performance_range": abs(max(all_performance_values) - min(all_performance_values)),
                    "all_values_tested": all_param_values,
                    "all_performances": all_performance_values,
                    "total_experiments": len(all_param_values),
                    
                    # Recent trend information
                    "recent_values": recent_param_values,
                    "recent_performances": recent_performance_values,
                    "recent_experiments": len(recent_param_values),
                    
                    # Summary statistics
                    "mean_performance": np.mean(all_performance_values),
                    "std_performance": np.std(all_performance_values),
                    "unique_values_tested": len({ _canonical_param_value(v) for v in all_param_values })
                }
                
                print(f"DEBUG: {param_name} - Best from iter {impacts[param_name]['best_iteration']}: "
                      f"{impacts[param_name]['best_value']} -> {impacts[param_name]['best_performance']:.4f}")
                print(f"DEBUG: {param_name} - Worst from iter {impacts[param_name]['worst_iteration']}: "
                      f"{impacts[param_name]['worst_value']} -> {impacts[param_name]['worst_performance']:.4f}")
                
            else:
                # Single value case
                impacts[param_name] = {
                    "current_value": all_param_values[0],
                    "current_performance": all_performance_values[0],
                    "current_iteration": all_trajectory_info[0]["iteration"],
                    "total_experiments": 1,
                    "note": "Only one experiment - need more data to assess impact"
                }
    
    return {
        "impact_analysis": impacts,
        "hyperparameter_history": hyperparameter_history
    }
def filter_latest_metrics(latest_metrics, interval=5):
    """
    Filter latest_metrics to keep only values at specified epoch intervals.
    Modifies the latest_metrics dictionary in place.
    
    Args:
        latest_metrics: Dictionary containing metrics data
        interval: Epoch interval to keep (default: 5)
    """
    # Check if metrics exist in latest_metrics
    if not latest_metrics:
        return
    
    # Get epochs array (assuming it exists)
    epochs = latest_metrics.get("epochs", [])
    if not epochs:
        return
    
    # Find indices for epochs we want to keep (1, 5, 10, 15, ...)
    keep_indices = []
    for i, epoch in enumerate(epochs):
        if epoch == 1 or epoch % interval == 0:
            keep_indices.append(i)
    
    # Filter all metric arrays in latest_metrics
    for key, value in latest_metrics.items():
        if isinstance(value, list) and len(value) == len(epochs):
            # Filter the array to keep only interval values
            latest_metrics[key] = [value[i] for i in keep_indices]

def create_comprehensive_hyperparameter_analysis(trajectories: List[Dict], available_hyperparams: Dict, metric_info: Dict) -> str:
    """Create detailed analysis of all available hyperparameters using complete trajectory history."""
    impact_data = identify_hyperparameter_impact(trajectories, available_hyperparams, metric_info)
    impacts = impact_data["impact_analysis"]
    history = impact_data["hyperparameter_history"]
    
    analysis = "\n=== DETAILED HYPERPARAMETER ANALYSIS (COMPLETE HISTORY) ===\n"
    
    for param_name, param_config in available_hyperparams.items():
        analysis += f"\n--- {param_name.upper()} ---\n"
        analysis += f"Range: {param_config['min']} to {param_config['max']}\n"
        analysis += f"Type: {param_config['type']}\n"
        
        if param_name == "batch_size" and "options" in param_config:
            analysis += f"Valid options: {param_config['options']}\n"
        
        if param_name in impacts:
            impact_info = impacts[param_name]
            if impact_info["total_experiments"] >= 2:
                best_perf_str = f"{impact_info['best_performance']:.4f}" if isinstance(impact_info['best_performance'], (int, float)) else str(impact_info['best_performance'])
                worst_perf_str = f"{impact_info['worst_performance']:.4f}" if isinstance(impact_info['worst_performance'], (int, float)) else str(impact_info['worst_performance'])
                range_str = f"{impact_info['performance_range']:.4f}" if isinstance(impact_info['performance_range'], (int, float)) else str(impact_info['performance_range'])
                mean_perf_str = f"{impact_info['mean_performance']:.4f}" if isinstance(impact_info['mean_performance'], (int, float)) else str(impact_info['mean_performance'])
                
                analysis += f"BEST EVER: {impact_info['best_value']} in iteration {impact_info['best_iteration']} ({metric_info['name']}: {best_perf_str})\n"
                analysis += f"WORST EVER: {impact_info['worst_value']} in iteration {impact_info['worst_iteration']} ({metric_info['name']}: {worst_perf_str})\n"
                analysis += f"Performance impact range: {range_str}\n"
                analysis += f"Total experiments: {impact_info['total_experiments']}\n"
                analysis += f"Unique values tested: {impact_info['unique_values_tested']}\n"
                analysis += f"Average performance: {mean_perf_str} ± {impact_info['std_performance']:.4f}\n"
                
                # Show recent trend if different from overall
                if impact_info["recent_experiments"] > 0:
                    recent_best = max(impact_info["recent_performances"]) if metric_info["higher_is_better"] else min(impact_info["recent_performances"])
                    recent_best_str = f"{recent_best:.4f}"
                    analysis += f"Recent best ({len(impact_info['recent_values'])} experiments): {recent_best_str}\n"
                
                # Provide insight on value exploration
                unique_display = sorted({ _display_param_value(v) for v in impact_info.get("all_values_tested", []) })
                analysis += f"All tested values: {unique_display}\n"
                
            else:
                current_perf_str = f"{impact_info['current_performance']:.4f}" if isinstance(impact_info['current_performance'], (int, float)) else str(impact_info['current_performance'])
                
                analysis += f"Current value: {impact_info['current_value']} (iteration {impact_info['current_iteration']})\n"
                analysis += f"Current performance: {current_perf_str}\n"
                analysis += "Status: NEEDS MORE EXPERIMENTS\n"
        else:
            analysis += "Status: NEVER TESTED - HIGH PRIORITY FOR EXPLORATION\n"
        
        # Show only last 3 experiments in history, but best/worst are from complete history
        if param_name in history:
            total_experiments = len(history[param_name])
            analysis += f"Recent experiment history (last 3 of {total_experiments} total):\n"
            
            # Show only last 3 experiments
            experiments_to_show = history[param_name][-3:]
                
            for exp in experiments_to_show:
                metric_val = exp.get(metric_info['name'], 'N/A')
                metric_val_str = f"{metric_val:.4f}" if isinstance(metric_val, (int, float)) else str(metric_val)
                analysis += f"  Iter {exp['iteration']}: {param_name}={exp['param_value']} → {metric_info['name']}={metric_val_str}\n"
    
    return analysis
def create_focused_analysis_prompt(trajectories: List[Dict], config: Dict) -> str:
    """Create a focused, reasoning-oriented analysis prompt."""
    if len(trajectories) < 1:
        return "No experiment data available."
    
    metric_info = get_metric_info(config)
    available_hyperparams = get_hyperparameter_ranges(config)
    performance_analysis = analyze_performance_pattern(trajectories, metric_info)
    
    # Get latest experiment details
    latest = trajectories[-1]
    latest_hparams = latest.get("hyperparameters", {})
    latest_metrics = latest.get("metrics", {}).copy()  # Create a copy to avoid modifying original
    
    # Filter metrics to reduce prompt size (keep only interval epochs)
    filter_latest_metrics(latest_metrics, interval=5)
    
    # Extract key metrics from latest experiment
    primary_metric = metric_info["name"]
    primary_val = latest_metrics.get(primary_metric, [])
    train_metric = f"train_{primary_metric}" if not primary_metric.startswith("train_") else primary_metric
    train_val = latest_metrics.get(train_metric, [])
    
    latest_primary = primary_val[-1] if isinstance(primary_val, list) and primary_val else (primary_val if isinstance(primary_val, (int, float)) else "N/A")
    latest_train = train_val[-1] if isinstance(train_val, list) and train_val else (train_val if isinstance(train_val, (int, float)) else "N/A")
    
    # Determine target direction
    target_direction = ">" if metric_info["higher_is_better"] else "<"
    
    # Create focused summary with FILTERED metrics only - fixed formatting issue
    latest_hparams_str = str(latest_hparams).replace('{', '{{').replace('}', '}}')
    latest_metrics_str = str(latest_metrics).replace('{', '{{').replace('}', '}}')
    metric_name = metric_info["name"]
    
    # Handle formatting for string values like "N/A"
    latest_primary_str = f"{latest_primary:.4f}" if isinstance(latest_primary, (int, float)) else str(latest_primary)
    latest_train_str = f"{latest_train:.4f}" if isinstance(latest_train, (int, float)) else str(latest_train)
    
    summary = f"""=== CURRENT SITUATION ===
TARGET: {metric_name} {target_direction} {metric_info['target']:.3f}
CURRENT: {latest_primary_str} ({performance_analysis.get('performance_level', 'unknown').upper()})
GAP TO TARGET: {performance_analysis.get('target_gap', 0):.4f}
TREND: {performance_analysis.get('status', 'unknown').upper()}
OVERFITTING: {performance_analysis.get('overfitting_level', 'UNKNOWN')} (Train-Primary Gap: {performance_analysis.get('overfitting_gap', 0):.4f})

=== LATEST EXPERIMENT (Iteration {latest.get('iteration', 'N/A')}) ===
Hyperparameters Used: {latest_hparams_str}
Final Train {metric_name}: {latest_train}
Final {metric_name}: {latest_primary}
Performance: {performance_analysis.get('reasoning', 'No reasoning available')}
Pervious training tragectories : {latest_metrics}

=== ALL AVAILABLE HYPERPARAMETERS TO OPTIMIZE ==="""
    
    for param, ranges in available_hyperparams.items():
        current_val = latest_hparams.get(param, 'NEVER_USED')
        if param == "batch_size" and "options" in ranges:
            summary += f"\n{param}: {ranges['options']} (current: {current_val})"
        else:
            summary += f"\n{param}: {ranges['min']} to {ranges['max']} (current: {current_val})"
    
    # Add comprehensive hyperparameter analysis
    hyperparameter_analysis = create_comprehensive_hyperparameter_analysis(trajectories, available_hyperparams, metric_info)
    summary += hyperparameter_analysis
    
    # Add comparison with previous experiment if available
    if len(trajectories) >= 2:
        prev = trajectories[-2]
        prev_hparams = prev.get("hyperparameters", {})
        prev_metrics = prev.get("metrics", {})
        prev_primary = prev_metrics.get(primary_metric, [])
        prev_primary_final = prev_primary[-1] if isinstance(prev_primary, list) and prev_primary else (prev_primary if isinstance(prev_primary, (int, float)) else "N/A")
        
        prev_primary_str = f"{prev_primary_final:.4f}" if isinstance(prev_primary_final, (int, float)) else str(prev_primary_final)
        
        summary += f"\n\n=== PREVIOUS EXPERIMENT COMPARISON ===\n"
        summary += f"Previous {primary_metric}: {prev_primary_str}\n"
        summary += f"{primary_metric} Change: {performance_analysis.get('change', 0):+.4f}\n"
        
        # Show ALL hyperparameter changes
        summary += "Hyperparameter changes made:\n"
        for param in available_hyperparams.keys():
            prev_val = prev_hparams.get(param, "NOT_SET")
            curr_val = latest_hparams.get(param, "NOT_SET")
            if prev_val != curr_val:
                summary += f"  {param}: {prev_val} → {curr_val}\n"
            else:
                summary += f"  {param}: {curr_val} (unchanged)\n"
    
    # Debug info to confirm filtering worked
    epochs_included = latest_metrics.get("epochs", [])
    if epochs_included:
        print(f"DEBUG: Filtered metrics - using epochs: {epochs_included}")
        print(f"DEBUG: Reduced from ~75 epochs to {len(epochs_included)} epochs")
    
    return summary

def analyze_training_trajectory(results: Dict, config: Dict) -> str:
    """Main analysis function with focused reasoning."""
    trajectories = config.get("previous_trajectories", [])
    metric_info = get_metric_info(config)
    
    # Add current results
    current_trajectory = {
        "iteration": int(os.environ.get("ITERATION", "0")),
        "metrics": results.get("metrics", {}),
        "epochs": results.get("epochs", []),
        "hyperparameters": json.loads(os.environ.get("PREVIOUS_HYPERPARAMETERS", "{}")),
    }
    
    all_trajectories = trajectories + [current_trajectory]
    available_hyperparams = get_hyperparameter_ranges(config)
    
    # Create focused analysis prompt
    situation_summary = create_focused_analysis_prompt(all_trajectories, config)
    
    # Enhanced prompt that considers ALL hyperparameters
    prompt = [
        {
            'role': 'system',
            'content': f'''You are an expert ML optimizer. Your goal is to optimize {metric_info['name']} to {'exceed' if metric_info['higher_is_better'] else 'be below'} {metric_info['target']:.3f}.

CRITICAL: You must consider ALL available hyperparameters, not just learning rate!

AVAILABLE HYPERPARAMETERS: {list(available_hyperparams.keys())}
PRIMARY METRIC: {metric_info['name']} ({'higher is better' if metric_info['higher_is_better'] else 'lower is better'})

REASONING APPROACH:
1. Identify the main problem (overfitting, underfitting, or optimization issue)
2. Consider which hyperparameter would have the BIGGEST impact on the problem
3. Learning rate affects optimization speed and convergence
4. Regularization parameters (weight_decay, dropout, etc.) affect overfitting
5. Tree-based parameters (max_depth, n_estimators, etc.) affect model capacity
6. Other parameters may affect model stability or generalization
7. Choose the MOST IMPACTFUL hyperparameter to adjust
8. Provide clear cause-effect reasoning

MANDATORY: You have to optimize the hyperparameters available in {list(available_hyperparams.keys())} so suggest only their values.'''
        },
        {
            'role': 'user',
            'content': f"""{situation_summary}

ANALYSIS TASK:
Analyze this situation and provide a strategy to reach the target {metric_info['name']} of {metric_info['target']:.3f}
***ALWAYS SUGGEST HYPERPARAMETERS IN THE RANGE GIVEN FOR HYPERPARAMETERS***.

REQUIRED ANALYSIS FORMAT:(**Suggest values with in the range given for hyperparameters**)
1. PROBLEM DIAGNOSIS: What is the main issue preventing target {metric_info['name']}?
2. HYPERPARAMETER IMPACT ASSESSMENT: 
   - How would changing each available hyperparameter help/hurt the {metric_info['name']}?
   - Consider the specific nature of this metric and model type
3. PRIMARY ACTION: Which hyperparameter should be changed and why is it MORE impactful than others?
4. SPECIFIC RECOMMENDATION: Exact value/direction to change
5. REASONING: Why this specific change will improve {metric_info['name']} more than alternatives?
6. EXPECTED OUTCOME: What {metric_info['name']} do you expect?

REMINDER: You have these hyperparameters available: {list(available_hyperparams.keys())}
TARGET: {metric_info['name']} {'>' if metric_info['higher_is_better'] else '<'} {metric_info['target']:.3f}"""
        }
    ]
    
    return api_call(prompt)

def save_analysis_summary(analysis: str, config: Dict):
    """Save a focused analysis summary for the hyperparameter optimizer."""
    trajectories = config.get("previous_trajectories", [])
    if not trajectories:
        return
    
    metric_info = get_metric_info(config)
    latest_trajectory = trajectories[-1]
    performance_analysis = analyze_performance_pattern(trajectories, metric_info)
    available_hyperparams = get_hyperparameter_ranges(config)
    impact_analysis = identify_hyperparameter_impact(trajectories, available_hyperparams, metric_info)
    
    # Get latest performance value
    latest_metrics = latest_trajectory.get("metrics", {})
    primary_metric = metric_info["name"]
    metric_val = latest_metrics.get(primary_metric, [])
    
    if isinstance(metric_val, list) and metric_val:
        latest_performance = metric_val[-1]
    elif isinstance(metric_val, (int, float)):
        latest_performance = metric_val
    else:
        latest_performance = None
    
    # Create focused summary for optimizer
    summary = {
        "analysis": analysis,
        "performance_status": performance_analysis,
        "available_hyperparameters": available_hyperparams,
        "hyperparameter_impact": impact_analysis,
        "latest_hyperparameters": latest_trajectory.get("hyperparameters", {}),
        "latest_performance": {
            primary_metric: latest_performance
        },
        "target_gap": performance_analysis.get("target_gap", 0),
        "overfitting_status": performance_analysis.get("overfitting_level", "UNKNOWN"),
        "hyperparameters_tested": list(latest_trajectory.get("hyperparameters", {}).keys()),
        "hyperparameters_never_tested": [param for param in available_hyperparams.keys() 
                                        if param not in latest_trajectory.get("hyperparameters", {})],
        "metric_info": metric_info
    }
    
    filepath = os.path.join(TEMP_DIR, "analysis_summary.json")
    try:
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Comprehensive analysis summary saved to {filepath}")
    except Exception as e:
        print(f"Error saving analysis summary: {e}")

def update_config_with_results(config: Dict, results: Dict, analysis: str):
    """Update info.json with new trajectory data."""
    iteration = int(os.environ.get("ITERATION", "0"))
    
    new_trajectory = {
        "iteration": iteration,
        "metrics": results.get('metrics', {}),
        "epochs": results.get('epochs', []),
        "analysis": analysis,
        "hyperparameters": json.loads(os.environ.get('PREVIOUS_HYPERPARAMETERS', '{}'))
    }
    
    if 'previous_trajectories' not in config:
        config['previous_trajectories'] = []
    
    config["previous_trajectories"].append(new_trajectory)
    
    filepath = os.path.join(TEMP_DIR, "info.json")
    try:
        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Successfully updated {filepath} with iteration {iteration} results")
    except Exception as e:
        print(f"Error writing to {filepath}: {e}")

def main():
    """Main execution function."""
    results = load_training_results()
    config = load_config()
    
    print("Analyzing training results with comprehensive hyperparameter focus...")
    
    # Get metric and hyperparameter info
    metric_info = get_metric_info(config)
    available_hyperparams = get_hyperparameter_ranges(config)
    
    print(f"Primary metric: {metric_info['name']} (target: {'>' if metric_info['higher_is_better'] else '<'} {metric_info['target']:.3f})")
    print(f"Available hyperparameters found in config: {list(available_hyperparams.keys())}")
    
    if not available_hyperparams:
        print("ERROR: No hyperparameters found in config!")
        return
    
    # Analyze with comprehensive approach
    analysis = analyze_training_trajectory(results, config)
    print(f"\nComprehensive Training Analysis for {metric_info['name']}:")
    print(analysis)
    
    # Update configuration and save focused summary
    update_config_with_results(config, results, analysis)
    save_analysis_summary(analysis, config)

if __name__ == "__main__":
    main()
