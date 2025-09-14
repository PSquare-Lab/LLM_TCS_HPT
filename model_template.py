#!/usr/bin/env python3
"""
Comprehensive Model Template for LLM Hyperparameter Optimization System

This template provides a complete, framework-agnostic structure that follows the exact
interface patterns expected by the LLM hyperparameter optimization system.

CRITICAL INTERFACE REQUIREMENTS:
- Read hyperparameters from ${TEMP_DIR}/hyperparameters.json (default: temp/hyperparameters.json)
- Read primary metric name from ${TEMP_DIR}/info.json at metrics.primary_metric
- Save results to ${TEMP_DIR}/results.json with exact schema: 
  {"metrics": {"primary_metric_name": [values...]}, "epochs": [1,2,3...]}
- Environment variable: TEMP_DIR (default: "temp")

USAGE:
1. Copy this template file to your project
2. Replace the marked sections with your actual model and data code
3. Keep all file paths and JSON keys unchanged for system compatibility
4. Run with: python model_template.py

The optimizer will automatically handle hyperparameter generation and results analysis.
"""

import os
import json
import time
import random
import logging
from typing import Dict, Any, List, Union, Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------
# INTERFACE CONSTANTS
# --------------------
TEMP_DIR = os.environ.get("TEMP_DIR", "temp")
HYPERPARAMETERS_FILE = os.path.join(TEMP_DIR, "hyperparameters.json")
INFO_FILE = os.path.join(TEMP_DIR, "info.json")
RESULTS_FILE = os.path.join(TEMP_DIR, "results.json")

# --------------------
# UTILITY FUNCTIONS
# --------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across different frameworks.
    
    This function attempts to set seeds for:
    - Python random module
    - NumPy (if available)
    - PyTorch (if available)
    - TensorFlow (if available)
    """
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
        logger.info(f"Set NumPy random seed to {seed}")
    except ImportError:
        logger.info("NumPy not available - skipping NumPy seeding")
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set PyTorch random seed to {seed}")
    except ImportError:
        logger.info("PyTorch not available - skipping PyTorch seeding")
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info(f"Set TensorFlow random seed to {seed}")
    except ImportError:
        logger.info("TensorFlow not available - skipping TensorFlow seeding")

def safe_type_convert(value: Any, target_type: type, default: Any) -> Any:
    """
    Safely convert value to target type with fallback to default.
    
    Args:
        value: Value to convert
        target_type: Target type (int, float, str, bool)
        default: Default value if conversion fails
        
    Returns:
        Converted value or default
    """
    try:
        if target_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        elif target_type == int:
            return int(float(value))  # Handle "32.0" -> 32
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return str(value)
        else:
            return target_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Type conversion failed for {value} to {target_type}: {e}. Using default: {default}")
        return default

def load_json_safe(filepath: str, default: Dict = None) -> Dict:
    """
    Safely load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Loaded JSON data or default
    """
    if default is None:
        default = {}
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {filepath}")
        return data
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}. Using default values.")
        return default
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}. Using default values.")
        return default
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}. Using default values.")
        return default

def save_json_safe(filepath: str, data: Dict) -> bool:
    """
    Safely save JSON file with error handling.
    
    Args:
        filepath: Path to save JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

# --------------------
# CONFIGURATION LOADING
# --------------------

def load_hyperparameters() -> Dict[str, Any]:
    """
    Load hyperparameters from temp/hyperparameters.json with safe defaults.
    
    The optimizer writes hyperparameters to this file in JSON format.
    This function applies type conversion and validation.
    
    Returns:
        Dictionary of hyperparameters with safe defaults
    """
    logger.info("Loading hyperparameters...")
    
    # Safe default hyperparameters - customize these for your model
    default_hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
        "seed": 42,
        "optimizer": "adam",
        "scheduler": "none",
        "momentum": 0.9,
        "use_augmentation": False,
        "hidden_size": 128,
        "num_layers": 2,
        "activation": "relu"
    }
    
    # Load from file
    loaded_hyperparameters = load_json_safe(HYPERPARAMETERS_FILE, {})
    
    # Merge with defaults and apply type conversion
    hyperparameters = {}
    
    for key, default_value in default_hyperparameters.items():
        raw_value = loaded_hyperparameters.get(key, default_value)
        
        # Apply type conversion based on default value type
        if isinstance(default_value, bool):
            hyperparameters[key] = safe_type_convert(raw_value, bool, default_value)
        elif isinstance(default_value, int):
            hyperparameters[key] = safe_type_convert(raw_value, int, default_value)
        elif isinstance(default_value, float):
            hyperparameters[key] = safe_type_convert(raw_value, float, default_value)
        else:
            hyperparameters[key] = safe_type_convert(raw_value, str, str(default_value))
    
    # Add any additional hyperparameters from the file that aren't in defaults
    for key, value in loaded_hyperparameters.items():
        if key not in hyperparameters:
            hyperparameters[key] = value
    
    # Override with environment variables if present
    for key in hyperparameters.keys():
        env_value = os.environ.get(key.upper())
        if env_value is not None:
            original_type = type(hyperparameters[key])
            hyperparameters[key] = safe_type_convert(env_value, original_type, hyperparameters[key])
            logger.info(f"Override {key} with environment variable: {hyperparameters[key]}")
    
    logger.info(f"Final hyperparameters: {hyperparameters}")
    return hyperparameters

def load_configuration() -> Dict[str, Any]:
    """
    Load configuration from temp/info.json to get primary metric name and other settings.
    
    Returns:
        Configuration dictionary with primary metric info
    """
    logger.info("Loading configuration...")
    
    default_config = {
        "metrics": {
            "primary_metric": "val_accuracy",
            "description": "Validation accuracy"
        },
        "model_info": "Model template",
        "optimization_goal": "Maximize validation accuracy"
    }
    
    config = load_json_safe(INFO_FILE, default_config)
    
    # Extract primary metric name
    primary_metric = config.get("metrics", {}).get("primary_metric", "val_accuracy")
    logger.info(f"Primary metric: {primary_metric}")
    
    return config

# --------------------
# DATA LOADING STUB
# --------------------

def load_data(hyperparameters: Dict[str, Any]) -> tuple:
    """
    Load and prepare your data here.
    
    REPLACE THIS SECTION WITH YOUR DATA LOADING CODE.
    
    Args:
        hyperparameters: Loaded hyperparameters dictionary
        
    Returns:
        Tuple of (train_data, val_data, test_data) or your preferred format
        
    Example implementations:
    
    # For image data:
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR10('data', train=True, transform=transform, download=True)
    val_data = datasets.CIFAR10('data', train=False, transform=transform)
    
    # For tabular data:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For text data:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # ... tokenization and dataset creation
    """
    logger.info("Loading data...")
    
    # PLACEHOLDER: Replace with your actual data loading
    batch_size = hyperparameters.get("batch_size", 32)
    
    # Simulate data sizes for demonstration
    train_samples = 1000
    val_samples = 200
    test_samples = 200
    
    logger.info(f"Data loaded - Train: {train_samples}, Val: {val_samples}, Test: {test_samples}")
    
    # Return placeholder data info
    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "num_classes": 10,  # Example for classification
        "feature_dim": 784   # Example feature dimension
    }

# --------------------
# MODEL DEFINITION STUB
# --------------------

def create_model(hyperparameters: Dict[str, Any], data_info: Dict) -> Any:
    """
    Create and initialize your model here.
    
    REPLACE THIS SECTION WITH YOUR MODEL DEFINITION CODE.
    
    Args:
        hyperparameters: Loaded hyperparameters dictionary
        data_info: Information about the loaded data
        
    Returns:
        Your initialized model
        
    Example implementations:
    
    # PyTorch model:
    import torch.nn as nn
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    model = SimpleNN(data_info['feature_dim'], hyperparameters['hidden_size'], 
                    data_info['num_classes'], hyperparameters['dropout_rate'])
    
    # TensorFlow/Keras model:
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hyperparameters['hidden_size'], activation='relu'),
        tf.keras.layers.Dropout(hyperparameters['dropout_rate']),
        tf.keras.layers.Dense(data_info['num_classes'], activation='softmax')
    ])
    
    # Scikit-learn model:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=hyperparameters.get('n_estimators', 100),
        max_depth=hyperparameters.get('max_depth', None),
        random_state=hyperparameters['seed']
    )
    """
    logger.info("Creating model...")
    
    # PLACEHOLDER: Replace with your actual model creation
    hidden_size = hyperparameters.get("hidden_size", 128)
    num_layers = hyperparameters.get("num_layers", 2)
    dropout_rate = hyperparameters.get("dropout_rate", 0.0)
    
    logger.info(f"Model architecture - Hidden size: {hidden_size}, Layers: {num_layers}, Dropout: {dropout_rate}")
    
    # Return placeholder model info
    return {
        "type": "placeholder_model",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "num_parameters": hidden_size * num_layers * 1000  # Simulated parameter count
    }

# --------------------
# TRAINING LOOP
# --------------------

def train_one_epoch(epoch: int, model: Any, data_info: Dict, hyperparameters: Dict[str, Any], 
                   primary_metric: str) -> Dict[str, float]:
    """
    Perform one training epoch and return metrics.
    
    REPLACE THIS SECTION WITH YOUR ACTUAL TRAINING STEP CODE.
    
    Args:
        epoch: Current epoch number (1-based)
        model: Your model object
        data_info: Data information
        hyperparameters: Hyperparameters dictionary
        primary_metric: Name of the primary metric to optimize
        
    Returns:
        Dictionary of metrics for this epoch
        
    Example implementations:
    
    # PyTorch training step:
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    # Validation step:
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            val_correct += pred.eq(target).sum().item()
            val_total += target.size(0)
    
    return {
        'train_loss': train_loss / len(train_loader),
        'train_accuracy': correct / total,
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': val_correct / val_total
    }
    """
    
    # PLACEHOLDER: Replace with your actual training logic
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    
    # Simulate training progress
    time.sleep(0.01)  # Simulate computation time
    
    # Generate realistic synthetic metrics that improve over epochs
    progress = min(epoch / hyperparameters.get("epochs", 20), 1.0)
    noise = (random.random() - 0.5) * 0.02  # Small random noise
    
    # Simulate different metric behaviors based on primary metric
    if "loss" in primary_metric.lower():
        # Loss metrics should decrease (lower is better)
        train_loss = max(0.01, 2.0 * (1 - progress * 0.8) + noise)
        val_loss = max(0.01, 2.2 * (1 - progress * 0.7) + noise * 1.5)
        primary_value = val_loss
        
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            primary_metric: float(primary_value)
        }
    else:
        # Accuracy/score metrics should increase (higher is better)
        base_accuracy = 0.1 + 0.8 * progress + noise
        train_acc = min(0.99, max(0.01, base_accuracy + 0.05))
        val_acc = min(0.95, max(0.01, base_accuracy))
        
        primary_value = val_acc
        
        metrics = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            primary_metric: float(primary_value)
        }
        
        # Add loss metrics for completeness
        train_loss = max(0.01, 2.0 * (1 - progress * 0.8) + noise)
        val_loss = max(0.01, 2.2 * (1 - progress * 0.7) + noise * 1.5)
        metrics.update({
            "train_loss": float(train_loss),
            "val_loss": float(val_loss)
        })
    
    # Add learning rate tracking
    metrics["learning_rate"] = float(learning_rate)
    
    logger.info(f"Epoch {epoch}: {primary_metric}={primary_value:.4f}")
    return metrics

def train_model(model: Any, data_info: Dict, hyperparameters: Dict[str, Any], 
               primary_metric: str) -> Dict[str, List[float]]:
    """
    Main training loop that collects metrics over all epochs.
    
    Args:
        model: Your model object
        data_info: Data information  
        hyperparameters: Hyperparameters dictionary
        primary_metric: Name of the primary metric to optimize
        
    Returns:
        Dictionary with metric trajectories over all epochs
    """
    logger.info("Starting training...")
    
    epochs = hyperparameters.get("epochs", 20)
    
    # Initialize metric collections
    all_metrics = {}
    epoch_logs = []
    
    for epoch in range(1, epochs + 1):
        # Train one epoch
        epoch_metrics = train_one_epoch(epoch, model, data_info, hyperparameters, primary_metric)
        epoch_logs.append(epoch_metrics)
        
        # Collect metrics for trajectories
        for metric_name, value in epoch_metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(float(value))
        
        # Optional: Early stopping logic
        if epoch > 5:  # Give some initial epochs
            recent_primary = all_metrics[primary_metric][-3:]
            if len(recent_primary) >= 3:
                # Check for stagnation (optional)
                if "loss" in primary_metric.lower():
                    # For loss, check if not improving (not decreasing)
                    if all(recent_primary[i] >= recent_primary[i+1] - 0.001 for i in range(len(recent_primary)-1)):
                        logger.info(f"Early stopping: {primary_metric} stagnated")
                else:
                    # For accuracy, check if not improving (not increasing)
                    if all(recent_primary[i] <= recent_primary[i+1] + 0.001 for i in range(len(recent_primary)-1)):
                        logger.info(f"Early stopping: {primary_metric} stagnated")
    
    # Log final performance
    final_primary = all_metrics[primary_metric][-1]
    logger.info(f"Training completed. Final {primary_metric}: {final_primary:.4f}")
    
    return all_metrics

# --------------------
# RESULTS SAVING
# --------------------

def save_results(all_metrics: Dict[str, List[float]], epochs_list: List[int], 
                primary_metric: str) -> bool:
    """
    Save results in the exact format expected by the LLM optimization system.
    
    CRITICAL: The results.json file must have this exact schema:
    {
        "metrics": {
            "primary_metric_name": [values...],
            "other_metric_name": [values...],
            ...
        },
        "epochs": [1, 2, 3, ...]
    }
    
    Args:
        all_metrics: Dictionary with metric trajectories
        epochs_list: List of epoch numbers
        primary_metric: Name of the primary metric
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Saving results...")
    
    # Ensure primary metric is included
    if primary_metric not in all_metrics:
        logger.error(f"Primary metric '{primary_metric}' not found in results!")
        return False
    
    # Prepare results in the exact expected format
    results = {
        "metrics": all_metrics,
        "epochs": epochs_list
    }
    
    # Save to the expected location
    success = save_json_safe(RESULTS_FILE, results)
    
    if success:
        logger.info(f"Results saved to {RESULTS_FILE}")
        logger.info(f"Primary metric ({primary_metric}) final value: {all_metrics[primary_metric][-1]:.4f}")
        
        # Optional: Save additional analysis info
        analysis_file = os.path.join(TEMP_DIR, "training_analysis.json")
        analysis = {
            "primary_metric": primary_metric,
            "final_value": all_metrics[primary_metric][-1],
            "best_value": max(all_metrics[primary_metric]) if "loss" not in primary_metric.lower() 
                         else min(all_metrics[primary_metric]),
            "total_epochs": len(epochs_list),
            "metrics_available": list(all_metrics.keys())
        }
        save_json_safe(analysis_file, analysis)
    
    return success

# --------------------
# MAIN EXECUTION
# --------------------

def main():
    """
    Main execution function that orchestrates the entire training workflow.
    
    This function follows the exact interface expected by the LLM optimization system:
    1. Load hyperparameters from temp/hyperparameters.json
    2. Load configuration to get primary metric name  
    3. Set up data and model
    4. Run training loop
    5. Save results to temp/results.json with exact schema
    """
    logger.info("=" * 50)
    logger.info("LLM Hyperparameter Optimization - Model Training")
    logger.info("=" * 50)
    
    try:
        # Step 1: Load hyperparameters from temp/hyperparameters.json
        hyperparameters = load_hyperparameters()
        
        # Step 2: Load configuration to get primary metric name from temp/info.json
        config = load_configuration()
        primary_metric = config.get("metrics", {}).get("primary_metric", "val_accuracy")
        
        # Step 3: Set random seed for reproducibility
        seed = hyperparameters.get("seed", 42)
        set_seed(seed)
        logger.info(f"Set random seed to {seed}")
        
        # Step 4: Load data (REPLACE THIS SECTION WITH YOUR DATA LOADING CODE)
        data_info = load_data(hyperparameters)
        
        # Step 5: Create model (REPLACE THIS SECTION WITH YOUR MODEL DEFINITION)
        model = create_model(hyperparameters, data_info)
        
        # Step 6: Run training loop and collect metrics
        all_metrics = train_model(model, data_info, hyperparameters, primary_metric)
        
        # Step 7: Prepare epoch list
        epochs_list = list(range(1, len(all_metrics[primary_metric]) + 1))
        
        # Step 8: Save results to temp/results.json with exact schema
        success = save_results(all_metrics, epochs_list, primary_metric)
        
        if success:
            logger.info("✅ Training completed successfully!")
            logger.info(f"📊 Primary metric ({primary_metric}): {all_metrics[primary_metric][-1]:.4f}")
            logger.info(f"📁 Results saved to: {RESULTS_FILE}")
        else:
            logger.error("❌ Failed to save results!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
