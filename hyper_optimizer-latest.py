#!/usr/bin/env python3

import ollama
import json
import time
import re
import sys
import os
import numpy as np
from typing import Dict, List, Any, Tuple, Union

# Configuration
LLM_MODEL = os.environ.get("LLM_MODEL")
TEMP_DIR =  os.environ.get("TEMP_DIR", "temp")

def clean_response(response):
    """Remove thinking tags from LLM response if present."""
    if "</think>" in response:
        return response.split("</think>", 1)[1].strip()
    return response

def api_call(messages):
    """Make API call to Ollama with error handling."""
    
    try:
        result = ollama.chat(model=LLM_MODEL, messages=messages)
        print(f"Prompt sent to LLM: {messages[1]['content']}...")  # Print first 200 chars of prompt
        response = result['message']['content']
        return clean_response(response)
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def load_config():
    """Load info.json configuration with error handling."""
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

def load_main_info_json():
    """Load main info.json from root directory to get default hyperparameters."""
    filepath = "info.json"  # Main folder directory
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found in main directory")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        sys.exit(1)

def load_analysis_summary():
    """Load the structured analysis summary from result analyzer."""
    filepath = os.path.join(TEMP_DIR, "analysis_summary.json")
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: analysis_summary.json not found, using basic analysis")
        return {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON in analysis_summary.json")
        return {}

def get_metric_info(config: Dict) -> Dict[str, Any]:
    """Extract primary metric information from configuration (same as analyzer)."""
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
        "r2_score": {"higher_is_better": True, "target": 0.60, "format": ".4f"},
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

def get_hyperparameter_type(param_name: str, param_config: Dict) -> str:
    """Determine the type of hyperparameter based on its configuration."""
    if "type" in param_config:
        return param_config["type"]
    
    # Infer type from other attributes
    if "options" in param_config or "values" in param_config:
        options = param_config.get("options", param_config.get("values", []))
        if options and isinstance(options[0], str):
            return "categorical"
        else:
            return "ordinal"
    elif "range" in param_config:
        if param_name in ['epochs', 'num_layers', 'hidden_size', 'max_depth']:
            return "integer"
        else:
            return "float"
    else:
        # Default inference based on parameter name
        if param_name in ['epochs', 'num_layers', 'hidden_size', 'max_depth', 'batch_size']:
            return "integer"
        elif param_name in ['hidden_dims', 'architecture', 'activation']:
            return "categorical"
        else:
            return "float"

def get_focused_hyperparameters_info(temp_config: Dict, main_config: Dict) -> Dict[str, Any]:
    """
    Extract hyperparameters to optimize from temp/info.json and get their defaults from main info.json.
    Returns info about which hyperparameters to focus on and their current defaults.
    """
    # Get hyperparameters to optimize from temp/info.json
    hyperparams_to_optimize = temp_config.get("hyperparameters", {})
    
    # Get default hyperparameters from main info.json
    main_hyperparams = main_config.get("hyperparameters", {})
    
    # Create focused info
    focused_info = {
        "to_optimize": hyperparams_to_optimize,
        "all_defaults": main_hyperparams,
        "fixed_params": {},
        "optimize_params": {}
    }
    
    # Separate fixed vs optimizable parameters
    for param_name, param_config in main_hyperparams.items():
        if param_name in hyperparams_to_optimize:
            # Extract range info from temp config, but use main config structure
            temp_param_info = hyperparams_to_optimize[param_name]
            
            # Handle different range formats (array vs min/max)
            if isinstance(temp_param_info.get("range"), list):
                range_info = {
                    "min": temp_param_info["range"][0],
                    "max": temp_param_info["range"][1]
                }
            else:
                range_info = temp_param_info.get("range", {"min": 0.001, "max": 0.1})
            
            # Determine parameter type
            param_type = get_hyperparameter_type(param_name, temp_param_info)
            
            # Include type, options, and range for parameters
            param_info = {
                "current_default": param_config.get("default", 0.01),
                "range": range_info,
                "type": param_type
            }
            
            # Add options for categorical/ordinal parameters
            if "options" in temp_param_info:
                param_info["options"] = temp_param_info["options"]
            elif "values" in temp_param_info:
                param_info["options"] = temp_param_info["values"]
            elif "options" in param_config:
                param_info["options"] = param_config["options"]
            elif "values" in param_config:
                param_info["options"] = param_config["values"]
            
            focused_info["optimize_params"][param_name] = param_info
        else:
            # For fixed parameters, get the default value with proper type handling
            default_value = param_config.get("default", 0.01)
            param_type = get_hyperparameter_type(param_name, param_config)
            
            # Handle different parameter types for fixed parameters
            if param_type == "categorical":
                # Keep string values as-is
                focused_info["fixed_params"][param_name] = default_value
            elif param_type == "ordinal" or param_type == "integer":
                # Ensure integer values
                if isinstance(default_value, (int, float)):
                    focused_info["fixed_params"][param_name] = int(default_value)
                else:
                    focused_info["fixed_params"][param_name] = default_value
            else:
                # Float parameters
                if isinstance(default_value, (int, float)):
                    focused_info["fixed_params"][param_name] = float(default_value)
                else:
                    focused_info["fixed_params"][param_name] = default_value
    
    # Debug print to verify extraction
    print(f"DEBUG - Fixed params: {focused_info['fixed_params']}")
    print(f"DEBUG - Optimize params: {focused_info['optimize_params']}")
    
    return focused_info

def generate_smart_hyperparameters(config: Dict, analysis_summary: Dict, main_config: Dict) -> str: 
    """Create a focused prompt for specific hyperparameter optimization.""" 
     
    # Get focused hyperparameter information 
    focused_info = get_focused_hyperparameters_info(config, main_config) 
    
    # Get metric information
    metric_info = analysis_summary.get("metric_info", {})
    if not metric_info:
        # Fallback to extracting from config
        metric_info = get_metric_info(main_config)
     
    # Extract key information from analysis 
    analysis = analysis_summary.get("analysis", "No analysis available") 
    performance_status = analysis_summary.get("performance_status", {}) 
    latest_hparams = analysis_summary.get("latest_hyperparameters", {}) 
    target_gap = analysis_summary.get("target_gap", 0) 
     
    current_metric_val = performance_status.get("current_primary", 0) 
    performance_level = performance_status.get("performance_level", "unknown") 
    trend = performance_status.get("status", "unknown") 
     
    # Create parameter information strings 
    fixed_params_str = "" 
    if focused_info["fixed_params"]: 
        fixed_params_list = [f"{param}: {value}" for param, value in focused_info["fixed_params"].items()] 
        fixed_params_str = "**Fixed Parameters (DO NOT CHANGE):**\n" + "\n".join(fixed_params_list) + "\n\n" 
     
    optimize_params_str = "" 
    if focused_info["optimize_params"]: 
        optimize_list = [] 
        for param, info in focused_info["optimize_params"].items():
            current_val = latest_hparams.get(param, info["current_default"])
            param_type = info.get("type", "float")
            
            # Handle different parameter types for display
            if param_type == "categorical":
                options = info.get("options", info.get("values", []))
                if options:
                    options_str = ", ".join([f'"{opt}"' if isinstance(opt, str) else str(opt) for opt in options])
                    optimize_list.append(f'{param}: choose from [{options_str}] (current: "{current_val}")')
                else:
                    optimize_list.append(f'{param}: categorical (current: "{current_val}")')
            elif param_type == "ordinal":
                options = info.get("options", info.get("values", []))
                if options:
                    options_str = ", ".join(map(str, options))
                    optimize_list.append(f"{param}: choose from [{options_str}] (current: {current_val})")
                else:
                    range_info = info.get("range", {})
                    range_min = range_info.get("min", "unknown")
                    range_max = range_info.get("max", "unknown")
                    optimize_list.append(f"{param}: {range_min} to {range_max} integers (current: {current_val})")
            else:
                # Handle continuous range parameters (float/integer)
                range_info = info.get("range", {})
                if isinstance(range_info, dict):
                    range_min = range_info.get("min", "unknown")
                    range_max = range_info.get("max", "unknown")
                elif isinstance(range_info, list) and len(range_info) == 2:
                    range_min, range_max = range_info
                else:
                    range_min = range_max = "unknown"
                
                optimize_list.append(f"{param}: {range_min} to {range_max} (current: {current_val})")
        
        optimize_params_str = "**Parameters to Optimize:**\n" + "\n".join(optimize_list)
    
    # Create metric-aware strategic prompt
    metric_name = metric_info.get("name", "val_accuracy")
    target_value = metric_info.get("target", 0.90)
    higher_is_better = metric_info.get("higher_is_better", True)
    target_direction = "exceed" if higher_is_better else "be below"
    comparison_symbol = ">" if higher_is_better else "<"
    
    prompt = [
        {
            'role': 'system',
            'content': (
                f"You are an expert in machine learning hyperparameter tuning. "
                f"You specialize in analyzing training patterns and proposing optimized hyperparameters "
                f"to improve {metric_name} performance. Your goal is to make {metric_name} {target_direction} {target_value:.3f}. "
                f"Remember: for {metric_name}, {'higher values are better' if higher_is_better else 'lower values are better'}. "
                f"Respond precisely and strictly follow formatting."
            )
        },
        {
            'role': 'user',
            'content': f"""🎯 **Optimization Goal:**  
Make {metric_name} {comparison_symbol} {target_value:.3f}

🧠 **Current Performance:**  
- Current {metric_name}: {current_metric_val:.4f} ({performance_level.upper()})
- Gap to Target: {target_gap:.4f}
- Performance Trend: {trend.upper()}

📂 **Latest Analysis (FOLLOW THIS STRICTLY):**  
{analysis}

🎚️ **Hyperparameter Configuration:**  
{fixed_params_str}{optimize_params_str}

---

🧪 **Your Task:**
Analyze the training performance and propose new values ONLY for the parameters marked "to optimize" above.
- Your decision **must be directly based on the latest analysis** (very important!)
- Do **not** suggest random or arbitrary values
- Suggest values only within the specified ranges
- Focus ONLY on the parameters that need optimization
- For categorical parameters, choose from the exact options provided
- Prioritize improving {metric_name} toward {target_value:.3f}
- Remember: {metric_name} {'should be maximized' if higher_is_better else 'should be minimized'}

📝 **Response Format (STRICT):**
**Reasoning:** [Brief explanation based on analysis and {metric_name} optimization]
**Hyperparameters:** {', '.join([f"{param}=<value>" for param in focused_info["optimize_params"].keys()])}"""
        }
    ]
    
    # Actually call the LLM and return the response
    return api_call(prompt)

def parse_hyperparameter_value(param_name: str, raw_value: str, param_info: Dict) -> Union[int, float, str]:
    """Parse hyperparameter value based on its type and constraints."""
    param_type = param_info.get("type", "float")
    
    try:
        if param_type == "categorical":
            # For categorical parameters, handle both string and numeric values
            value = raw_value.strip().strip('"').strip("'")
            
            # Try to convert to the appropriate type (int or float)
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string if conversion fails
            
            # Validate against options if available
            options_key = "options" if "options" in param_info else "values"
            if options_key in param_info:
                options = param_info[options_key]
                if value in options:
                    return value
                else:
                    # Try to find exact match by converting types
                    for option in options:
                        if str(option) == str(value):
                            return option
                    
                    # If no match found, use the default value from info.json instead of first option
                    default_value = param_info.get("current_default")
                    if default_value is not None and default_value in options:
                        print(f"⚠️ Value '{value}' not in options {options}, using default: {default_value}")
                        return default_value
                    else:
                        # If default is also not in options, use first option as last resort
                        print(f"⚠️ Value '{value}' not in options {options}, default not found, using first option: {options[0]}")
                        return options[0]
            return value
            
        elif param_type == "ordinal":
            # For ordinal parameters, convert to int and validate against options
            try:
                value = int(float(raw_value))  # Handle cases like "64.0"
            except ValueError:
                value = int(param_info.get("current_default", 32))
            
            options_key = "options" if "options" in param_info else "values"
            if options_key in param_info:
                valid_options = param_info[options_key]
                if value in valid_options:
                    return value
                else:
                    # Find closest valid option
                    closest_option = min(valid_options, key=lambda x: abs(x - value))
                    return closest_option
            
            # Apply range constraints if no options
            range_info = param_info.get("range", {})
            if "min" in range_info and value < range_info["min"]:
                value = int(range_info["min"])
            if "max" in range_info and value > range_info["max"]:
                value = int(range_info["max"])
            
            return max(1, value)  # Ensure positive
            
        elif param_type == "integer":
            # For integer parameters
            try:
                value = int(float(raw_value))  # Handle cases like "100.0"
            except ValueError:
                value = int(param_info.get("current_default", 1))
            
            # Apply range constraints
            range_info = param_info.get("range", {})
            if "min" in range_info and value < range_info["min"]:
                value = int(range_info["min"])
            if "max" in range_info and value > range_info["max"]:
                value = int(range_info["max"])
            
            return max(1, value)  # Ensure positive
            
        else:  # float type
            try:
                value = float(raw_value)
            except ValueError:
                value = float(param_info.get("current_default", 0.01))
            
            # Apply range constraints
            range_info = param_info.get("range", {})
            if "min" in range_info and value < range_info["min"]:
                value = float(range_info["min"])
            if "max" in range_info and value > range_info["max"]:
                value = float(range_info["max"])
            
            # Don't override with arbitrary minimum - respect the parsed value if it's within range
            return value if value > 0 else float(param_info.get("current_default", 0.01))
            
    except Exception as e:
        print(f"Error parsing {param_name} value '{raw_value}': {e}")
        # Return default value with proper type
        default_val = param_info.get("current_default", 0.01)
        if param_type == "categorical":
            return str(default_val)
        elif param_type in ["ordinal", "integer"]:
            return int(default_val) if isinstance(default_val, (int, float)) else 1
        else:
            return float(default_val) if isinstance(default_val, (int, float)) else 0.01

def extract_hyperparameters_enhanced(response_text: str, focused_info: Dict) -> Dict[str, Any]:
    """Enhanced hyperparameter extraction focused only on parameters to optimize with robust error handling."""
    
    # Initialize result dictionary with fixed parameters first
    extracted_params = {}
    
    # Always start with fixed parameters (unchanged)
    for param_name, value in focused_info["fixed_params"].items():
        extracted_params[param_name] = value
    
    if not response_text:
        print("Error: Empty response from LLM - using defaults for optimizable parameters")
        # Add defaults for optimizable parameters
        for param, info in focused_info["optimize_params"].items():
            extracted_params[param] = parse_hyperparameter_value(param, str(info["current_default"]), info)
        return extracted_params
    
    print(f"DEBUG: Parsing response text: {response_text[:200]}...")
    
    # Track successfully extracted parameters
    successfully_extracted = set()
    
    # Extract optimizable parameters from response with comprehensive error handling
    for param_name, param_info in focused_info["optimize_params"].items():
        try:
            # Special handling for array/list parameters like mlp_hidden_size
            if param_name == "mlp_hidden_size" or (param_info.get("type") == "categorical" and 
                isinstance(param_info.get("values", [None])[0], list)):
                
                # Look for array patterns like [256,256,256] or [256, 256, 256]
                array_patterns = [
                    rf"{param_name}=\s*(\[[^\]]+\])",
                    rf"{param_name}:\s*(\[[^\]]+\])",
                    rf"{param_name}\s+(\[[^\]]+\])",
                    rf"**{param_name}:**\s*(\[[^\]]+\])",
                    rf"\*\*{param_name}\*\*:\s*(\[[^\]]+\])"
                ]
                
                found = False
                for pattern in array_patterns:
                    try:
                        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            raw_value = match.group(1).strip()
                            print(f"DEBUG: Found array match for {param_name}: '{raw_value}'")
                            
                            try:
                                # Parse the array string into actual list
                                # Remove brackets and split by comma
                                array_str = raw_value.strip('[]')
                                array_values = [int(x.strip()) for x in array_str.split(',') if x.strip()]
                                
                                # Validate against available options
                                if "values" in param_info and array_values in param_info["values"]:
                                    extracted_params[param_name] = array_values
                                    successfully_extracted.add(param_name)
                                    found = True
                                    print(f"✅ Extracted {param_name}: '{raw_value}' -> {array_values}")
                                    break
                                elif "values" in param_info:
                                    # Try to find closest match
                                    best_match = find_closest_array_match(array_values, param_info["values"])
                                    if best_match:
                                        extracted_params[param_name] = best_match
                                        successfully_extracted.add(param_name)
                                        found = True
                                        print(f"✅ Extracted {param_name} (closest match): '{raw_value}' -> {best_match}")
                                        break
                                else:
                                    # No validation needed, use as-is
                                    extracted_params[param_name] = array_values
                                    successfully_extracted.add(param_name)
                                    found = True
                                    print(f"✅ Extracted {param_name}: '{raw_value}' -> {array_values}")
                                    break
                                    
                            except Exception as parse_error:
                                print(f"❌ Error parsing array {param_name} from '{raw_value}': {parse_error}")
                                continue
                    except Exception as regex_error:
                        print(f"❌ Regex error for array {param_name}: {regex_error}")
                        continue
                
                # If no array pattern found, fall back to default
                if not found:
                    print(f"⚠️ Could not extract array {param_name} from LLM response - using default")
                    try:
                        default_value = param_info["current_default"]
                        extracted_params[param_name] = default_value
                        print(f"✅ Using default for {param_name}: {default_value}")
                    except Exception as default_error:
                        print(f"❌ Error using default for {param_name}: {default_error}")
                        # Emergency fallback
                        if "values" in param_info and param_info["values"]:
                            extracted_params[param_name] = param_info["values"][0]
                            print(f"🚑 Emergency fallback for {param_name}: {extracted_params[param_name]}")
            
            else:
                # Regular scalar parameter extraction
                patterns = [
                    rf"{param_name}:\s*([^\s,\n]+)",
                    rf"{param_name}\s*=\s*([^\s,\n]+)",
                    rf"{param_name}\s+([^\s,\n]+)",
                    rf"**{param_name}:**\s*([^\s,\n]+)",
                    rf"\*\*{param_name}\*\*:\s*([^\s,\n]+)"
                ]
                
                found = False
                for pattern in patterns:
                    try:
                        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            raw_value = match.group(1).strip()
                            
                            # Remove trailing punctuation and quotes
                            raw_value = raw_value.rstrip('.,;').strip('"').strip("'")
                            
                            try:
                                parsed_value = parse_hyperparameter_value(param_name, raw_value, param_info)
                                extracted_params[param_name] = parsed_value
                                successfully_extracted.add(param_name)
                                found = True
                                
                                # Debug output with range info
                                if "options" in param_info:
                                    range_str = f"(options: {param_info['options']})"
                                elif "values" in param_info:
                                    range_str = f"(values: {param_info['values']})"
                                else:
                                    range_info = param_info.get("range", {})
                                    range_str = f"(range: [{range_info.get('min', 'N/A')}, {range_info.get('max', 'N/A')}])"
                                print(f"✅ Extracted {param_name}: '{raw_value}' -> {parsed_value} {range_str}")
                                break
                            except Exception as parse_error:
                                print(f"❌ Error parsing {param_name} from '{raw_value}': {parse_error}")
                                continue
                    except Exception as regex_error:
                        print(f"❌ Regex error for {param_name}: {regex_error}")
                        continue
                
                # If extraction failed for this parameter, use default value from info.json
                if not found:
                    print(f"⚠️ Could not extract {param_name} from LLM response - using default from info.json")
                    try:
                        default_value = param_info["current_default"]
                        parsed_value = parse_hyperparameter_value(param_name, str(default_value), param_info)
                        extracted_params[param_name] = parsed_value
                        print(f"✅ Using default for {param_name}: {parsed_value}")
                    except Exception as default_error:
                        print(f"❌ Error using default for {param_name}: {default_error}")
                        # Emergency fallback to basic defaults
                        if param_info.get("type") == "categorical" and "options" in param_info:
                            extracted_params[param_name] = param_info["options"][0]
                        elif param_info.get("type") == "categorical" and "values" in param_info:
                            extracted_params[param_name] = param_info["values"][0]
                        elif param_info.get("type") in ["integer", "ordinal"]:
                            extracted_params[param_name] = 1
                        else:
                            extracted_params[param_name] = 0.01
                        print(f"🚑 Emergency fallback for {param_name}: {extracted_params[param_name]}")
                        
        except Exception as param_error:
            print(f"❌ Critical error processing {param_name}: {param_error}")
            # Emergency fallback to basic defaults
            try:
                default_value = param_info["current_default"]
                extracted_params[param_name] = parse_hyperparameter_value(param_name, str(default_value), param_info)
                print(f"🚑 Emergency default for {param_name}: {extracted_params[param_name]}")
            except:
                # Last resort defaults
                if param_info.get("type") == "categorical" and "options" in param_info:
                    extracted_params[param_name] = param_info["options"][0]
                elif param_info.get("type") == "categorical" and "values" in param_info:
                    extracted_params[param_name] = param_info["values"][0]
                elif param_info.get("type") in ["integer", "ordinal"]:
                    extracted_params[param_name] = 1
                else:
                    extracted_params[param_name] = 0.01
                print(f"🚑 Last resort fallback for {param_name}: {extracted_params[param_name]}")
    
    # Final validation - ensure all required parameters are present
    try:
        all_required_params = set(focused_info["fixed_params"].keys()) | set(focused_info["optimize_params"].keys())
        missing_params = all_required_params - set(extracted_params.keys())
        
        if missing_params:
            print(f"⚠️ Missing parameters detected: {missing_params}")
            # Add missing parameters with defaults from info.json
            for param in missing_params:
                try:
                    if param in focused_info["fixed_params"]:
                        extracted_params[param] = focused_info["fixed_params"][param]
                        print(f"✅ Added missing fixed param {param}: {extracted_params[param]}")
                    elif param in focused_info["optimize_params"]:
                        info = focused_info["optimize_params"][param]
                        default_val = info["current_default"]
                        extracted_params[param] = parse_hyperparameter_value(param, str(default_val), info)
                        print(f"✅ Added missing optimizable param {param}: {extracted_params[param]}")
                except Exception as missing_error:
                    print(f"❌ Error adding missing param {param}: {missing_error}")
    
    except Exception as validation_error:
        print(f"❌ Error during final validation: {validation_error}")
    
    # Summary report
    total_optimizable = len(focused_info["optimize_params"])
    total_extracted = len(successfully_extracted)
    extraction_rate = (total_extracted / total_optimizable * 100) if total_optimizable > 0 else 0
    
    print(f"📊 Extraction Summary: {total_extracted}/{total_optimizable} parameters successfully extracted ({extraction_rate:.1f}%)")
    if total_extracted < total_optimizable:
        failed_params = set(focused_info["optimize_params"].keys()) - successfully_extracted
        print(f"📋 Parameters using defaults: {list(failed_params)}")
    
    print(f"DEBUG: Final extracted params: {extracted_params}")
    return extracted_params


def find_closest_array_match(target_array, valid_options):
    """Find the closest matching array from valid options."""
    if not valid_options or not target_array:
        return None
    
    # First, try exact match
    if target_array in valid_options:
        return target_array
    
    # If target array length matches any option, prefer those
    target_len = len(target_array)
    same_length_options = [opt for opt in valid_options if len(opt) == target_len]
    
    if same_length_options:
        # Find option with closest sum (as a simple similarity metric)
        target_sum = sum(target_array)
        closest = min(same_length_options, key=lambda x: abs(sum(x) - target_sum))
        return closest
    
    # Otherwise, return the first option as fallback
    return valid_options[0]
def apply_situation_based_adjustments(hyperparams: Dict[str, Any], analysis_summary: Dict, focused_info: Dict) -> Dict[str, Any]:
    """Apply situation-based adjustments only to optimizable parameters based on metric type."""
    
    # Create a copy to avoid modifying the original
    adjusted_hyperparams = hyperparams.copy()
    
    # Get metric information
    metric_info = analysis_summary.get("metric_info", {})
    higher_is_better = metric_info.get("higher_is_better", True)
    metric_name = metric_info.get("name", "val_accuracy")
    
    performance_status = analysis_summary.get("performance_status", {})
    trend = performance_status.get("status", "unknown")
    overfitting_level = performance_status.get("overfitting_level", "UNKNOWN")
    stability = performance_status.get("stability", "unknown")
    
    # Only adjust parameters that are marked for optimization
    optimizable_params = set(focused_info["optimize_params"].keys())
    
    print(f"Applying adjustments for {metric_name} ({'higher is better' if higher_is_better else 'lower is better'})")
    
    # Apply adjustments based on performance trend
    if trend == "degrading":
        # Performance is getting worse - need to be more conservative
        if "learning_rate" in adjusted_hyperparams and "learning_rate" in optimizable_params:
            current_lr = adjusted_hyperparams["learning_rate"]
            if isinstance(current_lr, (int, float)) and current_lr > 0:
                new_lr = max(0.0001, current_lr * 0.7)
                adjusted_hyperparams["learning_rate"] = new_lr
                print(f"Applied degrading performance strategy: reduced learning_rate to {new_lr}")
    
    elif trend == "stagnant":
        # Performance plateaued - might need more aggressive optimization
        if "learning_rate" in adjusted_hyperparams and "learning_rate" in optimizable_params:
            current_lr = adjusted_hyperparams["learning_rate"]
            if isinstance(current_lr, (int, float)) and current_lr > 0:
                # Get the max constraint if available
                max_lr = focused_info["optimize_params"].get("learning_rate", {}).get("range", {}).get("max", 0.1)
                new_lr = min(current_lr * 1.2, max_lr)
                adjusted_hyperparams["learning_rate"] = new_lr
                print(f"Applied stagnant performance strategy: increased learning_rate to {new_lr}")
    
    # Overfitting adjustments
    if overfitting_level == "HIGH":
        print("High overfitting detected - applying regularization strategies")
        
        # Increase dropout if available
        if "dropout_rate" in adjusted_hyperparams and "dropout_rate" in optimizable_params:
            current_dropout = adjusted_hyperparams["dropout_rate"]
            if isinstance(current_dropout, (int, float)) and current_dropout >= 0:
                max_dropout = focused_info["optimize_params"].get("dropout_rate", {}).get("range", {}).get("max", 0.8)
                new_dropout = min(current_dropout * 1.3, max_dropout)
                adjusted_hyperparams["dropout_rate"] = new_dropout
                print(f"Increased dropout_rate to {new_dropout}")
        
        # Increase weight decay if available
        if "weight_decay" in adjusted_hyperparams and "weight_decay" in optimizable_params:
            current_wd = adjusted_hyperparams["weight_decay"]
            if isinstance(current_wd, (int, float)) and current_wd >= 0:
                max_wd = focused_info["optimize_params"].get("weight_decay", {}).get("range", {}).get("max", 0.01)
                new_wd = min(current_wd * 1.5, max_wd)
                adjusted_hyperparams["weight_decay"] = new_wd
                print(f"Increased weight_decay to {new_wd}")
        
        # For tree-based models, reduce max_depth
        if "max_depth" in adjusted_hyperparams and "max_depth" in optimizable_params:
            current_depth = adjusted_hyperparams["max_depth"]
            if isinstance(current_depth, int) and current_depth > 1:
                min_depth = focused_info["optimize_params"].get("max_depth", {}).get("range", {}).get("min", 1)
                new_depth = max(current_depth - 1, min_depth)
                adjusted_hyperparams["max_depth"] = new_depth
                print(f"Reduced max_depth to {new_depth} for overfitting control")
        
        # Reduce n_estimators for tree ensembles to prevent overfitting
        if "n_estimators" in adjusted_hyperparams and "n_estimators" in optimizable_params:
            current_est = adjusted_hyperparams["n_estimators"]
            if isinstance(current_est, int) and current_est > 50:
                min_est = focused_info["optimize_params"].get("n_estimators", {}).get("range", {}).get("min", 10)
                new_est = max(int(current_est * 0.8), min_est)
                adjusted_hyperparams["n_estimators"] = new_est
                print(f"Reduced n_estimators to {new_est} for overfitting control")
    
    elif overfitting_level == "LOW":
        print("Low overfitting detected - can increase model capacity")
        
        # Increase model capacity carefully
        if "max_depth" in adjusted_hyperparams and "max_depth" in optimizable_params:
            current_depth = adjusted_hyperparams["max_depth"]
            if isinstance(current_depth, int):
                max_depth = focused_info["optimize_params"].get("max_depth", {}).get("range", {}).get("max", 10)
                new_depth = min(current_depth + 1, max_depth)
                adjusted_hyperparams["max_depth"] = new_depth
                print(f"Increased max_depth to {new_depth} (low overfitting allows more capacity)")
        
        # Increase n_estimators for better performance
        if "n_estimators" in adjusted_hyperparams and "n_estimators" in optimizable_params:
            current_est = adjusted_hyperparams["n_estimators"]
            if isinstance(current_est, int):
                max_est = focused_info["optimize_params"].get("n_estimators", {}).get("range", {}).get("max", 200)
                new_est = min(current_est + 25, max_est)
                adjusted_hyperparams["n_estimators"] = new_est
                print(f"Increased n_estimators to {new_est} (low overfitting allows less regularization)")

        # Reduce regularization slightly if performance allows
        if "weight_decay" in adjusted_hyperparams and "weight_decay" in optimizable_params:
            current_wd = adjusted_hyperparams["weight_decay"]
            if isinstance(current_wd, (int, float)) and current_wd > 0:
                min_wd = focused_info["optimize_params"].get("weight_decay", {}).get("range", {}).get("min", 0)
                new_wd = max(current_wd * 0.8, min_wd)
                adjusted_hyperparams["weight_decay"] = new_wd
                print(f"Reduced weight_decay to {new_wd} (low overfitting allows less regularization)")
    
    return adjusted_hyperparams

def save_hyperparameters(hyperparams: Dict[str, Any]):
    """Save hyperparameters to file, appending new iteration data."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save to info.json (existing functionality)
    info_filepath = os.path.join(TEMP_DIR, "info.json")
    try:
        # Load existing data if the file exists
        if os.path.exists(info_filepath):
            with open(info_filepath, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {"previous_trajectories": []}

        # Append the new iteration data
        new_iteration = {
            "iteration": int(os.environ.get("ITERATION", "0")),
            "hyperparameters": hyperparams
        }
        existing_data["previous_trajectories"].append(new_iteration)

        # Save the updated data back to the file
        with open(info_filepath, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"Successfully saved hyperparameters to {info_filepath}")
    except Exception as e:
        print(f"Error saving hyperparameters to info.json: {e}")
    
    # Save to hyperparameters.json (new functionality)
    hyperparams_filepath = os.path.join(TEMP_DIR, "hyperparameters.json")
    try:
        # Save hyperparameters as a clean dictionary
        with open(hyperparams_filepath, "w") as f:
            json.dump(hyperparams, f, indent=4)
        
        print(f"Successfully saved hyperparameters dictionary to {hyperparams_filepath}")
    except Exception as e:
        print(f"Error saving hyperparameters to hyperparameters.json: {e}")

def save_optimization_log(hyperparams: Dict[str, Any], reasoning: str, analysis_summary: Dict):
    """Save optimization log for debugging and analysis."""
    log_entry = {
        "timestamp": time.time(),
        "iteration": int(os.environ.get("ITERATION", "0")),
        "generated_hyperparameters": hyperparams,
        "reasoning": reasoning,
        "situation_metrics": {
            "trend": analysis_summary.get("trend_metrics", {}),
            "overfitting": analysis_summary.get("overfitting_info", {}),
            "latest_performance": analysis_summary.get("latest_performance", {})
        }
    }
    
    log_filepath = os.path.join(TEMP_DIR, "optimization_log.json")
    
    # Load existing log or create new one
    try:
        with open(log_filepath, "r") as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = {"entries": []}
    
    log_data["entries"].append(log_entry)
    
    # Keep only last 20 entries
    if len(log_data["entries"]) > 20:
        log_data["entries"] = log_data["entries"][-20:]
    
    try:
        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=4)
        print(f"Optimization log saved to {log_filepath}")
    except Exception as e:
        print(f"Error saving optimization log: {e}")

def main():
    """Main execution function."""
    # Load configuration and pre-calculated analysis
    temp_config = load_config()  # temp/info.json
    main_config = load_main_info_json()  # main info.json
    analysis_summary = load_analysis_summary()
    
    # Get focused hyperparameter information
    focused_info = get_focused_hyperparameters_info(temp_config, main_config)
    
    # Handle fixed_params - check if it's a dict or list
    if isinstance(focused_info['fixed_params'], dict):
        fixed_param_names = list(focused_info['fixed_params'].keys())
    else:
        fixed_param_names = focused_info['fixed_params']  # If it's already a list
    
    # Handle optimize_params - check if it's a dict or list  
    if isinstance(focused_info['optimize_params'], dict):
        optimize_param_names = list(focused_info['optimize_params'].keys())
    else:
        optimize_param_names = focused_info['optimize_params']  # If it's already a list
    
    print(f"Fixed parameters: {fixed_param_names}")
    print(f"Parameters to optimize: {optimize_param_names}")
    print("Generating optimized hyperparameters based on pre-calculated analysis...")
    
    # Generate hyperparameters using focused approach
    response = generate_smart_hyperparameters(temp_config, analysis_summary, main_config)
    
    # Initialize hyperparams with a fallback structure
    hyperparams = {}
    
    # Always ensure we have fixed parameters
    for param, value in focused_info["fixed_params"].items():
        hyperparams[param] = value
    
    if not response:
        print("❌ Error: Failed to generate hyperparameters - using defaults")
        # Use fallback strategy with proper validation
        for param, info in focused_info["optimize_params"].items():
            default_val = info["current_default"]
            parsed_val = parse_hyperparameter_value(param, str(default_val), info)
            hyperparams[param] = parsed_val
    else:
        print("📝 LLM Response:")
        print(response)
        
        # Extract hyperparameters
        try:
            extracted_hyperparams = extract_hyperparameters_enhanced(response, focused_info)
            if extracted_hyperparams and isinstance(extracted_hyperparams, dict):
                hyperparams = extracted_hyperparams
                print(f"🔧 Extracted hyperparams: {hyperparams}")
            else:
                print("❌ Extraction failed - using fallback")
                # Fallback to defaults
                for param, info in focused_info["optimize_params"].items():
                    hyperparams[param] = parse_hyperparameter_value(param, str(info["current_default"]), info)
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
            # Fallback to defaults
            for param, info in focused_info["optimize_params"].items():
                hyperparams[param] = parse_hyperparameter_value(param, str(info["current_default"]), info)
        
        # Apply situation-based adjustments only to optimizable parameters
        try:
            print(f"🔧 Before adjustments: {hyperparams}")
            adjusted_hyperparams = apply_situation_based_adjustments(hyperparams, analysis_summary, focused_info)
            if adjusted_hyperparams and isinstance(adjusted_hyperparams, dict):
                hyperparams = adjusted_hyperparams
                print(f"🔧 After adjustments: {hyperparams}")
            else:
                print("⚠️ Adjustments returned invalid result - keeping original")
        except Exception as e:
            print(f"❌ Error during adjustments: {e}")
    
    # Final validation - ensure hyperparams is never None
    if not hyperparams or not isinstance(hyperparams, dict):
        print("❌ Critical error: hyperparams is None or invalid - creating emergency fallback")
        hyperparams = {}
        # Emergency fallback with all parameters
        for param, value in focused_info["fixed_params"].items():
            hyperparams[param] = value
        for param, info in focused_info["optimize_params"].items():
            hyperparams[param] = parse_hyperparameter_value(param, str(info["current_default"]), info)
    
    print(f"\n✅ Final hyperparameters: {hyperparams}")
    
    # Validate that hyperparams contains expected keys
    expected_keys = set(focused_info["fixed_params"].keys()) | set(focused_info["optimize_params"].keys())
    actual_keys = set(hyperparams.keys())
    
    if not expected_keys.issubset(actual_keys):
        missing = expected_keys - actual_keys
        print(f"⚠️ Warning: Missing expected parameters: {missing}")
    
    # Save results
    save_hyperparameters(hyperparams)
    save_optimization_log(hyperparams, response or "Fallback used", analysis_summary)
    
    print("✅ Hyperparameter optimization completed successfully!")


if __name__ == "__main__":
    main()