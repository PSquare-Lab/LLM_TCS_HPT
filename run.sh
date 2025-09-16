#!/bin/bash

# Generic ML Hyperparameter Optimization Pipeline
# This script runs the optimization loop for any ML model.

# Configuration
export LLM_MODEL="qwen2.5-coder:32b"  # Default model, can be overridden by environment variable
export ITERATION=0
export MAX_ITERATIONS=10
export MIN_IMPROVEMENT_THRESHOLD=0.01
export PROJECT_NAME="Deepfm LLM: ${LLM_MODEL} Iterations: ${MAX_ITERATIONS}"
export TEMP_DIR="${TEMP_DIR:-temp}"
# export TEMP_DIR="temp"  # Directory for temporary files, can be overridden by environment variable
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1


echo "Starting Generic ML Hyperparameter Optimization Pipeline"
echo "========================================================="

# Check for required files
if [ ! -f "info.json" ]; then
    echo "Error: info.json not found!"
    echo "Please create info.json based on info_template.json"
    exit 1
fi

if [ ! -f "model.py" ]; then
    echo "Error: model.py not found!"
    echo "Please create your model.py script based on example_model.py"
    exit 1
fi

# Setup temp directory and copy info.json
echo "Setting up temp directory and copying configuration..."
mkdir -p ${TEMP_DIR}
cp info.json ${TEMP_DIR}/info.json
echo "Configuration copied to ${TEMP_DIR}/info.json"
echo "Note: All temporary files (hyperparameters.json, results.json) will be stored in ${TEMP_DIR}/ folder"

echo "Configuration loaded. Starting optimization loop..."

# Install basic dependencies (user may need to add more)
echo "Installing basic dependencies..."
#pip install torch matplotlib ollama > /dev/null 2>&1

# Optimization feedback loop
best_metric_value=999999  # Will be updated based on optimization goal
no_improvement_count=0

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    echo ""
    echo "=========================================="
    echo "Starting iteration $ITERATION"
    echo "=========================================="

    # Step 1: Generate hyperparameters
    echo "Step 1: Generating hyperparameters..."
    if ! python hyper_optimizer-latest.py; then
        echo "Error: Failed to generate hyperparameters"
        exit 1
    fi
    ollama stop $LLM_MODEL
    # Step 2: Train model
    echo "Step 2: Training model with generated hyperparameters..."
    if ! python model.py; then
        echo "Error: Training failed"
        exit 1
    fi

    # Step 3: Analyze results
    echo "Step 3: Analyzing training results..."
    export PREVIOUS_HYPERPARAMETERS=$(cat ${TEMP_DIR}/hyperparameters.json)
    if ! python3 Recommenderengine.py; then
        echo "Error: Failed to analyze results"
        exit 1
    fi

    # Step 4: Check for improvement
    if [ -f "${TEMP_DIR}/results.json" ]; then
        # Get the primary metric value from results
        current_metric=$(python -c "
import json
with open('${TEMP_DIR}/info.json', 'r') as f:
    config = json.load(f)
primary_metric = config.get('metrics', {}).get('primary_metric', 'val_loss')

with open('${TEMP_DIR}/results.json', 'r') as f:
    results = json.load(f)

metrics = results.get('metrics', {})
if primary_metric in metrics:
    values = metrics[primary_metric]
    if isinstance(values, list) and values:
        print(values[-1])  # Last value
    else:
        print(values)
else:
    print(999999)  # Large number if metric not found
")

        echo "Current metric value: $current_metric"
        echo "Best metric value so far: $best_metric_value"

        # Check if this is an improvement (assuming lower is better for most metrics)
        improvement=$(python -c "
import sys
current = float('$current_metric')
best = float('$best_metric_value')
improvement = best - current
print(improvement)
if improvement > float('$MIN_IMPROVEMENT_THRESHOLD'):
    sys.exit(0)  # Significant improvement
else:
    sys.exit(1)  # No significant improvement
")

        # if [ $? -eq 0 ]; then
        #     echo "Significant improvement found! ($improvement)"
        #     best_metric_value=$current_metric
        #     no_improvement_count=0
        # else
        #     echo "No significant improvement. Count: $((no_improvement_count + 1))"
        #     no_improvement_count=$((no_improvement_count + 1))
        # fi

        # # Early stopping check
        # if [ $no_improvement_count -ge 3 ]; then
        #     echo "No improvement for 3 consecutive iterations. Stopping optimization."
        #     break
        # fi

    else
        echo "Warning: ${TEMP_DIR}/results.json not found, continuing..."
    fi

    # Increment iteration counter
    ITERATION=$((ITERATION + 1))
    export ITERATION

    echo "Completed iteration $((ITERATION - 1))"
done

echo ""
echo "=========================================="
echo "Optimization completed!"
echo "Best metric value achieved: $best_metric_value"
echo "Total iterations: $ITERATION"
echo "=========================================="

# Copy final results back to main folder
echo "Copying final results to main folder..."
cp ${TEMP_DIR}/info.json info_final.json
echo "Final configuration saved as info_final.json"
echo "Temporary files remain in ${TEMP_DIR}/ folder:"
echo "- ${TEMP_DIR}/hyperparameters.json (latest hyperparameters)"
echo "- ${TEMP_DIR}/results.json (latest training results)"
echo "- ${TEMP_DIR}/info.json (working configuration file)"

# Plotting the graph of trajectories
if [ -f "plot_trajectories.py" ]; then
    echo "Plotting trajectories..."
    python plot_trajectories.py
fi

# Enhanced plotting with detailed analysis
if [ -f "enhanced_plotter.py" ]; then
    echo "Creating enhanced optimization analysis..."
    python enhanced_plotter.py
fi
