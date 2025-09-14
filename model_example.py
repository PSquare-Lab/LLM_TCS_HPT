#!/usr/bin/env python3
"""
Text Classification on SST2 (Stanford Sentiment Treebank binary classification) dataset using DistilBERT.
Following the standards and hyperparameters from the paper "AgentHPO: Large Language Model Agent for Hyper-Parameter Optimization".

- Strictly enforces hyperparameter ranges as specified in Table 2 of the paper
- Always uses binary mode for SST2 (no neutral examples)
- Validates all hyperparameters against the defined search space
- Saves accuracy as the primary metric as specified in the paper
"""

import os
import json
import sys
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import numpy as np
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create temp directory if it doesn't exist
temp_dir = os.environ.get('TEMP_DIR', '.')
os.makedirs(temp_dir, exist_ok=True)
DATA_DIR = "./data/stanfordSentimentTreebank"

def get_hyperparameters():
    """Read hyperparameters from a JSON file with strict SST2 constraints as per the paper."""
    hyperparameters_path = os.path.join(temp_dir, 'hyperparameters.json')
    try:
        with open(hyperparameters_path, 'r') as f:
            loaded_hyperparameters = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: hyperparameters.json not found or invalid. Using SST2 defaults within paper-specified ranges.")
        loaded_hyperparameters = {}

    # SST2-specific constraints from Table 2 in the paper
    valid_batch_sizes = [8, 16, 32, 64, 128]
    valid_activations = ['gelu', 'relu', 'silu']
    
    # Get and validate each hyperparameter against paper's specifications
    learning_rate = float(loaded_hyperparameters.get('learning_rate', 2e-5))
    # Learning rate is log-scale between 10^-6 and 10^-2 as per paper
    learning_rate = max(1e-6, min(1e-2, learning_rate))
    
    epochs = int(loaded_hyperparameters.get('epochs', 3))
    # Epochs must be between 1 and 4 (integer) as per paper
    epochs = max(1, min(4, epochs))
    
    # Batch size must be one of the specified ordinal values
    batch_size = int(loaded_hyperparameters.get('batch_size', 32))
    if batch_size not in valid_batch_sizes:
        # Find the closest valid batch size
        batch_size = min(valid_batch_sizes, key=lambda x: abs(x - batch_size))
        print(f"Warning: Batch size adjusted to {batch_size} to match paper's specified values {valid_batch_sizes}.")
    
    weight_decay = float(loaded_hyperparameters.get('weight_decay', 0.01))
    # Weight decay is log-scale between 10^-6 and 0.1 as per paper
    weight_decay = max(1e-6, min(0.1, weight_decay))
    
    # All dropout rates must be between 0 and 0.5 as per paper
    dropout_rate = max(0.0, min(0.5, float(loaded_hyperparameters.get('dropout_rate', 0.1))))
    attention_dropout = max(0.0, min(0.5, float(loaded_hyperparameters.get('attention_dropout', 0.1))))
    seq_classif_dropout = max(0.0, min(0.5, float(loaded_hyperparameters.get('seq_classif_dropout', 0.1))))
    
    activation = loaded_hyperparameters.get('activation', 'gelu')
    if activation not in valid_activations:
        print(f"Warning: Activation '{activation}' not in valid list {valid_activations}. Using 'gelu' instead.")
        activation = 'gelu'

    # SST2 is strictly binary classification as per paper Table 1
    binary_mode = True
    
    hyperparameters = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'max_length': int(loaded_hyperparameters.get('max_length', 128)),
        'warmup_steps': int(loaded_hyperparameters.get('warmup_steps', 500)),
        'adam_epsilon': float(loaded_hyperparameters.get('adam_epsilon', 1e-8)),
        'binary_mode': binary_mode,
        'dropout_rate': dropout_rate,
        'attention_dropout': attention_dropout,
        'seq_classif_dropout': seq_classif_dropout,
        'activation': activation,
    }
    
    print("Using SST2 binary classification as specified in the paper")
    print(f"Validated hyperparameters within paper-specified ranges: {hyperparameters}")
    
    return hyperparameters

def save_results(metrics, epochs):
    """Save training results to results.json with primary metric as accuracy."""
    results = {'metrics': metrics, 'epochs': epochs}
    try:
        with open(os.path.join(temp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")

class SentimentDataset(Dataset):
    """Custom dataset for Stanford Sentiment Treebank classification."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def map_sentiment_label(probability, binary_mode=False):
    """Map sentiment probability to class labels following SST2 specification."""
    prob = float(probability)
    
    if binary_mode:
        # SST-2 style binary classification as per paper
        # Remove neutral examples (0.4 < prob <= 0.6) to match GLUE SST-2
        if prob <= 0.4:
            return 0  # negative
        elif prob > 0.6:
            return 1  # positive
        else:
            return None  # neutral - exclude from SST-2 binary
    else:
        # Not used for SST2 per the paper
        raise ValueError("SST2 requires binary mode as specified in the paper")

def load_data():
    """Load the Stanford Sentiment Treebank dataset with SST2-specific processing."""
    # Get hyperparameters to confirm binary mode
    hyperparameters = get_hyperparameters()
    if not hyperparameters['binary_mode']:
        print("Warning: SST2 requires binary mode. Forcing binary_mode=True as per paper specifications.")
        hyperparameters['binary_mode'] = True
    
    try:
        # Load sentences
        sentences_path = os.path.join(DATA_DIR, 'datasetSentences.txt')
        sentences_df = pd.read_csv(sentences_path, sep='\t', skiprows=1, header=None, 
                                 names=['sentence_index', 'sentence'])
        
        # Fix sentence indices (they start from 1, but we need 0-based indexing)
        sentences_df['sentence_index'] = sentences_df['sentence_index'] - 1
        
        # Load sentiment labels
        labels_path = os.path.join(DATA_DIR, 'sentiment_labels.txt')
        labels_df = pd.read_csv(labels_path, sep='|', skiprows=1, header=None, 
                              names=['phrase_ids', 'sentiment_values'])
        
        # Load phrase dictionary
        dictionary_path = os.path.join(DATA_DIR, 'dictionary.txt')
        dictionary_df = pd.read_csv(dictionary_path, sep='|', header=None, 
                                  names=['phrase', 'phrase_id'])
        dictionary_df['phrase_id'] = dictionary_df['phrase_id'].astype(int)
        
        # Load dataset splits
        splits_path = os.path.join(DATA_DIR, 'datasetSplit.txt')
        splits_df = pd.read_csv(splits_path, skiprows=1, header=None, 
                              names=['sentence_index', 'splitset_label'])
        splits_df['sentence_index'] = splits_df['sentence_index'] - 1  # Convert to 0-based indexing
        
        print(f"Loaded SST2 dataset components: {len(sentences_df)} sentences, {len(labels_df)} labels")
        
    except Exception as e:
        raise ValueError(f"Error loading SST2 dataset files: {e}")
    
    # Create mappings
    sentiment_map = dict(zip(labels_df['phrase_ids'], labels_df['sentiment_values']))
    phrase_to_id = dict(zip(dictionary_df['phrase'], dictionary_df['phrase_id']))
    
    # Prepare the final dataset
    data = []
    skipped_sentences = 0
    
    for _, row in sentences_df.iterrows():
        sentence_idx = row['sentence_index']
        sentence_text = row['sentence'].strip()
        
        # Find corresponding phrase_id
        if sentence_text in phrase_to_id:
            phrase_id = phrase_to_id[sentence_text]
            if phrase_id in sentiment_map:
                sentiment_value = sentiment_map[phrase_id]
                sentiment_label = map_sentiment_label(sentiment_value, True)  # SST2 always uses binary
                
                # Skip neutral examples (SST-2 style)
                if sentiment_label is None:
                    continue
                
                # Find split label
                split_row = splits_df[splits_df['sentence_index'] == sentence_idx]
                if not split_row.empty:
                    split_label = split_row.iloc[0]['splitset_label']
                    
                    data.append({
                        'sentence': sentence_text,
                        'sentiment': sentiment_label,
                        'splitset_label': split_label,
                    })
                else:
                    skipped_sentences += 1
            else:
                skipped_sentences += 1
        else:
            skipped_sentences += 1
    
    print(f"SST2 processing complete: {len(data)} sentences kept, {skipped_sentences} skipped")
    
    df = pd.DataFrame(data)
    
    # Print dataset statistics as per paper's evaluation methodology
    print(f"SST2 Dataset split distribution (per paper specifications):")
    print(f"Train (split=1): {len(df[df['splitset_label'] == 1])}")
    print(f"Dev (split=3): {len(df[df['splitset_label'] == 3])}")
    print(f"Test (split=2): {len(df[df['splitset_label'] == 2])}")
    
    print(f"SST2 Label distribution (binary classification):")
    print(f"Negative (0): {len(df[df['sentiment'] == 0])} ({len(df[df['sentiment'] == 0])/len(df)*100:.1f}%)")
    print(f"Positive (1): {len(df[df['sentiment'] == 1])} ({len(df[df['sentiment'] == 1])/len(df)*100:.1f}%)")
    
    return df

def train_model():
    """Train the DistilBERT model following the SST2 implementation in the paper."""
    hyperparameters = get_hyperparameters()
    df = load_data()
    
    if df.empty:
        raise ValueError("No data loaded for SST2. Check dataset files per paper requirements.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data using standard SST2 splits as per paper
    X_train = df[df['splitset_label'] == 1]['sentence'].values
    y_train = df[df['splitset_label'] == 1]['sentiment'].values
    X_val = df[df['splitset_label'] == 3]['sentence'].values  # Development set
    y_val = df[df['splitset_label'] == 3]['sentiment'].values
    X_test = df[df['splitset_label'] == 2]['sentence'].values
    y_test = df[df['splitset_label'] == 2]['sentiment'].values
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, hyperparameters['max_length'])
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, hyperparameters['max_length'])
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, hyperparameters['max_length'])
    
    # Initialize model for binary classification as per paper Table 1
    print("Initializing DistilBERT for SST2 binary classification (2 classes) as specified in the paper")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,
        dropout=hyperparameters['dropout_rate'],
        attention_dropout=hyperparameters['attention_dropout'],
        seq_classif_dropout=hyperparameters['seq_classif_dropout']
    ).to(device)
    
    # Training arguments following paper's specifications
    training_args = TrainingArguments(
        output_dir=temp_dir,
        num_train_epochs=hyperparameters['epochs'],
        per_device_train_batch_size=hyperparameters['batch_size'],
        per_device_eval_batch_size=hyperparameters['batch_size'],
        learning_rate=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay'],
        adam_epsilon=hyperparameters['adam_epsilon'],
        warmup_steps=hyperparameters['warmup_steps'],
        logging_dir=os.path.join(temp_dir, 'logs'),
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        report_to=[],
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        lr_scheduler_type='linear',
        save_total_limit=2,
    )
    
    def compute_metrics(eval_pred):
        """Compute metrics with accuracy as the primary metric as specified in the paper."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Starting SST2 training with parameters from the paper...")
    
    # Train model
    train_result = trainer.train()
    
    # Evaluate on validation set
    print("Evaluating on validation set (dev)...")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    val_accuracy = val_results['eval_accuracy']
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    test_accuracy = test_results['eval_accuracy']
    
    # Create epoch list
    epoch_list = list(range(1, hyperparameters['epochs'] + 1))
    
    # Metrics following paper's evaluation methodology
    metrics = {
        'val_accuracy': val_accuracy,  # Primary metric as specified in paper
        'test_accuracy': test_accuracy,
        'hyperparameters': hyperparameters,
        'model_info': {
            'model_name': 'distilbert-base-uncased',
            'task': 'SST2',
            'num_classes': 2,
            'num_parameters': sum(p.numel() for p in model.parameters()),
        }
    }
    
    print(f"SST2 Results per paper specifications:")
    print(f"Validation Accuracy (Primary Metric): {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    save_results(metrics, epochs=epoch_list)
    
    return model, tokenizer

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed with error: {e}", file=sys.stderr)
        sys.exit(1)