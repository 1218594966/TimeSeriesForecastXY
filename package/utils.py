#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd


def save_metrics_to_csv(results, output_path):
    """
    Save evaluation metrics to a CSV file

    Args:
        results (dict): Dictionary of evaluation metrics
        output_path (str): Path to save the CSV file
    """
    # Prepare data
    metrics_data = []
    for feature, metrics in results.items():
        metrics_data.append({
            'Feature': feature,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2']
        })

    # Create DataFrame and save
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


def save_training_history(history, output_dir):
    """
    Save training history data as CSV

    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save the CSV file
    """
    history_df = pd.DataFrame({
        'Epoch': range(1, len(history['train_loss']) + 1),
        'TrainLoss': history['train_loss']
    })

    csv_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(csv_path, index=False)
    print(f"Training history data saved to: {csv_path}")


def load_model(model_path, model_class, device):
    """
    Load a saved model

    Args:
        model_path (str): Path to the saved model
        model_class: Model class to instantiate
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        model: Loaded model
    """
    import torch

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract configuration
    config = checkpoint['config']

    # Create model with the same configuration
    model = model_class(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        context_points=config['context_points'],
        target_points=config['target_points'],
        num_blocks=config['num_layers']
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model



def predict_future(model, last_sequence, scaler, n_future, feature_names, device):
    """
    Predict future values based on the last available sequence

    Args:
        model: Trained model
        last_sequence: Last available sequence of data (context window)
        scaler: Scaler used for normalization
        n_future: Number of future time steps to predict
        feature_names: List of feature names
        device: Device to run predictions on ('cpu' or 'cuda')

    Returns:
        future_predictions: Inverse-transformed future predictions
    """
    import torch
    import numpy as np
    from datetime import timedelta

    # Set model to evaluation mode
    model.eval()
    # Convert DataFrame to numpy array first, then to tensor
    # This fixes the "could not determine the shape of object type 'DataFrame'" error
    last_sequence_np = last_sequence.values
    last_sequence_tensor = torch.FloatTensor(last_sequence_np).unsqueeze(0).to(device)
    # Perform prediction
    with torch.no_grad():
        future_pred = model(last_sequence_tensor)
        # Ensure correct shape
        if future_pred.shape[1] != n_future:
            future_pred = future_pred.transpose(1, 2)
        future_pred = future_pred.cpu().numpy()[0]

    # Inverse transform predictions
    future_pred_inv = scaler.inverse_transform(future_pred)

    # Create dates for future predictions
    last_date = last_sequence.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(n_future)]

    # Create DataFrame with predictions
    future_df = pd.DataFrame(
        future_pred_inv,
        index=future_dates,
        columns=feature_names
    )

    return future_df