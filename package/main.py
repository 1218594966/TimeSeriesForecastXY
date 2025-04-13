#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime

import torch
import numpy as np
import pandas as pd

# Import from modules
from package.data_utils import load_and_preprocess_data, create_data_loaders
from model import XLSTM
from training import train_model, evaluate_model
from visualization import (
    plot_training_history,
    plot_time_series_comparison,
    plot_test_with_confidence
)
from utils import save_metrics_to_csv, save_training_history


def main(args):
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Load and preprocess data
    print(f"Loading CSV file: {args.csv_file}")
    df, df_scaled, feature_names, scaler = load_and_preprocess_data(args.csv_file)
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {feature_names}")

    # Create data loaders
    (X_train, y_train, X_test, y_test, X_train_dates, X_test_dates,
     y_train_dates, y_test_dates, train_loader, test_loader) = create_data_loaders(
        df_scaled, args.past_days, args.future_days, args.batch_size
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Initialize model
    input_dim = X_train.shape[2]  # Number of features
    output_dim = y_train.shape[2]  # Number of output features

    # Initialize XLSTM model
    model = XLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        context_points=args.past_days,
        target_points=args.future_days,
        num_blocks=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # Print model structure
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    print("\nStarting model training...")
    history = train_model(
        model, train_loader, optimizer, criterion, args.epochs, device
    )

    # Save model
    model_path = os.path.join(output_dir, 'xlstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': output_dim,
            'context_points': args.past_days,
            'target_points': args.future_days,
            'num_layers': args.num_layers
        }
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Plot and save training history
    plot_training_history(history, output_dir)
    save_training_history(history, output_dir)

    # Evaluate model on test set
    print("\nEvaluating model...")
    model.eval()

    # Get test set predictions
    y_test_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            # Ensure y_pred has correct shape (batch_size, future_days, features)
            if y_pred.shape[1] != args.future_days:
                y_pred = y_pred.transpose(1, 2)

            y_test_pred.append(y_pred.cpu().numpy())

    y_test_pred = np.vstack(y_test_pred)

    # Evaluate test set
    test_results, test_true_inv, test_pred_inv = evaluate_model(
        y_test.numpy(), y_test_pred, feature_names, scaler
    )

    # Print test set evaluation results
    print("\nTest set evaluation results:")
    print(f"Overall RMSE: {test_results['overall']['RMSE']:.4f}")
    print(f"Overall MAE: {test_results['overall']['MAE']:.4f}")
    print(f"Overall MAPE: {test_results['overall']['MAPE']:.2f}%")
    print(f"Overall R2: {test_results['overall']['R2']:.4f}")

    # Save test set evaluation metrics to CSV
    test_metrics_path = os.path.join(output_dir, 'test_metrics.csv')
    save_metrics_to_csv(test_results, test_metrics_path)

    # Process test set predictions for visualization
    print("\nPlotting test set predictions...")

    # Create dictionaries to store predictions by date
    test_pred_by_date = {}
    test_true_by_date = {}

    # Collect data organized by date, ensuring each date has only one prediction
    for i in range(len(y_test)):
        sample_dates = y_test_dates[i]
        true_sample = y_test[i].numpy()
        pred_sample = y_test_pred[i]

        for j in range(len(sample_dates)):
            date = sample_dates[j]
            # Only save the first prediction for each date
            if date not in test_pred_by_date:
                test_pred_by_date[date] = pred_sample[j]
                test_true_by_date[date] = true_sample[j]

    # Convert dictionaries to lists for plotting and further processing
    test_dates = list(test_pred_by_date.keys())
    test_true_values = np.array(list(test_true_by_date.values()))
    test_pred_values = np.array(list(test_pred_by_date.values()))

    # Sort by date
    sorted_indices = np.argsort(test_dates)
    sorted_test_dates = [test_dates[i] for i in sorted_indices]
    sorted_test_true = test_true_values[sorted_indices]
    sorted_test_pred = test_pred_values[sorted_indices]

    # Inverse transform to original scale
    sorted_test_true_inv = scaler.inverse_transform(sorted_test_true)
    sorted_test_pred_inv = scaler.inverse_transform(sorted_test_pred)

    # Plot test set results
    plot_time_series_comparison(
        sorted_test_true_inv, sorted_test_pred_inv, sorted_test_dates,
        feature_names, output_dir, title_prefix="test_"
    )

    # Process and evaluate the last prediction period
    print(f"\nEvaluating the last {args.future_days} days of test set...")

    # Get the last test sample
    last_test_sample = X_test[-1:].to(device)
    last_test_true = y_test[-1].numpy()
    last_test_dates = y_test_dates[-1]

    # Make prediction
    with torch.no_grad():
        last_test_pred = model(last_test_sample)
        # Ensure correct shape
        if last_test_pred.shape[1] != args.future_days:
            last_test_pred = last_test_pred.transpose(1, 2)
        last_test_pred = last_test_pred.cpu().numpy()[0]

    # Inverse transform
    last_test_true_inv = scaler.inverse_transform(last_test_true)
    last_test_pred_inv = scaler.inverse_transform(last_test_pred)

    # Save to CSV with dates as index
    last_test_df = pd.DataFrame(
        np.column_stack([last_test_true_inv, last_test_pred_inv]),
        index=last_test_dates,
        columns=[f"{col}_actual" for col in feature_names] + [f"{col}_predicted" for col in feature_names]
    )
    last_test_df.to_csv(os.path.join(output_dir, f'last_{args.future_days}_days_prediction.csv'))

    # Plot comparison of the last prediction period
    plot_time_series_comparison(
        last_test_true_inv, last_test_pred_inv, last_test_dates,
        feature_names, output_dir, title_prefix=f"last_{args.future_days}_days_"
    )

    # Plot test set with confidence intervals
    print("\nPlotting test set with confidence intervals...")
    plot_test_with_confidence(
        sorted_test_dates, sorted_test_true_inv, sorted_test_pred_inv,
        last_test_dates, last_test_true_inv, last_test_pred_inv,
        feature_names, scaler, output_dir, args.future_days
    )

    print(f"\nAll outputs saved to directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multivariate Multi-step XLSTM Time Series Forecasting')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--past_days', type=int, default=30, help='Number of past days for prediction')
    parser.add_argument('--future_days', type=int, default=15, help='Number of future days to predict')
    parser.add_argument('--hidden_dim', type=int, default=128, help='XLSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of XLSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()
    main(args)