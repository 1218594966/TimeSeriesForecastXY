#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from package.data_utils import load_and_preprocess_data
from package.model import XLSTM
from utils import load_model, predict_future


def main(args):
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"prediction_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, 'prediction_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Load data
    print(f"Loading new data for prediction: {args.new_data}")
    df, df_scaled, feature_names, scaler = load_and_preprocess_data(args.new_data)

    # Get last sequence for prediction
    n_past = args.past_days  # Context window size
    if len(df) < n_past:
        raise ValueError(f"Not enough data points in the new data. Need at least {n_past} points.")

    # Load the trained model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, XLSTM, device)
    model.eval()

    # Get the last sequence
    last_sequence = df_scaled.iloc[-n_past:]

    # Predict future values
    print(f"Predicting next {args.future_days} days...")
    future_predictions = predict_future(
        model, last_sequence, scaler, args.future_days, feature_names, device
    )

    # Save predictions to CSV
    predictions_path = os.path.join(output_dir, f'future_predictions_{args.future_days}_days.csv')
    future_predictions.to_csv(predictions_path)
    print(f"Future predictions saved to: {predictions_path}")

    # Plot predictions
    for feature in feature_names:
        plt.figure(figsize=(12, 6))

        # Plot historical data (last 2*n_past days)
        historical_data = df[feature].iloc[-2 * n_past:]
        plt.plot(historical_data.index, historical_data.values, 'b-',
                 label=f'Historical {feature}', linewidth=1.5)

        # Plot predictions
        plt.plot(future_predictions.index, future_predictions[feature].values, 'r-',
                 label=f'Predicted {feature}', linewidth=1.5)

        # Add vertical line at prediction start
        plt.axvline(x=df.index[-1], color='k', linestyle='--', alpha=0.5)

        plt.title(f'{feature} - Future Prediction ({args.future_days} days)')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(output_dir, f'future_prediction_{feature}.png'), dpi=300)
        plt.close()

    print(f"All prediction outputs saved to directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict future values using trained XLSTM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--new_data', type=str, required=True, help='Path to new data CSV file')
    parser.add_argument('--past_days', type=int, default=30, help='Number of past days for context window')
    parser.add_argument('--future_days', type=int, default=15, help='Number of future days to predict')

    args = parser.parse_args()
    main(args)