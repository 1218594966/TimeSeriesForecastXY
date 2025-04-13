#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_history(history, output_dir):
    """
    Plot training history

    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=1.5)
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()


def plot_time_series_comparison(true_values, pred_values, dates, feature_names, output_dir, title_prefix=""):
    """
    Plot time series prediction comparison and save data as CSV

    Args:
        true_values: Actual values
        pred_values: Predicted values
        dates: Dates for x-axis
        feature_names: List of feature names
        output_dir: Directory to save outputs
        title_prefix: Prefix for plot titles
    """
    # Ensure dates are datetime type
    if isinstance(dates[0], str):
        dates = [pd.to_datetime(d) for d in dates]

    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))

        # Plot using solid lines with different colors
        plt.plot(dates, true_values[:, i], 'b-', label='Actual', linewidth=1.5)
        plt.plot(dates, pred_values[:, i], 'r-', label='Predicted', linewidth=1.5)

        plt.title(f'{title_prefix}{feature} - Prediction Results')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)  # Reduce grid transparency for clearer visualization
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{title_prefix}prediction_{feature}.png'), dpi=300)
        plt.close()

        # Save data as CSV, using dates as index to ensure no duplicates
        plot_df = pd.DataFrame({
            'Actual': true_values[:, i],
            'Predicted': pred_values[:, i]
        }, index=dates)

        # Sort index for easier viewing
        plot_df = plot_df.sort_index()

        csv_path = os.path.join(output_dir, f'{title_prefix}prediction_{feature}.csv')
        plot_df.to_csv(csv_path)
        print(f"Plot data saved to: {csv_path}")


def plot_test_with_confidence(test_dates, test_true, test_pred, last_pred_dates,
                              last_pred_true, last_pred_pred, feature_names,
                              scaler, output_dir, future_days):
    """
    Plot test set with 95% confidence intervals, distinguishing between the main test set
    and the last prediction days, and save data as CSV

    Args:
        test_dates: Dates for test set
        test_true: Actual test values
        test_pred: Predicted test values
        last_pred_dates: Dates for last prediction period
        last_pred_true: Actual values for last prediction period
        last_pred_pred: Predicted values for last prediction period
        feature_names: List of feature names
        scaler: Scaler used for inverse transformation
        output_dir: Directory to save outputs
        future_days: Number of future days in the prediction
    """
    # Get test set error statistics
    errors = np.abs(test_pred - test_true)

    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(14, 6))

        # Main part of test set
        test_dates_main = test_dates[:-len(last_pred_dates)]
        test_true_main = test_true[:-(len(last_pred_dates)), i]
        test_pred_main = test_pred[:-(len(last_pred_dates)), i]

        # Plot main test set, using thin lines
        plt.plot(test_dates_main, test_true_main, 'b-', linewidth=1.5, label='Test Set Actual')
        plt.plot(test_dates_main, test_pred_main, 'r-', linewidth=1.5, label='Test Set Predicted')

        # Calculate test set error statistics
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])

        # Plot last prediction days, using different colored solid lines
        plt.plot(last_pred_dates, last_pred_true[:, i], 'g-', linewidth=1.5, label=f'Last {future_days} Days Actual')
        plt.plot(last_pred_dates, last_pred_pred[:, i], 'm-', linewidth=1.5, label=f'Last {future_days} Days Predicted')

        # Add confidence intervals (based on entire test set errors)
        upper_bound = test_pred_main + 1.96 * std_error
        lower_bound = test_pred_main - 1.96 * std_error
        plt.fill_between(test_dates_main, lower_bound, upper_bound,
                         color='r', alpha=0.1, label='95% Confidence Interval')

        # Add confidence intervals for last prediction days
        upper_bound_last = last_pred_pred[:, i] + 1.96 * std_error
        lower_bound_last = last_pred_pred[:, i] - 1.96 * std_error
        plt.fill_between(last_pred_dates, lower_bound_last, upper_bound_last,
                         color='m', alpha=0.1)

        # Add vertical line to separate
        if len(test_dates_main) > 0:
            plt.axvline(x=test_dates_main[-1], color='k', linestyle='--', alpha=0.5)

        plt.title(f'{feature} - Test Set Comparison (with Confidence Intervals)')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(output_dir, f'test_confidence_{feature}.png'), dpi=300)
        plt.close()

        # Save data as CSV
        # Main test set data - using dates as index to ensure no duplicates
        main_data = {
            'Actual': test_true_main,
            'Predicted': test_pred_main,
            'LowerBound': lower_bound,
            'UpperBound': upper_bound
        }
        main_df = pd.DataFrame(main_data, index=test_dates_main)
        main_df = main_df.sort_index()  # Ensure index is sorted

        # Last prediction days data - using dates as index to ensure no duplicates
        last_pred_data = {
            'Actual': last_pred_true[:, i],
            'Predicted': last_pred_pred[:, i],
            'LowerBound': lower_bound_last,
            'UpperBound': upper_bound_last
        }
        last_pred_df = pd.DataFrame(last_pred_data, index=last_pred_dates)
        last_pred_df = last_pred_df.sort_index()  # Ensure index is sorted

        # Save main test set data
        main_csv_path = os.path.join(output_dir, f'test_confidence_main_{feature}.csv')
        main_df.to_csv(main_csv_path)

        # Save last prediction days data
        last_pred_csv_path = os.path.join(output_dir, f'test_confidence_last_{future_days}_days_{feature}.csv')
        last_pred_df.to_csv(last_pred_csv_path)

        print(f"Confidence interval plot data saved to: {main_csv_path} and {last_pred_csv_path}")