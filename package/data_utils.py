#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def load_and_preprocess_data(csv_file):
    """
    Load and preprocess data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file

    Returns:
        tuple: (original_df, scaled_df, feature_names, scaler)
    """
    # Load data
    df = pd.read_csv(csv_file)

    # Process date column (if exists)
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break

    if date_col:
        print(f"Detected date column: {date_col}")
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        # If no date column, create a default date index
        print("No date column detected, creating default date index")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=len(df) - 1)
        date_range = pd.date_range(start=start_date, end=end_date, periods=len(df))
        df.index = date_range

    # Remove non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < len(df.columns):
        print(f"Removed non-numeric columns: {set(df.columns) - set(numeric_cols)}")
        df = df[numeric_cols]

    # Check and handle missing values
    if df.isnull().sum().sum() > 0:
        print("Detected missing values, using forward fill method")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)  # Handle missing values at the beginning

    # Save original feature names
    feature_names = df.columns.tolist()

    # Create normalizer and transform data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Put scaled data back into DataFrame to maintain index
    df_scaled = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)

    return df, df_scaled, feature_names, scaler


def create_sequences_with_dates(df, n_past, n_future):
    """
    Create input sequences and target sequences, preserving date information

    Args:
        df (DataFrame): DataFrame with time series data
        n_past (int): Number of past time steps (context window)
        n_future (int): Number of future time steps to predict

    Returns:
        tuple: (X, y, X_dates, y_dates)
    """
    data_values = df.values
    date_indices = df.index

    X, y = [], []
    X_dates, y_dates = [], []

    for i in range(len(data_values) - n_past - n_future + 1):
        X.append(data_values[i:i + n_past])
        y.append(data_values[i + n_past:i + n_past + n_future])

        X_dates.append(date_indices[i:i + n_past])
        y_dates.append(date_indices[i + n_past:i + n_past + n_future])

    return np.array(X), np.array(y), X_dates, y_dates


def create_data_loaders(df_scaled, n_past, n_future, batch_size):
    """
    Create train and test data loaders

    Args:
        df_scaled (DataFrame): Scaled DataFrame
        n_past (int): Number of past time steps
        n_future (int): Number of future time steps
        batch_size (int): Batch size

    Returns:
        tuple: Training and testing data with their corresponding dates and loaders
    """
    # Create sequences
    X, y, X_dates, y_dates = create_sequences_with_dates(df_scaled, n_past, n_future)
    print(f"Sequence shapes: X: {X.shape}, y: {y.shape}")

    # Split into train and test sets (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train_dates, X_test_dates = X_dates[:train_size], X_dates[train_size:]
    y_train_dates, y_test_dates = y_dates[:train_size], y_dates[train_size:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (X_train, y_train, X_test, y_test,
            X_train_dates, X_test_dates, y_train_dates, y_test_dates,
            train_loader, test_loader)