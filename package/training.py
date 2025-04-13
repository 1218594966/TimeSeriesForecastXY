#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def train_model(model, train_loader, optimizer, criterion, epochs, device):
    """
    Train the model

    Args:
        model: The model to train
        train_loader: DataLoader with training data
        optimizer: Optimizer
        criterion: Loss function
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        dict: Training history
    """
    train_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # Ensure y_pred and y_batch have consistent shapes (batch_size, future_days, features)
            if y_pred.shape != y_batch.shape:
                y_pred = y_pred.transpose(1, 2)

            loss = criterion(y_pred, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}')

    return {'train_loss': train_losses}


def evaluate_model(y_true, y_pred, feature_names, scaler):
    """
    Evaluate model performance, calculating multiple metrics

    Args:
        y_true: True values
        y_pred: Predicted values
        feature_names: List of feature names
        scaler: Scaler used for inverse transformation

    Returns:
        tuple: (results dict, inverse-transformed true values, inverse-transformed predictions)
    """
    # Inverse transform
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, y_true.shape[-1]))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))

    # Adjust shapes for per-feature metric calculation
    n_samples = y_true.shape[0]
    n_steps = y_true.shape[1]
    n_features = y_true.shape[2]

    y_true_reshaped = y_true_inv.reshape(n_samples * n_steps, n_features)
    y_pred_reshaped = y_pred_inv.reshape(n_samples * n_steps, n_features)

    # Calculate overall metrics
    results = {
        'overall': {
            'RMSE': np.sqrt(mean_squared_error(y_true_reshaped, y_pred_reshaped)),
            'MAE': mean_absolute_error(y_true_reshaped, y_pred_reshaped),
            'MAPE': mean_absolute_percentage_error(y_true_reshaped, y_pred_reshaped) * 100,
            'R2': r2_score(y_true_reshaped, y_pred_reshaped)
        }
    }

    # Calculate metrics for each feature
    for i, feature in enumerate(feature_names):
        y_true_feature = y_true_reshaped[:, i]
        y_pred_feature = y_pred_reshaped[:, i]

        # Skip MAPE calculation for features that may contain zero values (to avoid division by zero)
        try:
            mape = mean_absolute_percentage_error(y_true_feature, y_pred_feature) * 100
        except:
            mape = np.nan

        results[feature] = {
            'RMSE': np.sqrt(mean_squared_error(y_true_feature, y_pred_feature)),
            'MAE': mean_absolute_error(y_true_feature, y_pred_feature),
            'MAPE': mape,
            'R2': r2_score(y_true_feature, y_pred_feature)
        }

    return results, y_true_inv, y_pred_inv


def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on a validation set

    Args:
        model: The model to validate
        val_loader: DataLoader with validation data
        criterion: Loss function
        device: Device to validate on ('cpu' or 'cuda')

    Returns:
        float: Validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # Ensure y_pred and y_batch have consistent shapes
            if y_pred.shape != y_batch.shape:
                y_pred = y_pred.transpose(1, 2)

            # Calculate loss
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss


def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion,
                              max_epochs, patience, device):
    """
    Train the model with early stopping based on validation loss

    Args:
        model: The model to train
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        optimizer: Optimizer
        criterion: Loss function
        max_epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before stopping
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        dict: Training history including validation loss
    """
    train_losses = []
    val_losses = []

    # Initialize best validation loss and patience counter
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # Ensure consistent shapes
            if y_pred.shape != y_batch.shape:
                y_pred = y_pred.transpose(1, 2)

            loss = criterion(y_pred, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{max_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Load the best model state if early stopping occurred
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }


def create_train_val_test_split(X, y, X_dates, y_dates, train_ratio=0.7, val_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets

    Args:
        X: Input sequences
        y: Target sequences
        X_dates: Dates for input sequences
        y_dates: Dates for target sequences
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation

    Returns:
        tuple: Training, validation, and test data with their corresponding dates
    """
    # Calculate split indices
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Split data
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Split dates
    X_train_dates, y_train_dates = X_dates[:train_size], y_dates[:train_size]
    X_val_dates, y_val_dates = X_dates[train_size:train_size + val_size], y_dates[train_size:train_size + val_size]
    X_test_dates, y_test_dates = X_dates[train_size + val_size:], y_dates[train_size + val_size:]

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_dates, y_train_dates, X_val_dates, y_val_dates, X_test_dates, y_test_dates)


def create_data_loaders_with_validation(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """
    Create DataLoaders for training, validation, and test sets

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def predict_on_test_set(model, test_loader, device):
    """
    Make predictions on the test set

    Args:
        model: Trained model
        test_loader: DataLoader with test data
        device: Device to run predictions on

    Returns:
        tuple: (test_true, test_pred) as numpy arrays
    """
    model.eval()
    test_pred = []
    test_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)

            # Ensure correct shape
            if y_pred.shape[1] != y_batch.shape[1]:
                y_pred = y_pred.transpose(1, 2)

            # Collect predictions and true values
            test_pred.append(y_pred.cpu().numpy())
            test_true.append(y_batch.numpy())

    # Concatenate batches
    test_pred = np.vstack(test_pred)
    test_true = np.vstack(test_true)

    return test_true, test_pred


def evaluate_per_horizon(y_true, y_pred, feature_names, scaler, horizon_steps=None):
    """
    Evaluate model performance for each prediction horizon step

    Args:
        y_true: True values (shape: [n_samples, n_steps, n_features])
        y_pred: Predicted values (same shape as y_true)
        feature_names: List of feature names
        scaler: Scaler used for inverse transformation
        horizon_steps: List of specific horizon steps to evaluate, or None for all steps

    Returns:
        dict: Results for each horizon step
    """
    n_samples, n_steps, n_features = y_true.shape

    if horizon_steps is None:
        horizon_steps = range(n_steps)

    horizon_results = {}

    for step in horizon_steps:
        # Extract predictions and true values for this horizon step
        y_true_step = y_true[:, step, :]
        y_pred_step = y_pred[:, step, :]

        # Inverse transform
        y_true_inv = scaler.inverse_transform(y_true_step)
        y_pred_inv = scaler.inverse_transform(y_pred_step)

        # Calculate overall metrics for this horizon step
        step_results = {
            'overall': {
                'RMSE': np.sqrt(mean_squared_error(y_true_inv, y_pred_inv)),
                'MAE': mean_absolute_error(y_true_inv, y_pred_inv),
                'MAPE': mean_absolute_percentage_error(y_true_inv, y_pred_inv) * 100,
                'R2': r2_score(y_true_inv, y_pred_inv)
            }
        }

        # Calculate metrics for each feature
        for i, feature in enumerate(feature_names):
            y_true_feature = y_true_inv[:, i]
            y_pred_feature = y_pred_inv[:, i]

            # Skip MAPE calculation for features that may contain zero values
            try:
                mape = mean_absolute_percentage_error(y_true_feature, y_pred_feature) * 100
            except:
                mape = np.nan

            step_results[feature] = {
                'RMSE': np.sqrt(mean_squared_error(y_true_feature, y_pred_feature)),
                'MAE': mean_absolute_error(y_true_feature, y_pred_feature),
                'MAPE': mape,
                'R2': r2_score(y_true_feature, y_pred_feature)
            }

        horizon_results[f'horizon_{step + 1}'] = step_results

    return horizon_results