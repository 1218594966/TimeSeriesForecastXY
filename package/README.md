# XLSTM Time Series Forecasting

A modular implementation of multivariate time series forecasting using XLSTM models. This package provides tools for loading data, preprocessing, model training, evaluation, visualization, and prediction.

## Features

- Data loading and preprocessing for time series data
- XLSTM model implementation with time series decomposition
- Training and evaluation utilities
- Visualization tools with confidence intervals
- Standalone prediction script for forecasting with trained models

## Project Structure

```
├── data_utils.py       # Data loading, preprocessing, and sequence creation
├── model.py            # XLSTM model implementation
├── training.py         # Model training and evaluation functions
├── visualization.py    # Plotting and visualization functions
├── utils.py            # Utility functions for metrics, saving, and loading
├── predict.py          # Standalone prediction script
├── main.py             # Main script for training new models
├── __init__.py         # Package initialization
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training a New Model

```bash
python main.py --csv_file data.csv --past_days 30 --future_days 15 --hidden_dim 128 --num_layers 2 --dropout 0.2 --epochs 100 --batch_size 32 --learning_rate 0.001
```

Key parameters:
- `--csv_file`: Path to your CSV data file
- `--past_days`: Number of past days to use as context window
- `--future_days`: Number of future days to predict
- `--hidden_dim`: Hidden dimension size for XLSTM model
- `--num_layers`: Number of XLSTM layers
- `--epochs`: Number of training epochs

### Making Predictions with a Trained Model

```bash
python predict.py --model_path output_20250413_143357/xlstm_model.pth --new_data data.csv --past_days 30 --future_days 15
```

Key parameters:
- `--model_path`: Path to a trained model file
- `--new_data`: Path to new data for prediction
- `--past_days`: Context window size (must match the trained model)
- `--future_days`: Number of days to predict

## Output

Both training and prediction create timestamped output directories containing:
- Visualization plots
- CSV files with predictions and metrics
- Model files (for training)
- Configuration details

## Example

Train a model:
```bash
D:\Anaconda\envs\pytorch\python.exe main.py --csv_file data.csv --past_days 30 --future_days 15 --epochs 100
```

Make predictions:
```bash
D:\Anaconda\envs\pytorch\python.exe predict.py --model_path output_20250413_120000/xlstm_model.pth --new_data latest_stock_data.csv --future_days 30
```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torch
- einops