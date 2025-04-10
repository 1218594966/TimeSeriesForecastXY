import pandas as pd
import numpy as np
import chardet
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read())['encoding']


def auto_select_seq_len(input_size):
    if input_size <= 5:
        return 8
    elif input_size <= 10:
        return 16
    elif input_size <= 20:
        return 32
    return 64


def load_and_clean_data(file_path, selected_columns):
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    data = pd.read_csv(file_path, encoding=encoding, engine='python')
    data.columns = data.columns.str.strip()

    selected_columns = [col.strip().strip("'").strip('"') for col in selected_columns.split(',')]
    data = data[selected_columns]

    str_cols = data.select_dtypes(include=['object']).columns
    data[str_cols] = data[str_cols].replace('--', np.nan)
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    data = data[~(z_scores > 3).any(axis=1)]

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    data = data.dropna(axis=1)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(data)

    return data, dataset, scaler
