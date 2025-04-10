import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 设置中文字体（Windows 系统推荐使用 SimHei）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model, model_name, test_loader, scaler, target_feature_index, num_features):
    model.eval()
    all_targets = []
    all_predictions = []

    criterion = torch.nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, _ = model(inputs)
            pred = outputs[:, -1, :]
            loss = criterion(pred, targets)
            total_loss += loss.item()
            all_predictions.append(pred.numpy())
            all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    dummy_pred = np.zeros((len(predictions), num_features))
    dummy_target = np.zeros((len(targets), num_features))

    dummy_pred[:, :] = predictions
    dummy_target[:, :] = targets

    predictions = scaler.inverse_transform(dummy_pred)
    targets = scaler.inverse_transform(dummy_target)

    y_true = targets[:, target_feature_index]
    y_pred = predictions[:, target_feature_index]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} 评估:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return {
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "test_loss": total_loss / len(test_loader)
    }


def plot_rolling_predictions_all_features(model, name, test_data, scaler, seq_len, future_days, target_feature_index, num_features):
    model.eval()
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.numpy()

    initial = test_data[-(seq_len + future_days):-future_days].astype(np.float32)
    preds = []
    current = torch.tensor(initial)

    with torch.no_grad():
        for _ in range(future_days):
            output, _ = model(current.unsqueeze(0))
            next_pred = output[0, -1, :]
            preds.append(next_pred)
            current = torch.cat([current[1:], next_pred.unsqueeze(0)], dim=0)

    preds = torch.stack(preds).numpy()
    future_real = test_data[-future_days:]

    dummy_pred = np.zeros_like(future_real)
    dummy_pred[:, :] = preds
    preds_inverse = scaler.inverse_transform(dummy_pred)

    dummy_real = np.zeros_like(future_real)
    dummy_real[:, :] = future_real
    real_inverse = scaler.inverse_transform(dummy_real)

    # 可视化
    days = np.arange(1, future_days + 1)
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(days, real_inverse[:, target_feature_index], label='真实值', color='blue', marker='o')
    plt.plot(days, preds_inverse[:, target_feature_index], label='预测值', color='red', marker='x')

    # 误差线
    for i in range(future_days):
        plt.vlines(
            x=days[i],
            ymin=min(real_inverse[i, target_feature_index], preds_inverse[i, target_feature_index]),
            ymax=max(real_inverse[i, target_feature_index], preds_inverse[i, target_feature_index]),
            color='gray',
            linestyle='dotted',
            alpha=0.4
        )

    mae = mean_absolute_error(real_inverse[:, target_feature_index], preds_inverse[:, target_feature_index])
    rmse = np.sqrt(mean_squared_error(real_inverse[:, target_feature_index], preds_inverse[:, target_feature_index]))

    plt.title(f"{name} - 未来 {future_days} 天预测\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plt.xlabel("未来天数")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{name}_rolling_prediction.png", dpi=300)
    plt.close()

    # 返回预测、实际值和起始索引
    start_idx = len(test_data) - seq_len - future_days
    end_idx = len(test_data) - future_days - 1
    return preds_inverse, real_inverse, f"{start_idx}-{end_idx}"


def save_last_days_results_to_csv(preds_dict, reals_dict, index_dict, future_days):
    for name in preds_dict:
        preds, reals = preds_dict[name], reals_dict[name]
        columns = [f"Feature_{i+1}_Pred" for i in range(preds.shape[1])] + \
                  [f"Feature_{i+1}_Real" for i in range(reals.shape[1])]
        df = pd.DataFrame(np.hstack([preds, reals]), columns=columns)
        df.insert(0, "Initial_Index_Range", index_dict[name])
        df.to_csv(f"outputs/{name}_last_{future_days}_days_results.csv", index=False)


def save_initial_sequence_to_csv(test_data, range_str, seq_len, scaler, num_features):
    start, end = map(int, range_str.split('-'))
    sequence = test_data[start:end+1]
    dummy = np.zeros_like(sequence)
    dummy[:, :] = sequence
    sequence_unscaled = scaler.inverse_transform(dummy)
    df = pd.DataFrame(sequence_unscaled, columns=[f"Feature_{i+1}" for i in range(num_features)])
    df.to_csv("outputs/initial_sequence.csv", index=False)
