from src.data_preprocessing import load_and_clean_data, auto_select_seq_len
from src.dataset_builder import create_dataset
from models import build_models
from src.train import train_model
from src.evaluate import (
    evaluate_model,
    plot_rolling_predictions_all_features,
    save_last_days_results_to_csv,
    save_initial_sequence_to_csv
)
from sklearn.preprocessing import MinMaxScaler
import torch
import os
import pandas as pd

# ----------------- 全局配置 -----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
FUTURE_DAYS = 5
TARGET_FEATURE_INDEX = 5
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.01

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("src/models/saved_models", exist_ok=True)

    # 1. 数据加载和预处理
    selected_columns = input("请输入要展示的列名（逗号分隔）: ")
    data, dataset, scaler = load_and_clean_data(DATA_PATH, selected_columns)
    input_size = dataset.shape[1]

    # 2. 构建数据集
    seq_len = auto_select_seq_len(input_size)
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[:train_size], dataset[train_size:]
    trainX, trainY = create_dataset(train, seq_len)
    testX, testY = create_dataset(test, seq_len)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trainX, trainY), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(testX, testY), batch_size=BATCH_SIZE, shuffle=False)

    # 3. 模型构建与加载/训练
    models = build_models(input_size)
    trained_models = {}
    all_train_losses = {}

    for name, model in models.items():
        model_path = f"src/models/saved_models/{name}.pth"

        if os.path.exists(model_path):
            print(f"✅ 检测到已保存模型，加载：{model_path}")
            model.load_state_dict(torch.load(model_path, weights_only=True))
            trained_models[name] = model
        else:
            print(f"🚀 未检测到模型，开始训练: {name}")
            trained, losses = train_model(model, name, train_loader, EPOCHS, LEARNING_RATE)
            trained_models[name] = trained
            all_train_losses[name] = losses
            torch.save(trained.state_dict(), model_path)
            print(f"✅ 模型已保存到: {model_path}")

    # 4. 模型评估
    evaluation_results = []
    for name, model in trained_models.items():
        result = evaluate_model(model, name, test_loader, scaler, TARGET_FEATURE_INDEX, dataset.shape[1])
        evaluation_results.append(result)

    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv("outputs/model_evaluation_results.csv", index=False)
    print("📊 模型评估结果已保存：outputs/model_evaluation_results.csv")

    # 5. 滚动预测 + 可视化
    rolling_predictions_all_models = {}
    real_values_all_models = {}
    initial_index_ranges = {}

    for name, model in trained_models.items():
        preds, real_values, index_range = plot_rolling_predictions_all_features(
            model, name, test, scaler, seq_len, FUTURE_DAYS, TARGET_FEATURE_INDEX, dataset.shape[1]
        )
        rolling_predictions_all_models[name] = preds
        real_values_all_models[name] = real_values
        initial_index_ranges[name] = index_range

    save_last_days_results_to_csv(rolling_predictions_all_models, real_values_all_models, initial_index_ranges, FUTURE_DAYS)
    save_initial_sequence_to_csv(test, list(initial_index_ranges.values())[0], seq_len, scaler, dataset.shape[1])
    print("📁 滚动预测结果与初始序列已保存至 outputs 目录")

if __name__ == '__main__':
    main()
