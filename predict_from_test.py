import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 定义必要的模型组件类
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# xLSTM Block Stack configuration classes
class mLSTMBlockConfig:
    def __init__(self):
        self.default_config = True


class sLSTMBlockConfig:
    def __init__(self):
        self.default_config = True


class xLSTMBlockStackConfig:
    def __init__(self, mlstm_block, slstm_block, num_blocks=3, embedding_dim=256,
                 add_post_blocks_norm=True, block_map=1, context_length=336):
        self.mlstm_block = mlstm_block
        self.slstm_block = slstm_block
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim
        self.add_post_blocks_norm = add_post_blocks_norm
        self.block_map = block_map
        self.context_length = context_length


# 简化版 xLSTM Block Stack 实现
class xLSTMBlockStack(nn.Module):
    def __init__(self, config):
        super(xLSTMBlockStack, self).__init__()
        self.config = config

        # 创建 LSTM 层
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=config.embedding_dim if i > 0 else config.embedding_dim,
                hidden_size=config.embedding_dim,
                batch_first=True
            ) for i in range(config.num_blocks)
        ])

        # 创建 Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embedding_dim) for _ in range(config.num_blocks)
        ])

        # 创建门控机制
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.embedding_dim, config.embedding_dim),
                nn.Sigmoid()
            ) for _ in range(config.num_blocks)
        ])

        # 后处理规范化
        if config.add_post_blocks_norm:
            self.post_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.post_norm = nn.Identity()

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        batch_size, seq_len, _ = x.shape

        for i in range(self.config.num_blocks):
            # LSTM处理
            lstm_out, _ = self.lstm_layers[i](x)

            # 应用门控机制
            gate = self.gates[i](lstm_out)
            gated_output = gate * lstm_out

            # 残差连接和规范化
            x = x + gated_output
            x = self.layer_norms[i](x)

        # 后处理规范化
        x = self.post_norm(x)
        return x


# XLSTM模型
class XLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_points, target_points, num_blocks=3, dropout=0.1):
        super(XLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_points = context_points
        self.target_points = target_points

        # 配置xLSTM
        mlstm_config = mLSTMBlockConfig()
        slstm_config = sLSTMBlockConfig()

        config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            num_blocks=num_blocks,
            embedding_dim=hidden_dim,
            add_post_blocks_norm=True,
            block_map=1,
            context_length=context_points
        )

        # 批归一化
        self.batch_norm = nn.BatchNorm1d(input_dim)

        # 分解层
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        # 季节性和趋势线性层
        self.Linear_Seasonal = nn.Linear(context_points, target_points)
        self.Linear_Trend = nn.Linear(context_points, target_points)

        # 初始化权重
        self.Linear_Seasonal.weight = nn.Parameter((1 / context_points) * torch.ones([target_points, context_points]))
        self.Linear_Trend.weight = nn.Parameter((1 / context_points) * torch.ones([target_points, context_points]))

        # 投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # xLSTM堆栈
        self.xlstm_stack = xLSTMBlockStack(config)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # 时间序列分解
        seasonal_init, trend_init = self.decomposition(x)

        # 转换维度以适应线性层
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        # 应用季节性和趋势线性层
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # 组合季节性和趋势
        combined = seasonal_output + trend_output  # [batch_size, input_dim, target_points]

        # 投影到隐藏维度
        projected = self.input_projection(combined.permute(0, 2, 1))  # [batch_size, target_points, hidden_dim]

        # 应用xLSTM处理
        xlstm_out = self.xlstm_stack(projected)  # [batch_size, target_points, hidden_dim]

        # Dropout
        xlstm_out = self.dropout(xlstm_out)

        # 投影回输出维度
        output = self.output_projection(xlstm_out)  # [batch_size, target_points, output_dim]

        return output  # 返回形状为 [batch_size, target_points, output_dim]


def load_model(model_path, device):
    """
    加载保存的模型
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = XLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        context_points=config['context_points'],
        target_points=config['target_points'],
        num_blocks=config['num_layers'],
        dropout=0.1  # 预测时可以使用较小的dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式

    print(f"模型已加载，输入维度: {config['input_dim']}, 输出维度: {config['output_dim']}")
    print(f"上下文点数(历史天数): {config['context_points']}, 目标点数(预测天数): {config['target_points']}")

    return model, config


def preprocess_data(df, date_col=None):
    """
    预处理数据，返回归一化后的数据和日期索引
    """
    # 处理日期列
    # 处理日期列（如果存在）
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break

    if date_col:
        print(f"检测到日期列: {date_col}")
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        # 如果没有日期列，创建一个假设的日期索引（从当前日期向前推）
        print("未检测到日期列，创建默认日期索引")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=len(df) - 1)
        date_range = pd.date_range(start=start_date, end=end_date, periods=len(df))
        df.index = date_range

    # 确保索引是按照日期排序的
    df = df.sort_index()

    # 移除非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < len(df.columns):
        non_numeric = set(df.columns) - set(numeric_cols)
        print(f"移除了非数值列: {non_numeric}")
        df = df[numeric_cols]

    # 检查并处理缺失值
    if df.isnull().sum().sum() > 0:
        print("检测到缺失值，使用前向填充方法")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)  # 处理开头的缺失值

    # 保存原始特征名称
    feature_names = df.columns.tolist()

    # 创建归一化器并转换数据
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # 将归一化后的数据放回DataFrame以保持索引
    df_scaled = pd.DataFrame(data_scaled, index=df.index, columns=df.columns)

    return df_scaled, scaler, feature_names


def predict_future_from_test_set(model, test_df, test_size_percent, last_n_days, future_days, device, scaler):
    """
    从测试集的最后n天预测未来L天

    参数:
    - model: 训练好的模型
    - test_df: 完整的数据集，DataFrame
    - test_size_percent: 测试集占总数据的百分比
    - last_n_days: 使用测试集的最后几天进行预测
    - future_days: 预测未来的天数
    - device: 计算设备
    - scaler: 用于反归一化的scaler对象

    返回:
    - 预测结果
    - 最后n天的实际数据
    - 最后n天的日期
    - 预测的未来日期
    """
    # 计算测试集的大小
    total_size = len(test_df)
    train_size = int(total_size * (1 - test_size_percent / 100))

    # 分割测试集
    test_data = test_df.iloc[train_size:]
    print(f"测试集大小: {len(test_data)} 天")

    # 确保last_n_days不超过测试集大小
    if last_n_days > len(test_data):
        print(f"警告: 请求的最后几天({last_n_days})超过测试集大小({len(test_data)})。将使用整个测试集。")
        last_n_days = len(test_data)

    # 获取测试集的最后n天
    last_n_days_data = test_data.iloc[-last_n_days:]
    last_n_days_dates = last_n_days_data.index
    print(f"使用测试集的最后 {last_n_days} 天进行预测，从 {last_n_days_dates[0]} 到 {last_n_days_dates[-1]}")

    # 转换为模型输入格式
    input_tensor = torch.FloatTensor(last_n_days_data.values).unsqueeze(0).to(device)  # [1, n_days, features]

    # 预测
    with torch.no_grad():
        predictions = model(input_tensor)

        # 确保预测结果形状正确 [1, future_days, features]
        if predictions.shape[1] != future_days:
            predictions = predictions.transpose(1, 2)

        predictions = predictions.cpu().numpy()[0]  # [future_days, features]

    # 反归一化预测结果
    predictions_inv = scaler.inverse_transform(predictions)

    # 生成未来日期
    last_date = last_n_days_dates[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days, freq='D')

    # 获取最后n天的原始数据（用于对比）
    last_n_days_orig = scaler.inverse_transform(last_n_days_data.values)

    return predictions_inv, last_n_days_orig, last_n_days_dates, future_dates


def main(args):
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_prediction_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    model, config = load_model(args.model_path, device)

    # 模型的历史天数和预测天数
    model_past_days = config['context_points']
    model_future_days = config['target_points']

    # 加载数据
    print(f"正在加载CSV文件: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    # 预处理数据
    df_scaled, scaler, feature_names = preprocess_data(df, args.date_column)

    # 确定预测的天数
    future_days = args.future_days if args.future_days else model_future_days

    # 确保last_n_days至少等于模型期望的输入天数
    last_n_days = args.last_n_days if args.last_n_days else model_past_days
    if last_n_days < model_past_days:
        print(
            f"警告: 请求的最后几天({last_n_days})小于模型训练的历史天数({model_past_days})。将使用 {model_past_days} 天。")
        last_n_days = model_past_days

    # 从测试集的最后n天预测未来L天
    predictions, last_n_days_data, last_n_days_dates, future_dates = predict_future_from_test_set(
        model, df_scaled, args.test_size, last_n_days, future_days, device, scaler)

    # 创建预测结果DataFrame
    predictions_df = pd.DataFrame(
        predictions,
        index=future_dates,
        columns=feature_names
    )

    # 保存预测结果
    csv_path = os.path.join(output_dir, 'test_set_future_predictions.csv')
    predictions_df.to_csv(csv_path)
    print(f"预测结果已保存到: {csv_path}")

    # 绘制预测图表
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))

        # 绘制最后n天的历史数据
        plt.plot(last_n_days_dates, last_n_days_data[:, i], 'b-', label=f'最后{last_n_days}天历史数据', linewidth=1.5)

        # 绘制预测数据
        plt.plot(future_dates, predictions[:, i], 'r-', label=f'未来{future_days}天预测', linewidth=1.5)

        # 添加垂直线分隔历史和预测
        plt.axvline(x=last_n_days_dates[-1], color='k', linestyle='--', alpha=0.5)

        plt.title(f'{feature} - 基于测试集最后{last_n_days}天预测未来{future_days}天')
        plt.xlabel('日期')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图表
        plt.savefig(os.path.join(output_dir, f'test_prediction_{feature}.png'), dpi=300)
        plt.close()

    # 保存历史数据和预测数据的合并图表数据
    # 最后n天的历史数据
    history_df = pd.DataFrame(
        last_n_days_data,
        index=last_n_days_dates,
        columns=feature_names
    )

    # 合并历史和预测
    combined_data = pd.concat([history_df, predictions_df])
    combined_csv_path = os.path.join(output_dir, 'history_and_predictions.csv')
    combined_data.to_csv(combined_csv_path)
    print(f"历史数据和预测数据已合并保存到: {combined_csv_path}")

    print(f"\n所有输出已保存到目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用测试集的最后n天数据预测未来L天')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型文件路径')
    parser.add_argument('--date_column', type=str, help='日期列名称，如果CSV中有日期列')
    parser.add_argument('--test_size', type=float, default=20, help='测试集占总数据的百分比，默认20%')
    parser.add_argument('--last_n_days', type=int, help='使用测试集最后几天的数据，如不指定则使用模型训练的历史天数')
    parser.add_argument('--future_days', type=int, help='需要预测的未来天数，如不指定则使用模型的默认预测天数')

    args = parser.parse_args()
    main(args)