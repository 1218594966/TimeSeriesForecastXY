import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from datetime import datetime, timedelta
from einops import rearrange

# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 官方xLSTM实现相关类
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


# 官方xLSTM模型
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


def create_sequences_with_dates(df, n_past, n_future):
    """
    创建输入序列和目标序列，并保存对应的日期信息
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


def train_model(model, train_loader, optimizer, criterion, epochs, device):
    """
    训练模型
    """
    train_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向传播
            y_pred = model(X_batch)

            # 确保 y_pred 和 y_batch 的形状一致 (batch_size, future_days, features)
            if y_pred.shape != y_batch.shape:
                y_pred = y_pred.transpose(1, 2)

            loss = criterion(y_pred, y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1}/{epochs}, 训练损失: {train_loss:.4f}')

    return {'train_loss': train_losses}


def evaluate_model(y_true, y_pred, feature_names, scaler):
    """
    评估模型性能，计算多个指标
    """
    # 反归一化
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, y_true.shape[-1]))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))

    # 调整形状以便按特征计算指标
    n_samples = y_true.shape[0]
    n_steps = y_true.shape[1]
    n_features = y_true.shape[2]

    y_true_reshaped = y_true_inv.reshape(n_samples * n_steps, n_features)
    y_pred_reshaped = y_pred_inv.reshape(n_samples * n_steps, n_features)

    # 计算整体指标
    results = {
        'overall': {
            'RMSE': np.sqrt(mean_squared_error(y_true_reshaped, y_pred_reshaped)),
            'MAE': mean_absolute_error(y_true_reshaped, y_pred_reshaped),
            'MAPE': mean_absolute_percentage_error(y_true_reshaped, y_pred_reshaped) * 100,
            'R2': r2_score(y_true_reshaped, y_pred_reshaped)
        }
    }

    # 计算每个特征的指标
    for i, feature in enumerate(feature_names):
        y_true_feature = y_true_reshaped[:, i]
        y_pred_feature = y_pred_reshaped[:, i]

        # 跳过可能包含零值的特征的MAPE计算（避免除以零）
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


def save_metrics_to_csv(results, output_path):
    """
    将评估指标保存为CSV文件
    """
    # 准备数据
    metrics_data = []
    for feature, metrics in results.items():
        metrics_data.append({
            'Feature': feature,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2']
        })

    # 创建DataFrame并保存
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_path, index=False)
    print(f"指标已保存到: {output_path}")


def plot_time_series_comparison(true_values, pred_values, dates, feature_names, output_dir, title_prefix=""):
    """
    绘制时间序列预测结果对比图，并将数据保存为CSV，确保每个日期只有一个预测值
    """
    # 确保日期是datetime类型
    if isinstance(dates[0], str):
        dates = [pd.to_datetime(d) for d in dates]

    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))

        # 使用实线绘制，使用不同颜色区分
        plt.plot(dates, true_values[:, i], 'b-', label='实际值', linewidth=1.5)
        plt.plot(dates, pred_values[:, i], 'r-', label='预测值', linewidth=1.5)

        plt.title(f'{title_prefix}{feature} - 预测结果')
        plt.xlabel('日期')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)  # 降低网格透明度，使图形更清晰
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(output_dir, f'{title_prefix}prediction_{feature}.png'), dpi=300)
        plt.close()

        # 将数据保存为CSV，使用日期作为索引以确保没有重复
        plot_df = pd.DataFrame({
            'Actual': true_values[:, i],
            'Predicted': pred_values[:, i]
        }, index=dates)

        # 确保索引排序，便于查看
        plot_df = plot_df.sort_index()

        csv_path = os.path.join(output_dir, f'{title_prefix}prediction_{feature}.csv')
        plot_df.to_csv(csv_path)
        print(f"图表数据已保存到: {csv_path}")


def plot_test_with_confidence(test_dates, test_true, test_pred, last_pred_dates,
                              last_pred_true, last_pred_pred, feature_names,
                              scaler, output_dir, future_days):
    """
    绘制测试集带有95%置信区间的图，区分测试集最后预测天数和前面部分，并将数据保存为CSV
    """
    # 获取测试集误差统计
    errors = np.abs(test_pred - test_true)

    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(14, 6))

        # 测试集前部分
        test_dates_main = test_dates[:-len(last_pred_dates)]
        test_true_main = test_true[:-(len(last_pred_dates)), i]
        test_pred_main = test_pred[:-(len(last_pred_dates)), i]

        # 绘制测试集主要部分，使用细线
        plt.plot(test_dates_main, test_true_main, 'b-', linewidth=1.5, label='测试集实际值')
        plt.plot(test_dates_main, test_pred_main, 'r-', linewidth=1.5, label='测试集预测值')

        # 计算测试集误差统计
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])

        # 绘制测试集最后预测天数，使用不同颜色的实线
        plt.plot(last_pred_dates, last_pred_true[:, i], 'g-', linewidth=1.5, label=f'最后{future_days}天实际值')
        plt.plot(last_pred_dates, last_pred_pred[:, i], 'm-', linewidth=1.5, label=f'最后{future_days}天预测值')

        # 添加置信区间（基于整个测试集误差）
        upper_bound = test_pred_main + 1.96 * std_error
        lower_bound = test_pred_main - 1.96 * std_error
        plt.fill_between(test_dates_main, lower_bound, upper_bound,
                         color='r', alpha=0.1, label='95% 置信区间')

        # 为最后预测天数添加置信区间
        upper_bound_last = last_pred_pred[:, i] + 1.96 * std_error
        lower_bound_last = last_pred_pred[:, i] - 1.96 * std_error
        plt.fill_between(last_pred_dates, lower_bound_last, upper_bound_last,
                         color='m', alpha=0.1)

        # 添加垂直线分隔
        if len(test_dates_main) > 0:
            plt.axvline(x=test_dates_main[-1], color='k', linestyle='--', alpha=0.5)

        plt.title(f'{feature} - 测试集预测对比（带置信区间）')
        plt.xlabel('日期')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(output_dir, f'test_confidence_{feature}.png'), dpi=300)
        plt.close()

        # 将数据保存为CSV
        # 测试集主要部分数据 - 使用日期作为索引以确保没有重复
        main_data = {
            'Actual': test_true_main,
            'Predicted': test_pred_main,
            'LowerBound': lower_bound,
            'UpperBound': upper_bound
        }
        main_df = pd.DataFrame(main_data, index=test_dates_main)
        main_df = main_df.sort_index()  # 确保索引排序

        # 最后预测天数数据 - 使用日期作为索引以确保没有重复
        last_pred_data = {
            'Actual': last_pred_true[:, i],
            'Predicted': last_pred_pred[:, i],
            'LowerBound': lower_bound_last,
            'UpperBound': upper_bound_last
        }
        last_pred_df = pd.DataFrame(last_pred_data, index=last_pred_dates)
        last_pred_df = last_pred_df.sort_index()  # 确保索引排序

        # 保存测试集主要部分数据
        main_csv_path = os.path.join(output_dir, f'test_confidence_main_{feature}.csv')
        main_df.to_csv(main_csv_path)

        # 保存最后预测天数数据
        last_pred_csv_path = os.path.join(output_dir, f'test_confidence_last_{future_days}_days_{feature}.csv')
        last_pred_df.to_csv(last_pred_csv_path)

        print(f"置信区间图表数据已保存到: {main_csv_path} 和 {last_pred_csv_path}")


def save_training_history(history, output_dir):
    """
    保存训练历史数据为CSV
    """
    history_df = pd.DataFrame({
        'Epoch': range(1, len(history['train_loss']) + 1),
        'TrainLoss': history['train_loss']
    })

    csv_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(csv_path, index=False)
    print(f"训练历史数据已保存到: {csv_path}")


def main(args):
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # 加载数据
    print(f"正在加载CSV文件: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    # 处理日期列（如果存在）
    date_col = None
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

    # 移除非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < len(df.columns):
        print(f"移除了非数值列: {set(df.columns) - set(numeric_cols)}")
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

    # 设置预测的参数
    n_past = args.past_days
    n_future = args.future_days

    # 创建序列
    X, y, X_dates, y_dates = create_sequences_with_dates(df_scaled, n_past, n_future)
    print(f"序列形状: X: {X.shape}, y: {y.shape}")

    # 按8:2拆分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train_dates, X_test_dates = X_dates[:train_size], X_dates[train_size:]
    y_train_dates, y_test_dates = y_dates[:train_size], y_dates[train_size:]

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 初始化模型
    input_dim = X_train.shape[2]  # 特征数量
    output_dim = y_train.shape[2]  # 输出特征数量

    # 使用官方XLSTM模型
    model = XLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        context_points=n_past,
        target_points=n_future,
        num_blocks=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # 打印模型结构
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    print("\n开始训练模型...")
    history = train_model(
        model, train_loader, optimizer, criterion, args.epochs, device
    )

    # 保存模型
    model_path = os.path.join(output_dir, 'xlstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': output_dim,
            'context_points': n_past,
            'target_points': n_future,
            'num_layers': args.num_layers
        }
    }, model_path)
    print(f"模型已保存到: {model_path}")

    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失', linewidth=1.5)
    plt.title('模型训练历史')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

    # 保存训练历史数据为CSV
    save_training_history(history, output_dir)

    # ==================== 评估模型 ====================
    print("\n评估模型...")
    model.eval()

    # 获取测试集预测结果
    y_test_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            # 确保 y_pred 形状正确 (batch_size, future_days, features)
            if y_pred.shape[1] != n_future:
                y_pred = y_pred.transpose(1, 2)

            y_test_pred.append(y_pred.cpu().numpy())

    y_test_pred = np.vstack(y_test_pred)

    # 评估测试集
    test_results, test_true_inv, test_pred_inv = evaluate_model(
        y_test.numpy(), y_test_pred, feature_names, scaler
    )

    # 输出测试集评估结果
    print("\n测试集评估结果:")
    print(f"整体 RMSE: {test_results['overall']['RMSE']:.4f}")
    print(f"整体 MAE: {test_results['overall']['MAE']:.4f}")
    print(f"整体 MAPE: {test_results['overall']['MAPE']:.2f}%")
    print(f"整体 R2: {test_results['overall']['R2']:.4f}")

    # 保存测试集评估指标为CSV
    test_metrics_path = os.path.join(output_dir, 'test_metrics.csv')
    save_metrics_to_csv(test_results, test_metrics_path)

    # ==================== 绘制测试集的预测结果 ====================
    # 修复：使用字典确保每个日期只有一个预测值
    print("\n绘制测试集预测结果...")

    # 创建用于存储预测结果的字典，以日期为键
    test_pred_by_date = {}
    test_true_by_date = {}

    # 收集数据并按日期进行整理，确保每个日期只有一个预测值
    for i in range(len(y_test)):
        sample_dates = y_test_dates[i]
        true_sample = y_test[i].numpy()
        pred_sample = y_test_pred[i]

        for j in range(len(sample_dates)):
            date = sample_dates[j]
            # 只保存每个日期的第一个预测值（或者可以选择取平均值）
            if date not in test_pred_by_date:
                test_pred_by_date[date] = pred_sample[j]
                test_true_by_date[date] = true_sample[j]

    # 将字典转换为列表用于绘图和后续处理
    test_dates = list(test_pred_by_date.keys())
    test_true_values = np.array(list(test_true_by_date.values()))
    test_pred_values = np.array(list(test_pred_by_date.values()))

    # 按日期排序
    sorted_indices = np.argsort(test_dates)
    sorted_test_dates = [test_dates[i] for i in sorted_indices]
    sorted_test_true = test_true_values[sorted_indices]
    sorted_test_pred = test_pred_values[sorted_indices]

    # 反归一化
    sorted_test_true_inv = scaler.inverse_transform(sorted_test_true)
    sorted_test_pred_inv = scaler.inverse_transform(sorted_test_pred)

    # 绘制测试集结果
    plot_time_series_comparison(
        sorted_test_true_inv, sorted_test_pred_inv, sorted_test_dates,
        feature_names, output_dir, title_prefix="测试集_"
    )

    # ==================== 测试集的最后预测天数 ====================
    print(f"\n评估测试集的最后{n_future}天...")

    # 获取测试集最后一个样本
    last_test_sample = X_test[-1:].to(device)
    last_test_true = y_test[-1].numpy()
    last_test_dates = y_test_dates[-1]

    # 进行预测
    with torch.no_grad():
        last_test_pred = model(last_test_sample)
        # 确保形状正确
        if last_test_pred.shape[1] != n_future:
            last_test_pred = last_test_pred.transpose(1, 2)
        last_test_pred = last_test_pred.cpu().numpy()[0]

    # 反归一化
    last_test_true_inv = scaler.inverse_transform(last_test_true)
    last_test_pred_inv = scaler.inverse_transform(last_test_pred)

    # 保存为CSV，使用日期为索引
    last_test_df = pd.DataFrame(
        np.column_stack([last_test_true_inv, last_test_pred_inv]),
        index=last_test_dates,
        columns=[f"{col}_真实值" for col in feature_names] + [f"{col}_预测值" for col in feature_names]
    )
    last_test_df.to_csv(os.path.join(output_dir, f'last_{n_future}_days_prediction.csv'))

    # 绘制最后预测天数对比图
    plot_time_series_comparison(
        last_test_true_inv, last_test_pred_inv, last_test_dates,
        feature_names, output_dir, title_prefix=f"最后{n_future}天_"
    )

    # ==================== 带有置信区间的测试集图表 ====================
    print("\n绘制带置信区间的测试集图表...")

    # 绘制测试集带有置信区间的图
    plot_test_with_confidence(
        sorted_test_dates, sorted_test_true_inv, sorted_test_pred_inv,
        last_test_dates, last_test_true_inv, last_test_pred_inv,
        feature_names, scaler, output_dir, n_future
    )

    print(f"\n所有输出已保存到目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多变量多步XLSTM时间序列预测')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--past_days', type=int, default=30, help='用于预测的过去天数')
    parser.add_argument('--future_days', type=int, default=15, help='需要预测的未来天数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='XLSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='XLSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')

    args = parser.parse_args()
    main(args)






