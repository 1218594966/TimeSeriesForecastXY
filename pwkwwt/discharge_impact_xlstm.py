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


# XLSTM模型相关组件
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


# xLSTM Block Stack配置类
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


# xLSTM Block Stack 实现
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


# 单一排口下游多监测点的XLSTM预测模型
class DischargeImpactXLSTM(nn.Module):
    def __init__(self, discharge_features, upstream_features, env_features,
                 downstream_points, water_parameters, context_points, target_points,
                 hidden_dim=128, num_blocks=3, dropout=0.1):
        """
        Args:
            discharge_features: 排口排放参数数量
            upstream_features: 上游水质参数数量
            env_features: 环境因素参数数量
            downstream_points: 下游监测点数量
            water_parameters: 每个监测点的水质参数数量
            context_points: 输入序列长度（时间步）
            target_points: 预测序列长度（时间步）
            hidden_dim: 隐藏层维度
            num_blocks: XLSTM块数量
            dropout: Dropout比率
        """
        super(DischargeImpactXLSTM, self).__init__()
        self.discharge_features = discharge_features
        self.upstream_features = upstream_features
        self.env_features = env_features
        self.downstream_points = downstream_points
        self.water_parameters = water_parameters
        self.context_points = context_points
        self.target_points = target_points

        # 总输入特征数
        self.input_dim = discharge_features + upstream_features + env_features
        # 总输出特征数 (下游点数 * 每个点的水质参数数)
        self.output_dim = downstream_points * water_parameters

        # 批归一化
        self.batch_norm = nn.BatchNorm1d(self.input_dim)

        # 时间序列分解层
        kernel_size = 25  # 可调整的超参数
        self.decomposition = series_decomp(kernel_size)

        # 季节性和趋势线性层
        self.Linear_Seasonal = nn.Linear(context_points, target_points)
        self.Linear_Trend = nn.Linear(context_points, target_points)

        # 初始化权重
        self.Linear_Seasonal.weight = nn.Parameter((1 / context_points) * torch.ones([target_points, context_points]))
        self.Linear_Trend.weight = nn.Parameter((1 / context_points) * torch.ones([target_points, context_points]))

        # 投影层
        self.input_projection = nn.Linear(self.input_dim, hidden_dim)

        # 构建xLSTM堆栈配置
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

        # xLSTM堆栈
        self.xlstm_stack = xLSTMBlockStack(config)

        # 空间注意力机制 - 考虑不同监测点的相对重要性
        self.spatial_attention = nn.Sequential(
            nn.Linear(hidden_dim, downstream_points),
            nn.Softmax(dim=-1)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 下游多点预测层
        # 每个下游点都有一个输出投影层
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, water_parameters)
            for _ in range(downstream_points)
        ])

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_dim]
        返回: [batch_size, target_points, downstream_points, water_parameters]
        """
        batch_size, seq_len, _ = x.shape

        # 时间序列分解
        seasonal_init, trend_init = self.decomposition(x)

        # 转换维度以适应线性层
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        trend_init = trend_init.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]

        # 应用季节性和趋势线性层
        seasonal_output = self.Linear_Seasonal(seasonal_init)  # [batch_size, input_dim, target_points]
        trend_output = self.Linear_Trend(trend_init)  # [batch_size, input_dim, target_points]

        # 组合季节性和趋势
        x_time = seasonal_output + trend_output  # [batch_size, input_dim, target_points]
        x_time = x_time.permute(0, 2, 1)  # [batch_size, target_points, input_dim]

        # 投影到隐藏维度
        x_projected = self.input_projection(x_time)  # [batch_size, target_points, hidden_dim]

        # 应用xLSTM处理
        xlstm_out = self.xlstm_stack(x_projected)  # [batch_size, target_points, hidden_dim]

        # Dropout
        xlstm_out = self.dropout(xlstm_out)

        # 计算空间注意力权重
        spatial_weights = self.spatial_attention(xlstm_out)  # [batch_size, target_points, downstream_points]

        # 为每个下游监测点生成预测
        all_outputs = []
        for i in range(self.downstream_points):
            # 应用每个下游点的输出投影
            point_output = self.output_projections[i](xlstm_out)  # [batch_size, target_points, water_parameters]

            # 应用空间注意力权重
            point_weight = spatial_weights[:, :, i].unsqueeze(-1)  # [batch_size, target_points, 1]
            weighted_output = point_output * point_weight  # [batch_size, target_points, water_parameters]

            all_outputs.append(weighted_output)

        # 堆叠所有下游点的输出
        stacked_output = torch.stack(all_outputs,
                                     dim=2)  # [batch_size, target_points, downstream_points, water_parameters]

        return stacked_output


# 数据预处理和序列创建类
class DischargeDataProcessor:
    def __init__(self, past_days=30, future_days=15, test_size=0.2):
        """
        Args:
            past_days: 用于预测的过去天数
            future_days: 需要预测的未来天数
            test_size: 测试集比例
        """
        self.past_days = past_days
        self.future_days = future_days
        self.test_size = test_size
        self.scalers = {}
        self.feature_groups = {}
        self.feature_map = {}  # 添加特征映射字典，用于跟踪实际列名和模型中使用的列名之间的映射

    def load_data(self, discharge_file, downstream_file, upstream_file=None, env_file=None):
        """
        加载各类数据文件

        Args:
            discharge_file: 排口排放数据文件路径
            downstream_file: 下游监测点水质数据文件路径
            upstream_file: 上游水质数据文件路径（可选）
            env_file: 环境因素数据文件路径（可选）

        返回:
            合并后的DataFrame
        """
        print(f"加载排口数据: {discharge_file}")
        discharge_df = pd.read_csv(discharge_file)

        print(f"加载下游监测点数据: {downstream_file}")
        downstream_df = pd.read_csv(downstream_file)

        # 确保数据有日期列
        date_col = self._find_date_column(discharge_df)
        if not date_col:
            raise ValueError("未找到日期列，请确保数据包含日期信息")

        # 转换日期列
        discharge_df[date_col] = pd.to_datetime(discharge_df[date_col])
        downstream_df[date_col] = pd.to_datetime(downstream_df[date_col])

        # 记录特征组的列名
        self.feature_groups['discharge'] = [col for col in discharge_df.columns if col != date_col]

        # 获取下游监测点ID
        point_id_col = self._find_point_id_column(downstream_df)
        if not point_id_col:
            raise ValueError("下游监测点数据中未找到监测点ID列")

        # 记录下游监测点数及其水质参数
        downstream_points = downstream_df[point_id_col].unique()
        self.downstream_points = downstream_points
        water_params = [col for col in downstream_df.columns
                        if col != date_col and col != point_id_col]
        self.feature_groups['downstream'] = water_params

        print(f"检测到 {len(downstream_points)} 个下游监测点")
        print(f"每个监测点有 {len(water_params)} 个水质参数")

        # 处理上游数据（如果有）
        if upstream_file:
            print(f"加载上游数据: {upstream_file}")
            upstream_df = pd.read_csv(upstream_file)
            upstream_df[date_col] = pd.to_datetime(upstream_df[date_col])
            # 重命名上游数据列，避免与其他DataFrame中的列名冲突
            upstream_cols = [col for col in upstream_df.columns if col != date_col]
            rename_dict = {col: f"Upstream_{col}" for col in upstream_cols}
            upstream_df = upstream_df.rename(columns=rename_dict)
            self.feature_groups['upstream'] = [f"Upstream_{col}" for col in upstream_cols]
        else:
            upstream_df = None
            self.feature_groups['upstream'] = []

        # 处理环境数据（如果有）
        if env_file:
            print(f"加载环境数据: {env_file}")
            env_df = pd.read_csv(env_file)
            env_df[date_col] = pd.to_datetime(env_df[date_col])
            # 环境数据通常有独特的列名(Rainfall, Humidity等)，不太可能冲突，但为安全起见也进行重命名
            env_cols = [col for col in env_df.columns if col != date_col]
            rename_dict = {col: f"Env_{col}" for col in env_cols if
                           col not in ['Rainfall', 'WindSpeed', 'Humidity', 'SolarRadiation']}
            env_df = env_df.rename(columns=rename_dict)
            self.feature_groups['env'] = [
                f"Env_{col}" if col not in ['Rainfall', 'WindSpeed', 'Humidity', 'SolarRadiation']
                else col for col in env_cols]
        else:
            env_df = None
            self.feature_groups['env'] = []

        # 合并数据
        merged_data = self._merge_all_data(
            discharge_df, downstream_df, upstream_df, env_df,
            date_col, point_id_col
        )

        return merged_data, date_col, point_id_col

    def _merge_all_data(self, discharge_df, downstream_df, upstream_df, env_df, date_col, point_id_col):
        """
        合并所有数据源

        返回:
            重塑后的DataFrame，每行是一个时间点，包含排口数据、下游各监测点数据、上游数据和环境数据
        """
        # 处理下游监测点数据 - 将不同监测点的数据转为宽表格式
        downstream_wide = downstream_df.pivot(
            index=date_col,
            columns=point_id_col,
            values=self.feature_groups['downstream']
        )

        # 创建特征映射字典来跟踪实际列名和模型中使用的列名
        self.feature_map = {}

        # 列名格式会是(param, point)形式，例如('COD', 'DS1')
        # 我们需要创建从模型列名(如'DS1_COD')到实际列名(如'COD_DS1')的映射
        model_columns = []
        for param, point in downstream_wide.columns:
            # 实际的列名格式是 param_point (例如 'COD_DS1')
            actual_col = f"{param}_{point}"
            # 模型期望的列名格式是 point_param (例如 'DS1_COD')
            model_col = f"{point}_{param}"
            self.feature_map[model_col] = actual_col
            model_columns.append(model_col)

        # 重命名列，采用实际数据的列名格式 param_point (例如 'COD_DS1')
        downstream_wide.columns = [f"{param}_{point}" for param, point in downstream_wide.columns]

        # 将排口数据设为索引
        discharge_df = discharge_df.set_index(date_col)

        # 合并排口和下游数据
        merged_data = pd.merge(
            discharge_df,
            downstream_wide,
            left_index=True,
            right_index=True,
            how='inner'
        )

        # 合并上游数据（如果有）
        if upstream_df is not None:
            upstream_df = upstream_df.set_index(date_col)
            merged_data = pd.merge(
                merged_data,
                upstream_df,
                left_index=True,
                right_index=True,
                how='inner'
            )

        # 合并环境数据（如果有）
        if env_df is not None:
            env_df = env_df.set_index(date_col)
            merged_data = pd.merge(
                merged_data,
                env_df,
                left_index=True,
                right_index=True,
                how='inner'
            )

        # 处理缺失值
        if merged_data.isnull().sum().sum() > 0:
            print("检测到缺失值，使用前向填充方法")
            merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

        return merged_data

    def _find_date_column(self, df):
        """查找日期列"""
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or '日期' in col or '时间' in col:
                return col
        return None

    def _find_point_id_column(self, df):
        """查找监测点ID列"""
        for col in df.columns:
            if 'id' in col.lower() or 'point' in col.lower() or '点位' in col or '站点' in col or '监测点' in col:
                return col
        return None

    def preprocess_data(self, merged_data):
        """
        预处理数据：归一化、创建序列等

        Args:
            merged_data: 合并后的DataFrame

        返回:
            训练和测试数据加载器
        """
        # 提取特征组 - 使用实际列名
        discharge_cols = [col for col in merged_data.columns if col in self.feature_groups['discharge']]

        # 构建下游监测点的输出特征列表 - 使用模型期望的列名格式
        model_output_features = []
        for point in self.downstream_points:
            for param in self.feature_groups['downstream']:
                model_output_features.append(f"{point}_{param}")

        # 将模型列名映射到实际列名
        output_features = [self.feature_map[col] if col in self.feature_map else col for col in model_output_features]

        # 检查是否所有输出特征都存在
        missing_columns = [col for col in output_features if col not in merged_data.columns]
        if missing_columns:
            print(f"警告: 以下列不存在于数据中: {missing_columns}")
            # 打印一些调试信息
            print("实际数据列: ", merged_data.columns.tolist())
            print("尝试读取的列: ", output_features)
            raise ValueError(f"数据中缺少列: {missing_columns}")

        upstream_cols = [col for col in merged_data.columns if col.startswith('Upstream_')]
        env_cols = [col for col in merged_data.columns if col.startswith('Env_') or
                    col in ['Rainfall', 'WindSpeed', 'Humidity', 'SolarRadiation']]

        # 整合输入特征
        input_features = discharge_cols + upstream_cols + env_cols
        print(f"输入特征 ({len(input_features)}): {input_features}")
        print(f"输出特征 ({len(output_features)}): {output_features}")

        # 调试信息
        print("可用的列: ", merged_data.columns.tolist())

        # 检查输入特征是否存在
        missing_input_columns = [col for col in input_features if col not in merged_data.columns]
        if missing_input_columns:
            print(f"警告: 以下输入列不存在: {missing_input_columns}")
            raise ValueError(f"数据中缺少输入列: {missing_input_columns}")

        # 创建归一化器并转换数据
        input_data = merged_data[input_features].values
        output_data = merged_data[output_features].values

        # 使用MinMaxScaler进行归一化
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

        input_scaled = input_scaler.fit_transform(input_data)
        output_scaled = output_scaler.fit_transform(output_data)

        self.scalers['input'] = input_scaler
        self.scalers['output'] = output_scaler

        # 创建时间序列
        X, y = self._create_sequences(input_scaled, output_scaled)
        print(f"序列形状: X: {X.shape}, y: {y.shape}")

        # 拆分训练集和测试集
        train_size = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'input_features': input_features,
            'output_features': output_features,
            'model_output_features': model_output_features,  # 保存模型期望的输出特征列名
            'input_size': len(input_features),
            'output_size': len(output_features)
        }

    def _create_sequences(self, input_data, output_data):
        """
        创建输入序列和目标序列

        Args:
            input_data: 归一化后的输入数据
            output_data: 归一化后的输出数据

        返回:
            X, y: 输入序列和目标序列
        """
        X, y = [], []

        for i in range(len(input_data) - self.past_days - self.future_days + 1):
            # 输入序列
            X.append(input_data[i:i + self.past_days])
            # 目标序列
            y.append(output_data[i + self.past_days:i + self.past_days + self.future_days])

        return np.array(X), np.array(y)

    def reshape_output_for_model(self, y_data):
        """
        重塑输出数据，转换为模型所需的形状
        [batch_size, future_days, total_features] -> [batch_size, future_days, num_points, features_per_point]
        """
        batch_size, future_days, total_features = y_data.shape
        num_points = len(self.downstream_points)
        features_per_point = len(self.feature_groups['downstream'])

        # 确保可以重塑
        assert total_features == num_points * features_per_point, "输出特征数量与监测点和参数数量不匹配"

        # 重塑
        reshaped = y_data.reshape(batch_size, future_days, num_points, features_per_point)
        return reshaped


# 训练和评估类
class DischargeImpactModel:
    def __init__(self, config):
        """
        初始化模型

        Args:
            config: 配置参数字典
        """
        self.config = config
        self.model = None
        self.data_processor = None
        self.trained = False

    def setup(self):
        """设置数据处理器和模型"""
        # 创建数据处理器
        self.data_processor = DischargeDataProcessor(
            past_days=self.config['past_days'],
            future_days=self.config['future_days'],
            test_size=self.config['test_size']
        )

    def load_data(self, discharge_file, downstream_file, upstream_file=None, env_file=None):
        """加载和处理数据"""
        # 加载数据
        merged_data, date_col, point_id_col = self.data_processor.load_data(
            discharge_file, downstream_file, upstream_file, env_file
        )

        # 预处理数据
        self.data = self.data_processor.preprocess_data(merged_data)

        # 保存原始合并数据
        self.merged_data = merged_data
        self.date_col = date_col
        self.point_id_col = point_id_col

        return self.data

    def build_model(self):
        """构建XLSTM模型"""
        # 获取特征维度
        discharge_dim = len([col for col in self.data['input_features'] if not (col.startswith('Upstream_') or
                                                                                col.startswith('Env_') or
                                                                                col in ['Rainfall', 'WindSpeed',
                                                                                        'Humidity', 'SolarRadiation'])])
        upstream_dim = len([col for col in self.data['input_features'] if col.startswith('Upstream_')])
        env_dim = len([col for col in self.data['input_features'] if col.startswith('Env_') or
                       col in ['Rainfall', 'WindSpeed', 'Humidity', 'SolarRadiation']])

        downstream_points = len(self.data_processor.downstream_points)
        water_parameters = len(self.data_processor.feature_groups['downstream'])

        # 创建模型
        self.model = DischargeImpactXLSTM(
            discharge_features=discharge_dim,
            upstream_features=upstream_dim,
            env_features=env_dim,
            downstream_points=downstream_points,
            water_parameters=water_parameters,
            context_points=self.config['past_days'],
            target_points=self.config['future_days'],
            hidden_dim=self.config['hidden_dim'],
            num_blocks=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(device)

        # 打印模型结构
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型总参数量: {total_params}")

        return self.model

    def train(self, epochs=100, batch_size=32, learning_rate=0.001, output_dir=None):
        """训练模型"""
        # 如果未提供输出目录，创建一个
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"discharge_impact_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # 保存配置
        self._save_config()

        # 创建数据加载器
        train_loader = DataLoader(
            self.data['train_dataset'],
            batch_size=batch_size,
            shuffle=True
        )

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练模型
        print("\n开始训练模型...")
        train_losses = []

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # 重塑输出数据
                y_batch_reshaped = self.data_processor.reshape_output_for_model(y_batch)

                # 前向传播
                y_pred = self.model(X_batch)

                # 计算损失
                loss = criterion(y_pred, y_batch_reshaped)

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

        # 保存模型
        model_path = os.path.join(output_dir, 'discharge_impact_xlstm.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f"模型已保存到: {model_path}")

        # 绘制训练历史
        self._plot_training_history(train_losses, epochs)

        # 标记模型已训练
        self.trained = True

        return {'train_loss': train_losses}

    def evaluate(self):
        """评估模型"""
        if not self.trained:
            print("模型尚未训练，请先训练模型")
            return

        print("\n评估模型...")
        self.model.eval()

        # 创建测试数据加载器
        test_loader = DataLoader(
            self.data['test_dataset'],
            batch_size=self.config['batch_size']
        )

        # 获取原始特征名称
        downstream_points = self.data_processor.downstream_points
        water_parameters = self.data_processor.feature_groups['downstream']

        # 用于存储预测结果
        y_test_pred = []
        y_test_true = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)

                # 重塑输出数据
                y_batch_reshaped = self.data_processor.reshape_output_for_model(y_batch)

                # 预测
                y_pred = self.model(X_batch)

                # 收集预测结果和真实值
                y_test_pred.append(y_pred.cpu().numpy())
                y_test_true.append(y_batch_reshaped.numpy())

        # 拼接结果
        y_test_pred = np.vstack(y_test_pred)
        y_test_true = np.vstack(y_test_true)

        # 计算每个监测点每个参数的评估指标
        metrics = {}

        # 获取测试数据的形状
        batch_size, future_days, num_points, num_params = y_test_pred.shape

        # 计算整体指标
        # 修改：将 4D 张量重新整形为原始输出格式，以便正确反归一化
        # 原始输出格式是 (样本, 时间步, 特征总数)，其中特征总数 = 监测点数 * 每个点的参数数

        # 首先将 4D 预测结果重新整形为 3D: (batch_size, future_days, num_points*num_params)
        y_pred_flat = y_test_pred.reshape(batch_size, future_days, num_points * num_params)
        y_true_flat = y_test_true.reshape(batch_size, future_days, num_points * num_params)

        # 然后将其转换为 2D 以便反归一化: (batch_size*future_days, num_points*num_params)
        y_pred_2d = y_pred_flat.reshape(-1, num_points * num_params)
        y_true_2d = y_true_flat.reshape(-1, num_points * num_params)

        # 确保维度匹配原始输出特征数量
        assert y_pred_2d.shape[1] == len(self.data['output_features']), "预测维度与输出特征数量不匹配"

        # 反归一化
        y_pred_inverse = self.data_processor.scalers['output'].inverse_transform(y_pred_2d)
        y_true_inverse = self.data_processor.scalers['output'].inverse_transform(y_true_2d)

        # 计算整体指标
        metrics['overall'] = {
            'RMSE': np.sqrt(mean_squared_error(y_true_inverse, y_pred_inverse)),
            'MAE': mean_absolute_error(y_true_inverse, y_pred_inverse),
            'R2': r2_score(y_true_inverse, y_pred_inverse)
        }

        try:
            metrics['overall']['MAPE'] = mean_absolute_percentage_error(y_true_inverse, y_pred_inverse) * 100
        except:
            metrics['overall']['MAPE'] = np.nan

        # 计算每个监测点和参数的指标
        for point_idx in range(num_points):
            point_name = downstream_points[point_idx]

            # 为每个监测点创建指标字典
            metrics[point_name] = {}

            for param_idx in range(num_params):
                param_name = water_parameters[param_idx]

                # 从整体反归一化数据中提取特定点和参数的数据
                # 计算该参数在展平后的索引位置
                feature_idx = point_idx * num_params + param_idx

                # 提取对应的反归一化数据
                y_true_point_param = y_true_inverse[:, feature_idx]
                y_pred_point_param = y_pred_inverse[:, feature_idx]

                # 计算指标
                metrics[point_name][param_name] = {
                    'RMSE': np.sqrt(mean_squared_error(y_true_point_param, y_pred_point_param)),
                    'MAE': mean_absolute_error(y_true_point_param, y_pred_point_param),
                    'R2': r2_score(y_true_point_param, y_pred_point_param)
                }

                try:
                    metrics[point_name][param_name]['MAPE'] = mean_absolute_percentage_error(
                        y_true_point_param, y_pred_point_param) * 100
                except:
                    metrics[point_name][param_name]['MAPE'] = np.nan

        # 输出评估结果
        print("\n测试集评估结果:")
        print(f"整体 RMSE: {metrics['overall']['RMSE']:.4f}")
        print(f"整体 MAE: {metrics['overall']['MAE']:.4f}")
        print(f"整体 R2: {metrics['overall']['R2']:.4f}")

        # 保存评估指标
        self._save_metrics(metrics)

        # 绘制预测结果
        self._plot_predictions(y_test_true, y_test_pred, downstream_points, water_parameters)

        return metrics

    def predict_last_days_validation(self, validation_window=45):
        """
        使用最后validation_window天的数据，其中前past_days天作为输入，预测后future_days天，
        并与实际值进行对比验证。具体使用数据集最后45天，前30天预测后15天，并输出带有正确日期的比较结果。

        Args:
            validation_window: 验证窗口的总天数 (past_days + future_days)

        Returns:
            预测结果和评估指标
        """
        if not self.trained:
            print("模型尚未训练，请先训练模型")
            return None

        past_days = self.config['past_days']
        future_days = self.config['future_days']

        # 验证窗口大小必须至少等于past_days + future_days
        if validation_window < past_days + future_days:
            validation_window = past_days + future_days
            print(f"验证窗口大小已调整为: {validation_window}")

        print(f"\n验证最后{validation_window}天数据，使用前{past_days}天预测后{future_days}天...")

        # 获取数据集的最后validation_window天
        last_days_data = self.merged_data.iloc[-validation_window:]
        print(f"验证数据范围: {last_days_data.index[0]} 至 {last_days_data.index[-1]}")

        # 提取输入特征和输出特征
        input_features = self.data['input_features']
        output_features = self.data['output_features']

        # 提取输入数据 (前past_days天)
        input_data = last_days_data[input_features].iloc[:past_days].values
        input_dates = last_days_data.index[:past_days]
        print(f"输入数据日期范围: {input_dates[0]} 至 {input_dates[-1]}")

        # 提取真实输出数据 (后future_days天)
        actual_output_data = last_days_data[output_features].iloc[past_days:past_days + future_days].values
        output_dates = last_days_data.index[past_days:past_days + future_days]
        print(f"预测数据日期范围: {output_dates[0]} 至 {output_dates[-1]}")

        # 使用保存的归一化器进行数据归一化
        input_scaled = self.data_processor.scalers['input'].transform(input_data)

        # 转换为模型输入格式
        model_input = torch.FloatTensor(input_scaled).unsqueeze(0).to(device)  # 添加批次维度

        # 预测
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(model_input)
            prediction = prediction.cpu().numpy()[0]  # 移除批次维度

        # 获取形状信息
        future_days_pred, num_points, num_params = prediction.shape

        # 创建结果目录
        validation_dir = os.path.join(self.output_dir, 'last_days_validation')
        os.makedirs(validation_dir, exist_ok=True)

        # 准备返回结果
        result = {}
        metrics = {}
        metrics['overall'] = {}

        # 获取监测点和参数名称
        downstream_points = self.data_processor.downstream_points
        water_parameters = self.data_processor.feature_groups['downstream']

        # 将预测结果重塑以便反归一化
        pred_flat = prediction.reshape(future_days_pred, num_points * num_params)

        # 反归一化预测结果
        pred_inverse = self.data_processor.scalers['output'].inverse_transform(pred_flat)

        # 整理预测结果和真实值
        all_predictions = []
        all_actuals = []

        # 创建输入时间和预测时间窗口的可视化
        plt.figure(figsize=(15, 5))
        plt.plot(input_dates, [0] * len(input_dates), 'bo-', markersize=4, label='输入时间窗口')
        plt.plot(output_dates, [0] * len(output_dates), 'ro-', markersize=4, label='预测时间窗口')
        plt.xlabel('日期')
        plt.title('模型输入和预测时间窗口')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(validation_dir, 'time_windows.png'), dpi=300)
        plt.close()

        # 创建汇总数据表（带日期的完整数据）
        all_validation_data = []

        for i, point in enumerate(downstream_points):
            result[point] = {}
            metrics[point] = {}

            for j, param in enumerate(water_parameters):
                # 计算特征索引
                feature_idx = i * num_params + j

                # 提取预测值和真实值
                pred_values = pred_inverse[:, feature_idx]
                actual_values = actual_output_data[:, feature_idx]

                # 存储结果
                result[point][param] = pred_values

                # 计算该点该参数的指标
                metrics[point][param] = {
                    'RMSE': np.sqrt(mean_squared_error(actual_values, pred_values)),
                    'MAE': mean_absolute_error(actual_values, pred_values),
                    'R2': r2_score(actual_values, pred_values)
                }

                try:
                    metrics[point][param]['MAPE'] = mean_absolute_percentage_error(actual_values, pred_values) * 100
                except:
                    metrics[point][param]['MAPE'] = np.nan

                # 收集所有预测值和真实值用于绘图和整体指标计算
                all_predictions.append(pred_values)
                all_actuals.append(actual_values)

                # 为每个监测点和参数创建CSV文件，使用实际日期
                comparison_df = pd.DataFrame({
                    '日期': output_dates,
                    '预测值': pred_values,
                    '实际值': actual_values,
                    '误差': pred_values - actual_values,
                    '相对误差%': np.abs((pred_values - actual_values) / actual_values) * 100
                })

                # 添加数据到汇总表
                for idx, date in enumerate(output_dates):
                    all_validation_data.append({
                        '日期': date,
                        '监测点': point,
                        '参数': param,
                        '预测值': pred_values[idx],
                        '实际值': actual_values[idx],
                        '误差': pred_values[idx] - actual_values[idx],
                        '相对误差%': np.abs((pred_values[idx] - actual_values[idx]) / actual_values[idx]) * 100 if
                        actual_values[idx] != 0 else np.nan
                    })

                # 保存CSV
                csv_path = os.path.join(validation_dir, f'{point}_{param}_validation.csv')
                comparison_df.to_csv(csv_path, index=False)

                # 绘制预测值与实际值对比图
                plt.figure(figsize=(12, 6))
                plt.plot(output_dates, actual_values, 'b-', linewidth=2, label='实际值')
                plt.plot(output_dates, pred_values, 'r-', linewidth=2, label='预测值')
                plt.title(f'{point} - {param} 预测vs实际 (RMSE: {metrics[point][param]["RMSE"]:.4f})', fontsize=14)
                plt.xlabel('日期', fontsize=12)
                plt.ylabel(param, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()

                # 保存图片
                plt.savefig(os.path.join(validation_dir, f'{point}_{param}_validation.png'), dpi=300)
                plt.close()

        # 保存汇总数据
        all_validation_df = pd.DataFrame(all_validation_data)
        all_validation_df.to_csv(os.path.join(validation_dir, 'all_validation_data.csv'), index=False)

        # 计算整体指标
        all_predictions = np.concatenate(all_predictions)
        all_actuals = np.concatenate(all_actuals)

        metrics['overall'] = {
            'RMSE': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
            'MAE': mean_absolute_error(all_actuals, all_predictions),
            'R2': r2_score(all_actuals, all_predictions)
        }

        try:
            metrics['overall']['MAPE'] = mean_absolute_percentage_error(all_actuals, all_predictions) * 100
        except:
            metrics['overall']['MAPE'] = np.nan

        # 保存评估指标
        self._save_validation_metrics(metrics, validation_dir)

        # 输出验证结果
        print("\n最后天数验证结果:")
        print(f"整体 RMSE: {metrics['overall']['RMSE']:.4f}")
        print(f"整体 MAE: {metrics['overall']['MAE']:.4f}")
        print(f"整体 R2: {metrics['overall']['R2']:.4f}")
        print(f"结果已保存到: {validation_dir}")

        return result, metrics, input_dates, output_dates

    def _save_validation_metrics(self, metrics, output_dir):
        """
        保存验证评估指标为CSV
        """
        # 准备整体指标
        overall_data = [{
            'Point': 'Overall',
            'Parameter': 'All',
            'RMSE': metrics['overall']['RMSE'],
            'MAE': metrics['overall']['MAE'],
            'MAPE': metrics['overall'].get('MAPE', np.nan),
            'R2': metrics['overall']['R2']
        }]

        # 准备各监测点各参数的指标
        detailed_data = []
        for point, params in metrics.items():
            if point == 'overall':
                continue

            for param, param_metrics in params.items():
                detailed_data.append({
                    'Point': point,
                    'Parameter': param,
                    'RMSE': param_metrics['RMSE'],
                    'MAE': param_metrics['MAE'],
                    'MAPE': param_metrics.get('MAPE', np.nan),
                    'R2': param_metrics['R2']
                })

        # 合并数据
        all_metrics = pd.DataFrame(overall_data + detailed_data)

        # 保存
        metrics_path = os.path.join(output_dir, 'validation_metrics.csv')
        all_metrics.to_csv(metrics_path, index=False)
        print(f"验证指标已保存到: {metrics_path}")

    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.output_dir, 'config.txt')
        with open(config_path, 'w') as f:
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")

    def _plot_training_history(self, train_losses, epochs):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=1.5)
        plt.title('模型训练历史')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()

        # 保存训练历史数据为CSV
        history_df = pd.DataFrame({
            'Epoch': range(1, len(train_losses) + 1),
            'TrainLoss': train_losses
        })
        history_df.to_csv(os.path.join(self.output_dir, 'training_history.csv'), index=False)

    def _save_metrics(self, metrics):
        """
        保存评估指标为CSV
        """
        # 准备整体指标
        overall_data = [{
            'Point': 'Overall',
            'Parameter': 'All',
            'RMSE': metrics['overall']['RMSE'],
            'MAE': metrics['overall']['MAE'],
            'MAPE': metrics['overall']['MAPE'],
            'R2': metrics['overall']['R2']
        }]

        # 准备各监测点各参数的指标
        detailed_data = []
        for point, params in metrics.items():
            if point == 'overall':
                continue

            for param, param_metrics in params.items():
                detailed_data.append({
                    'Point': point,
                    'Parameter': param,
                    'RMSE': param_metrics['RMSE'],
                    'MAE': param_metrics['MAE'],
                    'MAPE': param_metrics['MAPE'],
                    'R2': param_metrics['R2']
                })

        # 合并数据
        all_metrics = pd.DataFrame(overall_data + detailed_data)

        # 保存
        metrics_path = os.path.join(self.output_dir, 'evaluation_metrics.csv')
        all_metrics.to_csv(metrics_path, index=False)
        print(f"评估指标已保存到: {metrics_path}")

    def _plot_predictions(self, y_true, y_pred, downstream_points, water_parameters):
        """
        绘制预测结果图表
        """
        # 创建子目录
        plots_dir = os.path.join(self.output_dir, 'prediction_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 循环每个监测点和参数
        for point_idx, point in enumerate(downstream_points):
            for param_idx, param in enumerate(water_parameters):
                # 选择对应的数据
                true_values = y_true[:, :, point_idx, param_idx]
                pred_values = y_pred[:, :, point_idx, param_idx]

                # 平均所有批次的数据
                true_avg = np.mean(true_values, axis=0)
                pred_avg = np.mean(pred_values, axis=0)

                # 计算标准差（用于置信区间）
                pred_std = np.std(pred_values, axis=0)

                # 反归一化 - 找到对应的特征索引
                model_feature_name = f"{point}_{param}"
                feature_idx = None
                for idx, feature in enumerate(self.data['model_output_features']):
                    if feature == model_feature_name:
                        feature_idx = idx
                        break

                if feature_idx is not None:
                    # 为反归一化创建零矩阵
                    true_zeros = np.zeros((len(true_avg), len(self.data['output_features'])))
                    pred_zeros = np.zeros((len(pred_avg), len(self.data['output_features'])))
                    upper_zeros = np.zeros((len(pred_avg), len(self.data['output_features'])))
                    lower_zeros = np.zeros((len(pred_avg), len(self.data['output_features'])))

                    # 填充对应特征
                    true_zeros[:, feature_idx] = true_avg
                    pred_zeros[:, feature_idx] = pred_avg
                    upper_zeros[:, feature_idx] = pred_avg + 1.96 * pred_std
                    lower_zeros[:, feature_idx] = pred_avg - 1.96 * pred_std

                    # 反归一化
                    true_inv = self.data_processor.scalers['output'].inverse_transform(true_zeros)[:, feature_idx]
                    pred_inv = self.data_processor.scalers['output'].inverse_transform(pred_zeros)[:, feature_idx]
                    upper_inv = self.data_processor.scalers['output'].inverse_transform(upper_zeros)[:, feature_idx]
                    lower_inv = self.data_processor.scalers['output'].inverse_transform(lower_zeros)[:, feature_idx]

                    # 绘图
                    plt.figure(figsize=(10, 6))

                    # 天数作为x轴
                    days = range(1, self.config['future_days'] + 1)

                    # 绘制真实值和预测值
                    plt.plot(days, true_inv, 'b-', linewidth=1.5, label='实际值')
                    plt.plot(days, pred_inv, 'r-', linewidth=1.5, label='预测值')

                    # 绘制置信区间
                    plt.fill_between(days, lower_inv, upper_inv, color='r', alpha=0.1, label='95% 置信区间')

                    plt.title(f'{point} - {param} 预测结果')
                    plt.xlabel('预测天数')
                    plt.ylabel(param)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # 保存图片
                    plt.savefig(os.path.join(plots_dir, f'{point}_{param}_prediction.png'), dpi=300)
                    plt.close()

                    # 保存数据为CSV
                    pred_df = pd.DataFrame({
                        'Day': days,
                        'Actual': true_inv,
                        'Predicted': pred_inv,
                        'LowerBound': lower_inv,
                        'UpperBound': upper_inv
                    })
                    pred_df.to_csv(os.path.join(plots_dir, f'{point}_{param}_prediction.csv'), index=False)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='排口对下游水质影响XLSTM预测模型')

    # 数据文件
    parser.add_argument('--discharge_file', type=str, required=True, help='排口排放数据文件路径')
    parser.add_argument('--downstream_file', type=str, required=True, help='下游监测点水质数据文件路径')
    parser.add_argument('--upstream_file', type=str, default=None, help='上游水质数据文件路径')
    parser.add_argument('--env_file', type=str, default=None, help='环境因素数据文件路径')

    # 模型参数
    parser.add_argument('--past_days', type=int, default=30, help='用于预测的过去天数')
    parser.add_argument('--future_days', type=int, default=15, help='需要预测的未来天数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='XLSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='XLSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比率')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')

    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')

    # 验证参数
    parser.add_argument('--validate_last_days', action='store_true',
                        help='使用数据集最后的天数进行验证，用前past_days天预测后future_days天并与实际值对比')
    parser.add_argument('--validation_window', type=int, default=45,
                        help='验证窗口总天数，默认为45天')
    parser.add_argument('--only_validate', action='store_true',
                        help='仅执行最后天数验证，跳过完整训练和评估过程')

    args = parser.parse_args()

    # 创建配置字典
    config = {
        'past_days': args.past_days,
        'future_days': args.future_days,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'test_size': args.test_size
    }

    # 创建模型实例
    discharge_model = DischargeImpactModel(config)

    # 设置数据处理器
    discharge_model.setup()

    # 加载数据
    data = discharge_model.load_data(
        discharge_file=args.discharge_file,
        downstream_file=args.downstream_file,
        upstream_file=args.upstream_file,
        env_file=args.env_file
    )

    # 构建模型
    discharge_model.build_model()

    # 如果不是只执行验证，则进行完整训练和评估
    if not args.only_validate:
        # 训练模型
        discharge_model.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )

        # 评估模型
        metrics = discharge_model.evaluate()
    else:
        # 如果是只执行验证模式，使用预训练模型
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"discharge_impact_{timestamp}"

        os.makedirs(args.output_dir, exist_ok=True)
        discharge_model.output_dir = args.output_dir
        print("跳过训练过程，直接加载预训练模型...")

        # 这里假设已有预训练模型，实际使用时需要先手动训练一个模型
        model_path = os.path.join(args.output_dir, 'discharge_impact_xlstm.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            discharge_model.model.load_state_dict(checkpoint['model_state_dict'])
            discharge_model.trained = True
            print(f"已加载预训练模型: {model_path}")
        else:
            # 如果没有预训练模型，则进行简短训练
            print(f"未找到预训练模型，进行简短训练...")
            discharge_model.train(
                epochs=min(10, args.epochs),  # 只训练几轮
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir
            )

    # 执行最后天数验证
    if args.validate_last_days or args.only_validate:
        validation_window = args.validation_window
        results, validation_metrics, input_dates, output_dates = discharge_model.predict_last_days_validation(
            validation_window)
        print(f"\n已使用最后{validation_window}天数据，用前{args.past_days}天预测后{args.future_days}天并与实际值对比")
        print(f"输入日期范围: {input_dates[0]} 至 {input_dates[-1]}")
        print(f"预测日期范围: {output_dates[0]} 至 {output_dates[-1]}")
        print(f"验证结果已保存到: {os.path.join(discharge_model.output_dir, 'last_days_validation')}")

    print("\n模型训练和评估完成")
    print(f"结果保存在: {discharge_model.output_dir}")


if __name__ == "__main__":
    main()