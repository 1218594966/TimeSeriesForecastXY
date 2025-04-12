import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse


def generate_synthetic_data(n_days=365, n_features=3, seed=42):
    """
    生成合成的多变量时间序列数据
    :param n_days: 生成的天数
    :param n_features: 特征数量
    :param seed: 随机种子
    :return: 包含时间序列数据的DataFrame
    """
    np.random.seed(seed)

    # 创建日期范围
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # 创建基本趋势和季节性成分
    t = np.arange(n_days)
    trend = 0.01 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 30) + 3 * np.cos(2 * np.pi * t / 7)

    # 初始化DataFrame
    df = pd.DataFrame()
    df['date'] = dates

    # 为每个特征生成数据
    for i in range(n_features):
        # 每个特征使用不同的振幅和相位
        amplitude = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)

        # 生成特征值 = 趋势 + 季节性 + 噪声
        feature_trend = trend * np.random.uniform(0.5, 1.5)
        feature_seasonality = amplitude * seasonality * np.sin(t / 180 * np.pi + phase)
        noise = np.random.normal(0, 1, n_days)

        feature_values = 100 + feature_trend + feature_seasonality + noise
        df[f'feature_{i + 1}'] = feature_values

    return df


def main(args):
    df = generate_synthetic_data(args.days, args.features, args.seed)
    df.to_csv(args.output, index=False)
    print(f"已生成{args.days}天、{args.features}个特征的合成数据并保存到 {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成合成的多变量时间序列数据')
    parser.add_argument('--days', type=int, default=365, help='生成的天数')
    parser.add_argument('--features', type=int, default=3, help='特征数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default='synthetic_data.csv', help='输出文件名')

    args = parser.parse_args()
    main(args)