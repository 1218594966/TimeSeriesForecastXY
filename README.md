# 多变量时间序列预测 xLSTM 模型

基于PyTorch实现的多变量多步时间序列预测模型，使用最新的xLSTM架构进行长期时序预测。支持置信区间可视化、多维度评估指标，并提供完整的数据预处理和结果保存功能。

## 主要特性
- 🕰️ 支持多变量时间序列预测
- 🔮 实现xLSTM（mLSTM + sLSTM混合架构）
- 📈 完整数据预处理流程（缺失值处理、归一化等）
- 📊 丰富的可视化功能（趋势对比、置信区间）
- 📝 自动生成评估报告和训练日志
- ⚡ 支持GPU加速训练
## 快速开始
- 准备CSV格式的数据文件，需包含： 时间列（自动识别列名包含"date"或"time"的列）
- 示例数据格式：
date	feature1	feature2	...

2023/01/01	0.523	1.234	...
## 运行后自动生成包含以下内容的输出目录：
 ├── config.txt             # 保存的训练配置
 
 ├── xlstm_model.pth        # 训练好的模型权重
 
 ├── training_history.png   # 训练损失曲线
 
 ├── test_metrics.csv       # 测试集评估指标
 
 ├── prediction_*.png       # 各特征预测对比图
 
 ├── test_confidence_*.png  # 带置信区间的预测图
 
 └── training_history.csv   # 训练过程数据


## 环境要求
- Python 3.7+
- PyTorch 1.8+
- CUDA 11.3+ (可选)


