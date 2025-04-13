#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


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


# Simplified xLSTM Block Stack implementation
class xLSTMBlockStack(nn.Module):
    def __init__(self, config):
        super(xLSTMBlockStack, self).__init__()
        self.config = config

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=config.embedding_dim if i > 0 else config.embedding_dim,
                hidden_size=config.embedding_dim,
                batch_first=True
            ) for i in range(config.num_blocks)
        ])

        # Create Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.embedding_dim) for _ in range(config.num_blocks)
        ])

        # Create gating mechanism
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.embedding_dim, config.embedding_dim),
                nn.Sigmoid()
            ) for _ in range(config.num_blocks)
        ])

        # Post-processing normalization
        if config.add_post_blocks_norm:
            self.post_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.post_norm = nn.Identity()

    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        batch_size, seq_len, _ = x.shape

        for i in range(self.config.num_blocks):
            # LSTM processing
            lstm_out, _ = self.lstm_layers[i](x)

            # Apply gating mechanism
            gate = self.gates[i](lstm_out)
            gated_output = gate * lstm_out

            # Residual connection and normalization
            x = x + gated_output
            x = self.layer_norms[i](x)

        # Post-processing normalization
        x = self.post_norm(x)
        return x


# Official xLSTM model
class XLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_points, target_points, num_blocks=3, dropout=0.1):
        super(XLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_points = context_points
        self.target_points = target_points

        # Configure xLSTM
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

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)

        # Decomposition layer
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        # Seasonal and trend linear layers
        self.Linear_Seasonal = nn.Linear(context_points, target_points)
        self.Linear_Trend = nn.Linear(context_points, target_points)

        # Initialize weights
        self.Linear_Seasonal.weight = nn.Parameter((1 / context_points) * torch.ones([target_points, context_points]))
        self.Linear_Trend.weight = nn.Parameter((1 / context_points) * torch.ones([target_points, context_points]))

        # Projection layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # xLSTM stack
        self.xlstm_stack = xLSTMBlockStack(config)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # Time series decomposition
        seasonal_init, trend_init = self.decomposition(x)

        # Transform dimensions for linear layers
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        # Apply seasonal and trend linear layers
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # Combine seasonal and trend
        combined = seasonal_output + trend_output  # [batch_size, input_dim, target_points]

        # Project to hidden dimension
        projected = self.input_projection(combined.permute(0, 2, 1))  # [batch_size, target_points, hidden_dim]

        # Apply xLSTM processing
        xlstm_out = self.xlstm_stack(projected)  # [batch_size, target_points, hidden_dim]

        # Dropout
        xlstm_out = self.dropout(xlstm_out)

        # Project back to output dimension
        output = self.output_projection(xlstm_out)  # [batch_size, target_points, output_dim]

        return output  # Returns shape [batch_size, target_points, output_dim]