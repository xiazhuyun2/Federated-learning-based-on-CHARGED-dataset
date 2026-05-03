"""
TCN-LSTM 混合神经网络 — 本地客户端模型

架构:
  Input [B, seq_len, features]
    -> TCN (多层空洞因果卷积, 提取多尺度局部模式)
    -> LSTM (捕获长期时序依赖)
    -> FC Head (映射到预测窗口)
  Output [B, pred_len]
"""
import torch
import torch.nn as nn
from typing import List


# ============================================================
# Temporal Convolutional Network (TCN)
# ============================================================

class CausalConv1d(nn.Module):
    """因果卷积: 确保不会泄露未来信息"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        out = self.conv(x)
        # 裁剪右侧 padding 保证因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """单个 TCN 残差块: CausalConv -> BN -> ReLU -> Dropout -> CausalConv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 卷积用于残差连接维度匹配
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))

        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCN(nn.Module):
    """时间卷积网络: 多层 TCN Block, 空洞率指数增长"""

    def __init__(self, input_dim: int, channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, input_dim, seq_len) -> (B, channels[-1], seq_len)"""
        return self.network(x)


# ============================================================
# TCN-LSTM 混合模型
# ============================================================

class TCNLSTM(nn.Module):
    """
    TCN-LSTM 充电负荷预测模型
    Input:  (B, seq_len, input_dim)
    Output: (B, pred_len)
    """

    def __init__(
        self,
        input_dim: int,
        pred_len: int = 24,
        tcn_channels: List[int] = None,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        fc_hidden: int = 64,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 64, 64]

        self.pred_len = pred_len

        # TCN 部分: 提取多尺度局部特征
        self.tcn = TCN(input_dim, tcn_channels, tcn_kernel_size, tcn_dropout)

        # LSTM 部分: 捕获长期时序依赖
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # 全连接输出头
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden, pred_len),
        )

    def forward(self, x):
        """
        x: (B, seq_len, input_dim)
        """
        # TCN 需要 (B, C, L) 格式
        tcn_out = self.tcn(x.permute(0, 2, 1))   # -> (B, tcn_ch, seq_len)
        tcn_out = tcn_out.permute(0, 2, 1)         # -> (B, seq_len, tcn_ch)

        # LSTM
        lstm_out, _ = self.lstm(tcn_out)            # -> (B, seq_len, lstm_hidden)

        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]            # -> (B, lstm_hidden)

        # 预测
        pred = self.fc(last_hidden)                 # -> (B, pred_len)
        return pred


def build_model(input_dim: int, pred_len: int, model_cfg) -> TCNLSTM:
    """根据配置构建模型"""
    return TCNLSTM(
        input_dim=input_dim,
        pred_len=pred_len,
        tcn_channels=model_cfg.tcn_channels,
        tcn_kernel_size=model_cfg.tcn_kernel_size,
        tcn_dropout=model_cfg.tcn_dropout,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        lstm_dropout=model_cfg.lstm_dropout,
        fc_hidden=model_cfg.fc_hidden,
    )
