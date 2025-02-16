import os

import numpy as np
import pandas as pd
import talib
from binance import Client
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset

from bn import cl
from constants import SymbolEnum



input_size = 5  # 输入特征数（例如：开盘价、最高价、最低价、收盘价、成交量等）
model_dim = 64  # Transformer模型的维度
num_heads = 8  # 注意力头数
num_layers = 3  # Transformer层数
output_size = 1  # 预测的目标（如：下一步的收盘价）

past_data_len = 20
predict_data_len = 3


class CryptoEngineer:

    def get_binance_data(self,time_start:str,time_end:str) -> DataFrame:
        klines = cl.get_historical_klines(SymbolEnum.DOGEUSDT.value, Client.KLINE_INTERVAL_15MINUTE, time_start,
                                          time_end)
        data = pd.DataFrame(klines,
                            columns=["time", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_asset_volume",
                                     "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                                     "ignore"])
        data["time"] = pd.to_datetime(data["time"], unit='ms')
        data.set_index("time", inplace=True)
        return data

    def preprocess_data(self, data):
        # 将数据类型转换为适合的格式
        data['close'] = data['close'].astype(float)
        data['open'] = data['open'].astype(float)
        data['high'] = data['high'].astype(float)
        data['low'] = data['low'].astype(float)
        data['volume'] = data['volume'].astype(float)

        # 添加常见的技术指标
        data['SMA'] = talib.SMA(data['close'], timeperiod=20)  # 20日简单移动平均
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)  # 14日相对强弱指数
        # data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'], timeperiod=20)  # 布林带
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26,
                                                                          signalperiod=9)  # MACD

        # 删除空值
        data = data.dropna()

        return data


import torch.optim as optim
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, model_dim, num_heads, num_layers, output_size):
        super(TransformerModel, self).__init__()

        # 输入的嵌入层
        self.input_size = input_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 输入嵌入层 (将输入的多个特征映射到模型维度)
        self.embedding = nn.Linear(input_size, model_dim)

        # Transformer 层
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        # 输出层 (将模型的输出映射到预测的收盘价)
        self.output_layer = nn.Linear(model_dim, output_size)

    def forward(self, x):
        # 将输入数据通过线性层转换为模型维度
        x = self.embedding(x)

        # Transformer 模型的前向传播 (使用自身输入进行解码)
        x = self.transformer(x, x)

        # 这里假设模型的输出是(批次, 序列长度, 模型维度)
        x = x[:, -predict_data_len:, :]  # 获取未来n个时间步的输出

        # 通过输出层映射到最终预测的收盘价
        x = self.output_layer(x)

        return x


def get_train_data(df: DataFrame, past_data_len, predict_data_len):
    X = []
    y = []
    # features = ['SMA', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_hist']
    features = ['SMA', 'RSI',  'MACD', 'MACD_signal', 'MACD_hist']
    target_column = 'close'  # 预测目标列

    for i in range(past_data_len, len(df) - predict_data_len + 1):  # 从 used_data_len 开始到剩余足够预测数据的位置
        X.append(df.iloc[i - past_data_len:i][features].values)  # 使用 iloc 索引位置数据
        y.append(df.iloc[i:i + predict_data_len][target_column].values)  # 使用 iloc 索引位置数据

    return np.array(X), np.array(y)



def train():
    engineer = CryptoEngineer()
    df = engineer.get_binance_data("2024-06-05","2024-10-05")
    df = engineer.preprocess_data(df)
    tech_array, price_array = get_train_data(df, past_data_len, predict_data_len)

    # 定义模型的超参数

    # 创建模型
    model = TransformerModel(input_size, model_dim, num_heads, num_layers, output_size)
    model_path = "model_epoch.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)  # 加载先前保存的模型参数
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    else:
        print(f"No saved model found at {model_path}, starting from scratch.")


    X_train = torch.tensor(tech_array, dtype=torch.float32)
    y_train = torch.tensor(price_array, dtype=torch.float32)
    y_train = y_train.unsqueeze(-1)  # 在最后一维增加一个维度，使其形状变为 (batch_size, seq_len, 1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512)

    num_epochs = 10
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")


def test():
    engineer = CryptoEngineer()
    model = TransformerModel(input_size, model_dim, num_heads, num_layers, output_size)
    model_path = "model_epoch.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)  # 加载先前保存的模型参数
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    else:
        print(f"No saved model found at {model_path}, starting from scratch.")

    #  训练完成，回测一下
    test_data = engineer.get_binance_data("2024-11-06","2024-12-06")
    test_data = engineer.preprocess_data(test_data)
    test_tech_array, test_price_array = get_train_data(test_data, past_data_len, predict_data_len)

    max_balance = 0
    initial_balance = 10000  # 初始资金
    balance = initial_balance
    for i in range(past_data_len, len(test_price_array)):  # 保证有足够的未来数据来做判断
        input_data = torch.tensor(test_tech_array[i:i + 1], dtype=torch.float32)  # 当前时间步的数据
        with torch.no_grad():
            predict_price = model(input_data)  # 使用模型预测下一步的价格
        predict_price = predict_price.squeeze()
        v = test_price_array[i + predict_data_len][-1]
        print(f'prefict_price: {predict_price[-1]},real_price: {v}')

    # 7. 计算回测结果
    final_balance = balance
    profit = final_balance - initial_balance
    print(f"Final Balance: {final_balance}")
    print(f"Total Profit: {profit}")
    print(f"Max Balance: {max_balance}")

    # 8. 可视化回测结果
    plt.figure(figsize=(12, 6))
    plt.plot(capital_history, label="Portfolio Value")
    plt.title('Backtest Portfolio Value')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # train()
    test()