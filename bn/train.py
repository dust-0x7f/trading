import pandas as pd
import talib
from binance import Client
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from bn import cl
from constants import SymbolEnum


def get_binance_data() -> DataFrame:
    klines = cl.get_historical_klines(SymbolEnum.DOGEUSDT.value, Client.KLINE_INTERVAL_15MINUTE, "2025-01-15",
                                      "2025-02-05")
    data = pd.DataFrame(klines,
                        columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                 "quote_asset_volume",
                                 "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                                 "ignore"])
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit='ms')
    data.set_index("timestamp", inplace=True)
    return data


def preprocess_data(data):
    # 将数据类型转换为适合的格式
    data['close'] = data['close'].astype(float)
    data['open'] = data['open'].astype(float)
    data['high'] = data['high'].astype(float)
    data['low'] = data['low'].astype(float)
    data['volume'] = data['volume'].astype(float)

    # 添加常见的技术指标
    data['SMA'] = talib.SMA(data['close'], timeperiod=20)  # 20日简单移动平均
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)  # 14日相对强弱指数
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'], timeperiod=20)  # 布林带
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26,
                                                                      signalperiod=9)  # MACD

    # 删除空值
    data = data.dropna()

    return data


def standardize_data(data):
    scaler = StandardScaler()
    features = ['close', 'SMA', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_hist']

    # 对选定特征进行标准化
    data[features] = scaler.fit_transform(data[features])

    return data


df = get_binance_data()
# 对数据进行预处理
fe = FeatureEngineer()
df = fe.preprocess_data(df)
print(df)
