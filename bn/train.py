import pandas as pd
import talib
from binance import Client
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from bn import cl
from constants import SymbolEnum


class CryptoEngineer:

    def __init__(self):
        self.scaler = StandardScaler()

    def get_binance_data(self) -> DataFrame:
        klines = cl.get_historical_klines(SymbolEnum.DOGEUSDT.value, Client.KLINE_INTERVAL_15MINUTE, "2025-01-15",
                                          "2025-02-05")
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
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'], timeperiod=20)  # 布林带
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26,
                                                                          signalperiod=9)  # MACD

        # 删除空值
        data = data.dropna()

        return data

    def standardize_data(self, data):
        features = ['close', 'SMA', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_hist']

        # 对选定特征进行标准化
        data[features] = self.scaler.fit_transform(data[features])

        return data

    def df_to_ary(
            self,
            df,
    ):
        df['time'] = pd.to_datetime(df.index)
        df.set_index('time', inplace=True)

        # 提取 price_array，通常取的是 'close' 列
        price_array = df[['close']].values  # 这里获取的是价格数组

        # 提取技术指标的值，这些列包括：'SMA', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_hist'
        tech_indicator_list = [
            'SMA', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'MACD_signal', 'MACD_hist'
        ]
        tech_array = df[tech_indicator_list].values  # 获取技术指标数组

        # 提取 date_ary，时间戳数组
        date_ary = df.index.values  # 这里获取的是日期数组

        return date_ary, price_array, tech_array


if __name__ == '__main__':
    engineer = CryptoEngineer()
    df = engineer.get_binance_data()
    df = engineer.preprocess_data(df)
    df = engineer.standardize_data(df)
    date_ary, price_array, tech_array = engineer.df_to_ary(df)

