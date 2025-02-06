import pandas as pd


# 读取CSV数据
df = pd.read_csv("sample-15m-doge-usdt.csv", header=None)

# 设置列名
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_timestamp', 'close_volume', 'trade_count', 'buy_volume', 'sell_volume', 'unknown']

# 输出DataFrame
print(df)
