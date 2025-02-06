import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from pandas import DataFrame

from constants.constants import TimePeriodEnum, SymbolEnum
from data.indicators import compute_rsi

# 设置全局列名变量
Spot_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_timestamp', 'close_volume', 'trade_count',
                'buy_volume', 'sell_volume', 'ignore']

request_url = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{period}/{symbol}-{period}-{year}-{month}-{day}.zip"
file_name_template = "./zips/{symbol}-{period}-{year}-{month}-{day}.zip"

'''
下载现货k线图
'''


def download_spot_klines(symbol: SymbolEnum, period: TimePeriodEnum, year: int, month: str, day: str) -> str:
    # 格式化 request_url 和 file_name_template
    url = request_url.format(symbol=symbol.value, period=period.value, year=year, month=month, day=day)
    file_name = file_name_template.format(symbol=symbol.value, period=period.value, year=year, month=month, day=day)
    file_path = os.path.join(os.getcwd(), file_name)  # 获取完整路径
    file_dir = os.path.dirname(file_path)  # 获取目录路径

    # 如果目录不存在，则创建目录
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
        print(f"Directory {file_dir} created.")

    # 判断文件是否已经存在
    if os.path.exists(file_name):
        pass
    else:
        # 打印 URL 和文件名（你也可以执行下载操作）
        # 请求数据
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"File {file_name} downloaded successfully!")
        else:
            print(f"Failed to download file: {response.status_code}")
            raise Exception(f"download file {file_name} fail")
    return file_path


'''
读取下载现货的k线图
'''


def read_spot_kines(file_path_name: str) -> DataFrame:
    # 读取CSV数据
    df = pd.read_csv(file_path_name, header=None)

    return df


def get_day_df(symbol: SymbolEnum, period: TimePeriodEnum, time_end: str, window_size=30) -> DataFrame:
    start_date = datetime.strptime(time_end, '%Y-%m-%d')
    all_data = []

    for i in range(window_size):
        day = start_date - timedelta(days=i)
        year = day.year
        month = day.month
        day_of_month = day.day

        # 下载该天的数据
        try:
            file_name = download_spot_klines(symbol, period, year, f'{month:02d}', f'{day_of_month:02d}')
            df = read_spot_kines(file_name)
            all_data.append(df)
        except Exception as e:
            raise e
    res = pd.concat(all_data, ignore_index=True)
    res.columns = Spot_columns
    return res


def data_clean(df: DataFrame) -> DataFrame:
    return df.dropna()



if __name__ == '__main__':
    df = get_day_df(SymbolEnum.DOGEUSDT, TimePeriodEnum.FIVE_MINUTES, "2024-02-05")
    df = data_clean(df)
    print(df.columns)
    df['RSI'] = compute_rsi(df, window=14)
    print(df.head(50))
