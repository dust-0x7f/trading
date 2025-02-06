import os

import requests

from constants.constants import TimePeriodEnum, SymbolEnum
import pandas as pd

# 设置全局列名变量
COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
           'close_timestamp', 'close_volume', 'trade_count',
           'buy_volume', 'sell_volume', 'unknown']



request_url = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{period}/{symbol}-{period}-{year}-{month}-{day}.zip"
file_name_template = "./zips/{symbol}-{period}-{year}-{month}-{day}.zip"


'''
下载现货k线图
'''
def download_spot_klines(symbol: SymbolEnum, period: TimePeriodEnum, year, month, day):
    # 格式化月份和日期，确保它们是两位数
    month = f'{month:02d}'
    day = f'{day:02d}'

    # 格式化 request_url 和 file_name_template
    url = request_url.format(symbol=symbol.value, period=period.value, year=year, month=month, day=day)
    file_name = file_name_template.format(symbol=symbol.value, period=period.value, year=year, month=month, day=day)

    # 判断文件是否已经存在
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Skipping download.")
    else:
        # 打印 URL 和文件名（你也可以执行下载操作）
        print(f"URL: {url}")
        print(f"File Name: {file_name}")

        # 请求数据
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"File {file_name} downloaded successfully!")
        else:
            print(f"Failed to download file: {response.status_code}")

    # 读取并处理数据
    read_spot_kines(file_name)

'''
读取下载现货的k线图
'''
def read_spot_kines(file_name:str):
    # 读取CSV数据
    df = pd.read_csv(file_name, header=None)

    # 设置列名
    df.columns = COLUMNS

    # 输出DataFrame
    print(df)


if __name__ == '__main__':
    download_spot_klines(SymbolEnum.DOGEUSDT,TimePeriodEnum.FIFTEEN_MINUTES,2025,2,1)