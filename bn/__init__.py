import json

from binance import Client

api_key = ""
api_secret = ""
# 你的API Key和API Secret
with open('bn.json', 'r') as file:
    data = json.load(file)
    api_key = data['api_key']
    api_secret = data['api_secret']

cl = Client(api_key, api_secret)  # 在模块级别初始化
