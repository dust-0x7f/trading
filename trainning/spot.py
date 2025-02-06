import numpy as np

from constants import SymbolEnum, TimePeriodEnum
from data.spot import get_day_df

features = ['open','buy_volume','sell_volume']
targets = ['close','rsi']

def create_3d_data(df, window_size, features, targets):
    # 将 DataFrame 转换为 numpy 数组
    data = df[features + targets].values  # 选择 features 和 targets 列并转换为 numpy 数组

    x, y = [], []
    for i in range(len(data) - window_size):
        # 取出特征数据
        x.append(data[i:i + window_size, :len(features)])  # 取前 len(features) 列作为特征
        # 取出目标数据
        y.append(data[i + window_size, len(features):])  # 从 len(features) 开始取目标列
    return np.array(x), np.array(y)



if __name__ == '__main__':
    df = get_day_df(SymbolEnum.DOGEUSDT,TimePeriodEnum.FIFTEEN_MINUTES,time_end="2025-02-05")
    x_axis,y_axis = create_3d_data(df,48,features,targets)


    train_size = int(len(x_axis) * 0.8)
    x_train, x_test = x_axis[:train_size], x_axis[train_size:]
    y_train, y_test = y_axis[:train_size], y_axis[train_size:]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], len(features)))



