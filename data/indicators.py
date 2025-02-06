'''
滑动窗口 内技术指标计算
'''
from pandas import DataFrame


def compute_rsi(df:DataFrame, column='close', window=14):
    """
    计算RSI（相对强弱指数）

    :param df: 包含市场数据的 DataFrame
    :param column: 用来计算 RSI 的列，默认是 'close'
    :param window: RSI 计算的窗口大小，默认是14

    :return: 返回计算后的 RSI 值（pandas Series）
    """
    # 计算价格变动
    delta = df[column].diff()

    # 分离涨幅和跌幅
    gain = delta.where(delta > 0, 0)  # 涨幅（如果是负数，则为0）
    loss = -delta.where(delta < 0, 0)  # 跌幅（如果是正数，则为0）

    # 计算平均涨幅和平均跌幅
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # 计算相对强度（RS）
    rs = avg_gain / avg_loss

    # 计算RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi