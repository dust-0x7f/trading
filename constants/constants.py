from enum import Enum

class TimePeriodEnum(Enum):
    ONE_MINUTE = '1m'
    THREE_MINUTES = '3m'
    FIVE_MINUTES = '5m'
    FIFTEEN_MINUTES = '15m'
    THIRTY_MINUTES = '30m'
    ONE_HOUR = '1h'
    TWO_HOURS = '2h'
    FOUR_HOURS = '4h'
    SIX_HOURS = '6h'
    EIGHT_HOURS = '8h'
    TWELVE_HOURS = '12h'
    ONE_DAY = '1d'
    ONE_SECOND = '1s'

class SymbolEnum(Enum):
    DOGEUSDT = 'DOGEUSDT'
    BTCUSDT = 'BTCUSDT'
    ETHUSDT = 'ETHUSDT'
    XRPUSDT = 'XRPUSDT'
    LTCUSDT = 'LTCUSDT'
    ADAUSDT = 'ADAUSDT'
    SOLUSDT = 'SOLUSDT'
    BNBUSDT = 'BNBUSDT'

