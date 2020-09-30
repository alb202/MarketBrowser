import datetime as datetime

import main
import pandas as pd
from indicators import *
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday
from retracements import *

USFEDHOLIDAYS = USFederalHolidayCalendar()
USFEDHOLIDAYS.merge(GoodFriday, inplace=True)
MARKET_HOLIDAYS = [i.astype(datetime.datetime).strftime('%Y-%m-%d') for i in
                   list(pd.offsets.CustomBusinessDay(calendar=USFEDHOLIDAYS)
                        .__dict__['holidays'])][200:700]

FUNCTION_LOOKUP = {1: 'TIME_SERIES_MONTHLY_ADJUSTED',
                   2: 'TIME_SERIES_WEEKLY_ADJUSTED',
                   3: 'TIME_SERIES_DAILY_ADJUSTED',
                   4: 'TIME_SERIES_INTRADAY',
                   5: 'TIME_SERIES_INTRADAY',
                   6: 'TIME_SERIES_INTRADAY',
                   7: 'TIME_SERIES_INTRADAY'}

INTERVAL_LOOKUP = {4: '60min',
                   5: '30min',
                   6: '15min',
                   7: '5min'}


def make_rangebreaks(function):
    """Set the range breaks for x axis
    """
    if 'INTRADAY' in function:
        return [
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=MARKET_HOLIDAYS),
            dict(pattern='hour', bounds=[16, 9.5])  # hide Christmas and New Year's
        ]

    if 'DAILY' in function:
        return [
            dict(bounds=["sat", "mon"]),  # ,  # hide weekends
            dict(values=MARKET_HOLIDAYS),
        ]
    return None


def get_layout_params(symbol):
    """Create the layout parameters
    """
    symbol = symbol if symbol is not None else ' '
    layout = dict(
        width=1200,
        height=1200,
        title=symbol,
        xaxis1=dict(rangeselector=dict(buttons=list([
            dict(count=5, label="5d", step="day", stepmode="backward"),
            dict(count=15, label="15d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")])), type="date", rangeslider=dict(visible=False)),
        yaxis1=dict(title=dict(text="Price $ - US Dollars")),
        yaxis2=dict(title=dict(text="Volume")),
        yaxis3=dict(title=dict(text="MACD")),
        yaxis4=dict(title=dict(text="RSI")))
    # yaxis5=dict(title=dict(text="Hiekin Ashi")),
    # yaxis6=dict(title=dict(text="Moving Average Crossover")))
    return layout


def round_float_columns(data, digits=2):
    dtypes = data.dtypes
    print(dtypes)
    round_cols = [i for i, j in dtypes.items() if j == 'float64']
    print(round_cols)
    # new_data = data.copy(deep=True)
    for i in round_cols:
        data[i] = data[i].round(decimals=2)
    return data


def process_symbol_input(symbols):
    symbols = [i.strip(' ').upper() for i in symbols
        .replace('\n', '')
        .replace(',', ' ')
        .replace(';', ' ')
        .strip(' ')
        .split(' ')]
    return sorted(list({i for i in symbols if i is not None}))


def get_price_data(n_clicks, symbol, function, interval, no_api=False):
    """Get the data from main
    """
    if (('INTRADAY' in function) & (interval is None)) | \
            (n_clicks == 0) | \
            (symbol is None) | \
            (function is None):
        return pd.DataFrame({'datetime': [],
                             'open': [],
                             'high': [],
                             'low': [],
                             'close': [],
                             'volume': []})

    return main.main(
        {'function': [function],
         'symbol': [symbol.upper()],
         'interval': [interval],
         'config': None,
         'get_all': False,
         'no_return': False,
         'data_status': False,
         'get_symbols': False,
         'no_api': no_api})['prices']


def create_indicators(data, function, indicators={1, 2, 3, 4, 5}):
    """Calculate the indicators for the table
    """
    if (len(data) == 0) | (len(indicators) == 0):
        return pd.DataFrame.from_dict({i: [] for i in ['datetime']}, orient='columns')

    results = data.loc[:, ['datetime', 'open', 'high', 'low', 'close']].set_index('datetime')

    if 1 in indicators:
        _macd = MACD(data=data.loc[:, ['datetime', 'close']], function=function)
        _macd_df = _macd.table()
        results = results.join(_macd_df, how='outer')

    if 2 in indicators:
        _rsi = RSI(data=data.loc[:, ['datetime', 'close']], function=function)
        _rsi_df = _rsi.table()
        results = results.join(_rsi_df, how='outer')

    if len({3, 4, 5}.intersection(indicators)) > 0:
        _ha = HeikinAshi(data=data.loc[:, ['datetime', 'open', 'high', 'low', 'close']], function=function)
        _ha_df = _ha.table()

        ha_data_mac = _ha_df[['ha_close']].reset_index(drop=False)
        ha_data_maz = _ha_df[['ha_open', 'ha_close']].reset_index(drop=False)
        ha_data_mac.columns = ['datetime', 'close']
        ha_data_maz.columns = ['datetime', 'open', 'close']

        if 3 in indicators:
            _mac = MovingAverageCrossover(data=ha_data_mac,
                                          function=function,
                                          ma1_period=10,
                                          ma2_period=20)
            _mac_df = _mac.table()
            results = results.join(_mac_df, how='outer')

        if 4 in indicators:
            _maz = MovingAverageZone(datetime=ha_data_maz['datetime'],
                                     open=ha_data_maz['open'],
                                     close=ha_data_maz['close'],
                                     function=function)
            _maz_df = _maz.table()
            results = results.join(_maz_df, how='outer')

        if 5 in indicators:
            _retrace = Retracements(high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    dates=data['datetime'],
                                    function=function)
            _retrace.get_retracements(low=.38, high=.6)
            _retrace_df = _retrace.table()
            results = results.join(_retrace_df, how='outer')

    results = round_float_columns(data=results, digits=2)

    return results.reset_index(drop=False).sort_values('datetime', ascending=False)
