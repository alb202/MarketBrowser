import datetime as datetime

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday

USFEDHOLIDAYS = USFederalHolidayCalendar()
USFEDHOLIDAYS.merge(GoodFriday, inplace=True)
MARKET_HOLIDAYS = [i.astype(datetime.datetime).strftime('%Y-%m-%d') for i in
                   list(pd.offsets.CustomBusinessDay(calendar=USFEDHOLIDAYS)
                        .__dict__['holidays'])][200:700]


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
