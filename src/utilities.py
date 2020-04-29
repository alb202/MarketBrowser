"""This is the utils module.

It contains classes for general functionality
"""
import datetime as dt
import logging

import market_time
import numpy as np
import pytz

MARKET_TZ = pytz.timezone(market_time.MARKET_TZ_CODE)
UTC_TZ = pytz.timezone(market_time.UTC_TZ_CODE)

DTYPES = {"symbol": str,
          "datetime": 'datetime64',
          "open": float,
          "high": float,
          "low": float,
          "close": float,
          "adjusted_close": float,
          "volume": np.int64,
          "dividend_amount": float,
          "split_coefficient": float}

DATA_COLUMNS1 = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
DATA_COLUMNS2 = ["symbol", "datetime", "open", "high", "low", "close", "adjusted_close",
                 "volume", "dividend_amount"]
DATA_COLUMNS3 = ["symbol", "datetime", "open", "high", "low", "close", "adjusted_close",
                 "volume", "dividend_amount", "split_coefficient"]


def set_column_dtypes(dataframe, dtypes):
    logging.info("Setting column dtypes: %s", str(dtypes))
    for column in dataframe.columns:
        dataframe = dataframe.astype({column: dtypes[column]})
    return dataframe


def format_datetime(old_date, date_format):
    return dt.datetime.strptime(old_date.strftime(date_format), date_format)


def get_current_time(date_format='%Y-%m-%d %H:%M:%S',
                     set_to_utc=True,
                     old_timezone=MARKET_TZ,
                     new_timezone=UTC_TZ):
    current_time = format_datetime(dt.datetime.now(), date_format)
    if set_to_utc:
        current_time = convert_between_timezones(current_time, old_timezone, new_timezone)
    return current_time


def convert_between_timezones(old_datetime, old_timezone, new_timezone):
    return old_timezone.localize(old_datetime).astimezone(new_timezone)


def make_query(symbol, function, interval=None):
    query_dict = {'symbol': symbol, 'function': function}
    if interval is not None:
        query_dict['interval'] = interval
    query = ' & '.join(["({} == '{}')".format(k, v) for k, v in query_dict.items()])
    print(query)
    return query


def make_sql(symbol, function, interval=None):
    sql = f"SELECT * FROM {function} WHERE symbol=='{symbol}';"
    print('Made sql:', sql)
    return sql


def validate_args(args):
    if ('INTRADAY' not in args['function']) & \
            (args['interval'] is not None) & \
            (args['interval'] != ''):
        logging.info('Only intraday function requires intervals! Exiting ... ')
        exit(2)
    elif (args['symbol'] is None) | (args['symbol'] == ''):
        logging.info('The symbol must be a string of at least 1 character! Exiting ... ')
        exit(2)
    else:
        return args
