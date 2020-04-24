"""This is the utils module.

It contains classes for general functionality
"""

import logging
from datetime import datetime

import numpy as np

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


def format_datetime(dt, format):
    return datetime.strptime(dt.strftime(format), format)


def get_current_time(format='%Y-%m-%d %H:%M:%S'):
    return format_datetime(datetime.now(), format)


def make_query(symbol, function, interval=None):
    query_dict = {'symbol': symbol, 'function': function}
    if interval is not None:
        query_dict['interval'] = interval
    query = ' & '.join(["({} == '{}')".format(k, v) for k, v in query_dict.items()])
    print(query)
    return query


def validate_args(args):
    if (('INTRADAY' not in args['function']) and (args['interval'] is not None) and (args['interval'] != '')):
        logging.info('Only intraday function requires intervals! Exiting ... ')
        exit(2)
    elif ((args['symbol'] is None) or (args['symbol'] == '')):
        logging.info('The symbol must be a string of at least 1 character! Exiting ... ')
        exit(2)
    else:
        return (args)
