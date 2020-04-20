"""This is the utils module.

It contains classes for general functionality
"""

import logging

DTYPES = {"symbol": str,
          "datetime": str,
          "open": float,
          "high": float,
          "low": float,
          "close": float,
          "adjusted_close": float,
          "volume": int,
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
        dataframe = dataframe.astype({column:dtypes[column]})
    return dataframe
