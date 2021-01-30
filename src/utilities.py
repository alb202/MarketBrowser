"""This is the utils module.

It contains classes for general functionality
"""
import datetime as dt
import math
import sys

import logger

log = logger.get_logger(__name__)

DATA_COLUMNS1 = ["symbol", "datetime", "open", "high", "low", "close", "volume", "interval"]
DATA_COLUMNS2 = ["symbol", "datetime", "open", "high", "low", "close", "adjusted_close",
                 "volume", "dividend_amount"]
DATA_COLUMNS3 = ["symbol", "datetime", "open", "high", "low", "close", "adjusted_close",
                 "volume", "dividend_amount", "split_coefficient"]

DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def format_datetime(old_date, date_format):
    """Format a datetime object
    """
    log.info("Formatting datetime")
    return dt.datetime.strptime(old_date.strftime(date_format), date_format)


def get_current_time(date_format='%Y-%m-%d %H:%M:%S', old_timezone=None,
                     new_timezone=None, set_to_utc=True):
    """Get the current datetime
    """
    log.info("Getting the current datetime")
    current_time = format_datetime(dt.datetime.now(), date_format)
    if set_to_utc:
        current_time = convert_between_timezones(current_time, old_timezone, new_timezone)
    log.info(f"Current datetime: {str(current_time)}")
    return current_time


def convert_between_timezones(old_datetime, old_timezone, new_timezone):
    """Convert datetime between two timezones
    """
    return old_timezone.localize(old_datetime).astimezone(new_timezone)


def make_pandas_query(symbol, function, interval=None):
    """Make a pandas dataframe query for a symbol/function/interval
    """
    log.info("Make pandas query")
    query_dict = {'symbol': symbol, 'function': function}
    if interval is not None:
        query_dict['interval'] = interval
    return ' & '.join(["({} == '{}')".format(k, v) for k, v in query_dict.items()])


def validate_args(args):
    """Validate command line arguments
    """
    log.info("Validate the command line arguments")
    if args['function']:
        if (('TIME_SERIES_INTRADAY' not in args['function']) & \
            (args['interval'] is not None)) & (not args['get_all']):
            log.info('Only the intraday function requires intervals! Exiting ... ')
            sys.exit()
    if (not args['symbol']):
        log.info('At least one symbol must be provided! Exiting ... ')
        sys.exit()
    if (args['force_update'] & args['no_api']):
        log.info('Cannot use --force_update and --no_api flags! Exiting ... ')
        sys.exit()
    if (not args['function']) \
            & (not args['get_all']) \
            & (not args['data_status']) \
            & (not args['get_symbols']):
        log.info('At least one function or --get_all must be requested! Exiting ... ')
        sys.exit()
    if args['symbol']:
        try:
            args['symbol'] = [i.upper().strip(' ') for i in args['symbol']]
        except Exception as e:
            log.warn('Error formatting symbols! Exiting ... ')
            sys.exit()
    return args


def time_series_column_order(columns):
    """Order the columns in a pandas dataframe for time series data
    """
    log.info("Set the column order")
    if "split_coefficient" in columns:
        return DATA_COLUMNS3
    if "adjusted_close" in columns:
        return DATA_COLUMNS2
    return DATA_COLUMNS1


def get_new_data_test(dt1, dt2):
    """Determine if new data should be retrieved from api
    """
    log.info(f"Compare datetimes: {dt1} {dt2}")
    if dt1 is None:
        return True
    if dt1 < dt2:
        return True
    return False


def round_down(x, base=5):
    """Round a number down to the nearest multiple of <base>
    """
    return math.floor(x / base) * base


def convert_df_dtypes(df, dtypes):
    if type(dtypes) != dict:
        print('Need a dict of dtypes! No changes made')
        return df
    for key, value in dtypes.items():
        print('key', key, 'value', value)
        try:
            df[key] = df[key].values.astype(value, copy=True)
        except (KeyError, ValueError) as e:
            print(e)
    return df
