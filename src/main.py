"""This is the main module.

It is for the core application functionality
"""

import argparse

import logger
import market_time
import utilities
from config import Config
from data_status import DataStatus
from database import Database
from timeseries import TimeSeries

log = logger.get_logger(__name__)


def main(args):
    """Retrieve most up-to-date time series data for a single symbol/function
    """
    log.info('<<< Starting MarketBrowser >>>')
    log.info('<<< Loading config and connecting to database >>>')

    cfg = Config(args['config']) if args['config'] is not None \
        else Config("../resources/config.txt")
    db_connection = Database(cfg.view_db_location())
    db_connection.check_database()
    log.info('<<< Loading data status >>>')
    data_status = DataStatus(cfg)
    data_status.get_data_status(db_connection)
    if args['data_status']:
        # print('data status: ', data_status.data)
        return data_status.data
    log.info('<<< Checking market status >>>')
    market_time_info = market_time.MarketTime(cfg=cfg)
    last_business_hours = market_time.BusinessHours(market_time_info)
    last_market_time = last_business_hours.view_last_market_time()

    if args['get_all']:
        function_arg_list = ["TIME_SERIES_INTRADAY",
                             "TIME_SERIES_DAILY_ADJUSTED",
                             "TIME_SERIES_WEEKLY_ADJUSTED",
                             "TIME_SERIES_MONTHLY_ADJUSTED"]
        interval_arg_list = ["5min", "15min", "30min", "60min"]
    else:
        function_arg_list = args['function']
        interval_arg_list = args['interval']

    for symbol_arg in args['symbol']:
        for function_arg in function_arg_list:
            if function_arg == 'TIME_SERIES_INTRADAY':
                interval_arg_list_ = interval_arg_list
            else:
                interval_arg_list_ = [None]
            for interval_arg in interval_arg_list_:
                last_update = data_status.get_last_update(
                    symbol=symbol_arg,
                    function=function_arg,
                    interval=interval_arg)
                get_new_data = utilities.get_new_data_test(last_update,
                                                           last_market_time)
                log.info('<<< Getting timeseries data >>>')
                query = TimeSeries(con=db_connection,
                                   function=function_arg,
                                   symbol=symbol_arg,
                                   interval=interval_arg)
                query.get_local_data()
                if get_new_data:
                    log.info('<<< Getting most recent time series data >>>')
                    query.get_remote_data(cfg=cfg)

                    log.info('<<< Saving update information to database >>>')
                    if last_update is None:
                        data_status.add_status_entry(
                            symbol=symbol_arg,
                            function=function_arg,
                            interval=interval_arg)
                    else:
                        data_status.update_status_entry(
                            symbol=symbol_arg,
                            function=function_arg,
                            interval=interval_arg)
    log.info("<<< Saving data statuses to database >>>")
    data_status.save_table(database=db_connection)

    if args['no_return'] | \
            (not args['symbol']) | \
            (not args['function']) | \
            (not args['interval']):
        return True
    if query is not None:
        return query.view_data()


def parse_args():
    """Parse the command line arguments and return dictionary
    """
    log.info("Parsing input arguments")
    parser = argparse.ArgumentParser(description='Get stock or etf data ...')
    parser.add_argument('--function', metavar='FUNCTION', type=str, nargs='*',
                        default=None,
                        choices=["TIME_SERIES_INTRADAY",
                                 "TIME_SERIES_DAILY",
                                 "TIME_SERIES_DAILY_ADJUSTED",
                                 "TIME_SERIES_WEEKLY",
                                 "TIME_SERIES_WEEKLY_ADJUSTED",
                                 "TIME_SERIES_MONTHLY",
                                 "TIME_SERIES_MONTHLY_ADJUSTED"],
                        required=False, help='Get the time series type')
    parser.add_argument('--symbol', metavar='SYMBOL',
                        type=str, nargs='+', default="IBM",
                        required=False, help='Get the market symbol')
    parser.add_argument('--interval', metavar='INTERVAL',
                        type=str, nargs='*', default=None,
                        required=False,
                        choices=["5min", "15min", "30min", "60min"],
                        help='Get the time interval')
    parser.add_argument('--config', metavar='CONFIG', type=str,
                        nargs='?', default=None, required=False,
                        help='Path to config file')
    parser.add_argument('--get_all', action='store_true',
                        help='Get all the functions for symbol(s)')
    parser.add_argument('--no_return', action='store_true',
                        help='Do not return the data')
    parser.add_argument('--data_status', action='store_true',
                        help='View the data status table')
    args = parser.parse_args().__dict__
    log.info(f"Arguments: {str(args)}")
    return utilities.validate_args(args)


if __name__ == "__main__":
    main(parse_args())
