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
    cfg = Config("../resources/config.txt")
    db_connection = Database(cfg.view_db_location())
    db_connection.check_database()
    log.info('<<< Loading data status >>>')
    data_status = DataStatus(cfg)
    data_status.get_data_status(db_connection)
    log.info('<<< Checking market status >>>')
    market_time_info = market_time.MarketTime(cfg=cfg)
    last_business_hours = market_time.BusinessHours(market_time_info)
    last_market_time = last_business_hours.view_last_market_time()
    print(last_market_time)
    last_update = data_status.get_last_update(
        symbol=args['symbol'],
        function=args['function'],
        interval=args['interval'])
    get_new_data = utilities.get_new_data_test(last_update, last_market_time)

    log.info('<<< Getting timeseries data >>>')
    query = TimeSeries(con=db_connection,
                       function=args['function'],
                       symbol=args['symbol'],
                       interval=args['interval'])
    query.get_local_data()

    if get_new_data:
        log.info('<<< Getting most recent time series data >>>')
        # query.get_data_from_server()
        # query.process_data()
        query.get_remote_data(cfg=cfg)

        log.info('<<< Saving update information to database >>>')
        if last_update is None:
            data_status.add_status_entry(
                symbol=args['symbol'],
                function=args['function'],
                interval=args['interval'])
        else:
            data_status.update_status_entry(
                symbol=args['symbol'],
                function=args['function'],
                interval=args['interval'])
        log.info("<<< Saving new data to database >>>")

        # query.remove_nonunique_rows(database=db_connection)
        # query.remove_nonunique_dividend_rows(database=db_connection)
        # query.save_new_data(database=db_connection)
        data_status.save_table(database=db_connection)

    return query.view_data()


def parse_args():
    """Parse the command line arguments and return dictionary
    """
    log.info("Parsing input arguments")
    parser = argparse.ArgumentParser(description='Get stock or etf data ...')
    parser.add_argument('--function', metavar='FUNCTION', type=str, nargs='?',
                        default="TIME_SERIES_DAILY",
                        choices=["TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY",
                                 "TIME_SERIES_DAILY_ADJUSTED", "TIME_SERIES_WEEKLY",
                                 "TIME_SERIES_WEEKLY_ADJUSTED", "TIME_SERIES_MONTHLY",
                                 "TIME_SERIES_MONTHLY_ADJUSTED"],
                        required=False, help='Get the time series type')
    parser.add_argument('--symbol', metavar='SYMBOL', type=str, nargs='?', default="IBM",
                        required=False, help='Get the market symbol')
    parser.add_argument('--interval', metavar='INTERVAL', type=str, nargs='?', default=None,
                        required=False, choices=["5min", "15min", "30min", "60min"],
                        help='Get the time interval')
    args = parser.parse_args().__dict__
    log.info(f"Arguments: {str(args)}")
    return utilities.validate_args(args)


if __name__ == "__main__":
    main(parse_args())
