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

# import datetime as dt

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
    #
    # aaa = {'table': 'TIME_SERIES_MONTHLY',
    #        'where': {'symbol': 'MSFT',
    #                  # 'datetime': dt.datetime(year=2020, month=5, day=15, hour=0, minute=0, second=0).date(),
    #                  'open': 64.37#,
    #                  # 'high': 53.09,
    #                  # 'low': 52.53,
    #                  # 'close': 53.0,
    #                  # 'volume': 1269922,
    #                  # 'adjusted_close': 53.03,
    #                  # 'dividend_amount': 0.0
    #                  }}
    # DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    # DATE_FORMAT = '%Y-%m-%d'
    #
    # values = {
    #     'table': 'TIME_SERIES_INTRADAY',
    #     'where': {
    #         'symbol': 'TSLA',
    #         'interval': '5min',
    #         #
    #         'datetime': dt.datetime(year=2020, month=5, day=15, hour=15, minute=50, second=0)#.date()#.date()#.strftime(DATE_FORMAT)
    #         # 'datetime': '2020-05-15 00:00:00.000000'
    #         # 'low': 173.8,
    #         # 'open': 175.8,
    #     }}
    # print('values: ', values)
    # a = db_connection.delete_record(values=values)
    # # a = db_connection.get_record(values=values)
    # print('a', a)
    # import sqlalchemy as sa

    log.info('<<< Checking market status >>>')
    # log.info(f"Last market time: {last_business_hours.view_last_market_time()}")
    last_business_hours = market_time.LastBusinessHours(function=args['function'], cfg=cfg)
    last_market_time = last_business_hours.view_last_market_time()
    last_update = data_status.get_last_update(
        symbol=args['symbol'],
        function=args['function'],
        interval=args['interval'])
    get_new_data = utilities.get_new_data_test(last_update, last_market_time)

    log.info('<<< Getting timeseries data >>>')
    query = TimeSeries(cfg=cfg,
                       function=args['function'],
                       symbol=args['symbol'],
                       interval=args['interval'])
    query.get_data_from_database(con=db_connection, has_dt=True)

    if get_new_data:
        log.info('<<< Getting most recent time series data >>>')
        # query.remove_last_entry(delete_from_db=db_connection)
        query.get_data_from_server()
        # query.process_meta_data()
        query.process_data()

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
        # if (last_update > last_business_hours.view_last_complete_period()):
        #     print("Obsolete data needs to be deleted")
        # query.delete_obsolete_data(database=db_connection,
        #                            last_complete=last_business_hours.view_last_complete_period())
        query.save_new_data(database=db_connection)
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
