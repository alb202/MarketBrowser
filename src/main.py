"""This is the main module.

It is for the core application functionality
"""

import argparse
import logging

import market_time
import utilities
from data_status import DataStatus
from database import Database
from timeseries import TimeSeries

# logging.basicConfig(filemode='w', filename='../development.log', level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

DB_LOCATION = "../db/database.sqlite"
DB_SCHEMA = "../db/schema.sql"


def main(args):
    logging.info("Running...")
    db_connect = Database(db_location=DB_LOCATION, db_schema=DB_SCHEMA)
    data_status = DataStatus(db_connect)
    print("data status table: ", data_status.data)

    last_business_hours = market_time.LastBusinessHours()
    # last_business_hours = market_time.LastBusinessHours(
    #     testdate=[2020, 4, 24],
    #     testtime=[12, 25, 15])
    for i in last_business_hours.__dict__.items():
        print(i)

    print("Last update: ")
    is_market_open = data_status.get_last_update(
        symbol=args['symbol'],
        function=args['function'],
        interval=args['interval']) > last_business_hours.get_last_market_time()

    query = TimeSeries()
    query.get_data(
        function=args['function'],
        symbol=args['symbol'],
        interval=args['interval'])

    print("query.has_data", query.has_data)
    # print("\n\n\n\n")
    query.process_meta_data()
    print(query.meta_data)
    print(query.meta_data.dtypes)

    query.process_data()
    print(query.data)
    print(query.data.dtypes)

    data_status.add_status(
        symbol=args['symbol'],
        function=args['function'],
        interval=args['interval'])
    print("data status table: ", data_status.data.dtypes)
    print("data status table: ", data_status.data)
    data_status.save_table()

    # exit(0)

    # a = market_time.LastBusinessHours()
    # # a = market_time.LastBusinessHours(testdate=[2020, 4, 24], testtime=[12, 25, 15])
    # for i in a.__dict__.items():
    #     print(i)
    #
    # print("Last update: ")
    # print(data_status.get_last_update(
    #     symbol=args['symbol'],
    #     function=args['function'],
    #     interval=args['interval']) > a.get_last_market_time())

    # print(update_time.string_to_date(update_time.date_to_string.utcoffset(nyc_time.utcoffset(), DATE_FORMAT), DATE_FORMAT ))
    # print(dt.timedelta(nyc_time.utcoffset()))

    # print(dt.timedelta(utc_time, nyc_time))
    # print(pytz.timezone('UTC').localize(dt.datetime.now()), pytz.timezone('UTC').localize(dt.datetime.now()).tzinfo)
    #
    # print(pytz.timezone('America/New_York').localize(dt.datetime.now()),
    #       pytz.timezone('America/New_York').localize(dt.datetime.now()).tzinfo)
    #
    # print((pytz.timezone('UTC').localize(dt.datetime.now()) - pytz.timezone('America/New_York').localize(
    #     dt.datetime.now())))
    # d = a.get_last_market_day()
    # d =
    # print('is before', a.bef)
    # print(a.is_market_hours(a.get_testtime().time()))
    # print(a)


def parse_args():
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
    logging.info("Arguments: %s", str(args))
    return utilities.validate_args(args)


if __name__ == "__main__":
    main(parse_args())
