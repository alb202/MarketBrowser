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
    db_connect = Database(
        db_location=DB_LOCATION,
        db_schema=DB_SCHEMA)

    data_status = DataStatus(db_connect)
    print("data status table: ", data_status.data)

    last_business_hours = market_time.LastBusinessHours(function=args['function'])
    # last_business_hours = market_time.LastBusinessHours(
    #     function=args['function'],
    #     testdate=[2020, 3, 24],
    #     testtime=[12, 25, 15])
    for i in last_business_hours.__dict__.items():
        print(i)

    # print("Last update: ")
    last_update = data_status.get_last_update(
        symbol=args['symbol'],
        function=args['function'],
        interval=args['interval'])

    print("Last update: ", last_update)

    last_market_time = last_business_hours.get_last_market_time()

    if last_update is None:
        get_new_data = True
    elif last_update < last_market_time:
        get_new_data = True
    else:
        get_new_data = False
    print(get_new_data)

    query = TimeSeries()
    if not get_new_data:
        query.get_data_from_database(
            con=db_connect,
            has_dt=True,
            function=args['function'],
            symbol=args['symbol'],
            interval=args['interval'])

    if get_new_data:
        query.get_data_from_server(
            function=args['function'],
            symbol=args['symbol'],
            interval=args['interval'])

        print("query.has_data", query.has_data)
        query.process_meta_data()
        query.process_data()

        print(query.meta_data)
        print(query.meta_data.dtypes)

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
        print("data status table: ", data_status.data.dtypes)
        print("data status table: ", data_status.data)
        data_status.save_table()

    print(query.data)
    print(query.data.dtypes)
    db_connect.append_to_table(dataframe=query.data, table=args["function"])

    db_connect.__del__()

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
