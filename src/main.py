"""This is the main module.

It is for the core application functionality
"""

import argparse
import logging
from timeseries import TimeSeries
from database import Database

# logging.basicConfig(filemode='w', filename='../development.log', level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

DB_LOCATION = "../db/database.sqlite"
DB_SCHEMA = "../db/schema.sql"

def main(args):
    logging.info("Running...")
    db_connect = Database(db_location=DB_LOCATION, db_schema=DB_SCHEMA)

    query = TimeSeries(function=args['function'], symbol=args['symbol'], interval=args['interval'])
    query.get_data()
    # query.data.to_sql(name=args["function"],
    #                   con=db_connect.cur.connection,
    #                   if_exists='append',
    #                   index=False,
    #                   method=None)
    db_connect.append_to_table(dataframe=query.data, table=args["function"])

    print((query.meta_data))
    # print(query.data)
    # query.data.to_csv("test.csv", index=False, header=False)
    print(db_connect.does_table_exist("TIME_SERIES_INTRADAY"))
    print(db_connect.view_tables())
    print(db_connect.view_table_info("TIME_SERIES_DAILY_ADJUSTED"))

def parse_args():
    parser = argparse.ArgumentParser(description='Get stock or etf data ...')
    parser.add_argument('--function', metavar='FUNCTION', type=str, nargs='?',
                        default="TIME_SERIES_INTRADAY",
                        choices=["TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY",
                                 "TIME_SERIES_DAILY_ADJUSTED", "TIME_SERIES_WEEKLY",
                                 "TIME_SERIES_WEEKLY_ADJUSTED", "TIME_SERIES_MONTHLY",
                                 "TIME_SERIES_MONTHLY_ADJUSTED"],
                        required=False, help='Get the time series type')
    parser.add_argument('--symbol', metavar='SYMBOL', type=str, nargs='?', default="IBM",
                        required=False, help='Get the market symbol')
    parser.add_argument('--interval', metavar='INTERVAL', type=str, nargs='?', default="30min",
                        required=False, choices=["5min", "15min", "30min", "60min"],
                        help='Get the time interval')
    args = parser.parse_args().__dict__
    logging.info("Arguments: %s", str(args))
    return args


if __name__ == "__main__":
    main(parse_args())
