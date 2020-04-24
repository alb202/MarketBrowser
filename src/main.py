"""This is the main module.

It is for the core application functionality
"""

import argparse
import logging

import utils
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
    # refresh_table = db_connect.load_data_table(table="DATA_STATUS",
    #                                            has_dt=True)

    # print(refresh_table)
    # print(refresh_table.dtypes)
    # query = TimeSeries(function=args['function'], symbol=args['symbol'], interval=args['interval'])
    query = TimeSeries()
    query.get_data(function=args['function'], symbol=args['symbol'], interval=args['interval'])
    print("query.has_data", query.has_data)
    # print("\n\n\n\n")
    query.process_meta_data()
    print(query.meta_data)
    print(query.meta_data.dtypes)

    query.process_data()
    print(query.data)
    print(query.data.dtypes)

    data_status.add_status(symbol=args['symbol'], function=args['function'], interval=args['interval'])
    print("data status table: ", data_status.data.dtypes)
    print("data status table: ", data_status.data)
    data_status.save_table()
    # print("Load the TIME_SERIES_INTRADAY table:")
    # a = db_connect.load_data("TIME_SERIES_INTRADAY")
    # print(a.dtypes)
    # print(a)
    # metadata_tbl = pd.DataFrame.from_sql()

    # print("query data", query.raw_data)
    # query.data.to_sql(name=args["function"],
    #                   con=db_connect.cur.connection,
    #                   if_exists='append',
    #                   index=False,
    #                   method=None)
    # db_connect.append_to_table(dataframe=query.data, table=args["function"])

    # print((query.meta_data))
    # print(query.data)
    # query.data.to_csv("test.csv", index=False, header=False)
    # print(db_connect.does_table_exist("TIME_SERIES_INTRADAY"))
    # print(db_connect.view_tables())
    # print(db_connect.view_table_info("TIME_SERIES_DAILY_ADJUSTED"))


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
    return utils.validate_args(args)


if __name__ == "__main__":
    main(parse_args())
