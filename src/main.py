from timeseries import TimeSeries
import pandas as pd
import requests
import argparse

def main(args):
    print("Running...")
    query = TimeSeries()
    query.get_data(function=args['function'], symbol=args['symbol'], interval=args['interval'] )
    #function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo
    # print(requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo").json())

    print((query.meta_data))
    print((query.data))



def parse_args():
    parser = argparse.ArgumentParser(description='Get stock or etf data ...')
    parser.add_argument('--function', metavar='FUNCTION', type=str, nargs=1, default="TIME_SERIES_DAILY", required=False,
                        help='Get the time series type')
    parser.add_argument('--symbol', metavar='SYMBOL', type=str, nargs=1, default="IBM", required=False,
                        help='Get the market symbol')
    parser.add_argument('--interval', metavar='INTERVAL', type=str, default=None, required=False,
                        choices=["5min", "15min", "30min", "60min"],
                        help='Get the time interval')
    args = parser.parse_args().__dict__
    return(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)


