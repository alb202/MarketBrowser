from connect import TimeSeries
import pandas as pd
import requests


if __name__ == "__main__":
    print("Running...")
    # print(help(TimeSeries))
    #
    a = TimeSeries()
    a.get_data(
        function="TIME_SERIES_INTRADAY",
        symbol="IBM",
        interval="5min",
    )
    #function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo
    print((a.meta_data))
    print((a.data))
    # print(requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo").json())

