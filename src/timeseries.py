# from alpha_vantage import alphavantage
import requests
import pandas as pd

API_KEY = r"5V11PBP7KPJDDNUP"
URL = r"https://www.alphavantage.co/query?" #function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo
RETURN_FORMAT = 'json'

class TimeSeries:

    def __init__(self, function=None, symbol=None, interval=None):
        self._function = function
        self._symbol = symbol
        self._interval = interval
        self._data = None

        return(None)

    def get_data(self, function, symbol, interval):
        parameters = {"function": function,
                      "symbol": symbol,
                      "interval": interval,
                      "apikey": API_KEY
                      }
        # get_url = f"{URL}function={function}&symbol={symbol}&interval={interval}&apikey={API_KEY}"
        result = requests.get(url=URL, params=parameters).json()
        self.meta_data = result["Meta Data"]
        self.data = pd.DataFrame.from_dict(result[list(filter(lambda x: x != "Meta Data", result.keys()))[0]]).transpose()

        # return(None)