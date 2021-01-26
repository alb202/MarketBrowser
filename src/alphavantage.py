import random
import re
from itertools import cycle

import numpy as np
import pandas as pd
import requests

from utilities import *

log = logger.get_logger(__name__)


class AlphaVantage:
    AV_URL = 'https://www.alphavantage.co:443/query?'
    ERR_KEY = 'Error Message'
    NOTE_KEY = 'Note'
    INFO_KEY = 'Information'
    # PROXY_URL = 'https://www.us-proxy.org'
    # PROXY_URL = 'https://free-proxy-list.net/'
    PROXY_URL = 'https://api.proxyscrape.com/?request=getproxies&proxytype=http&timeout=250&country=all&ssl=all&anonymity=all&status=alive'
    DTYPES = {"symbol": str
        , "datetime": 'datetime64'
        , "open": float
        , "high": float
        , "low": float
        , "close": float
        , "adjusted_close": float
        , "volume": np.int64
        , "interval": str
        , "dividend_amount": float
        , "split_coefficient": float}

    def __init__(self, keys, max_retries=10, retry_sleep=0, use_proxy=False):
        self._keys = keys
        self.session = requests.Session()
        self.max_retries = max_retries
        self._sleep = retry_sleep
        self.use_proxy = use_proxy
        http_adapter = requests.adapters.HTTPAdapter()
        self.session.mount('http://', http_adapter)
        self.session.mount('https://', http_adapter)
        self.raw_data = None
        self.processed_data = None
        self.proxies = self.get_proxy_list()
        print('proxies: ', len(self.proxies), self.proxies)

    def get_proxy_list(self):
        ip_address = r'^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]):[0-9]+$'
        tries = 0
        proxies = []
        while (tries < 10) & (len(proxies) == 0):
            while True:
                try:
                    response = requests.get(self.PROXY_URL)
                except requests.exceptions.ConnectionError as e:
                    log.warn('Error getting proxies', e)
                else:
                    break
            for line in response.iter_lines():
                if re.match(ip_address, line.decode("utf-8")):
                    proxies.append(line.decode("utf-8"))
        return proxies

    @staticmethod
    def format_column_names(names):
        """Format column names for pandas
        """
        log.info("Formatting column names")
        return [i.split(".")[1].replace(" ", "_")[1:] for i in names]

    def get_data_from_api(self, symbol, function, interval, dividend_period=None):
        raw_data = self.call_server(
            symbol=symbol,
            function=function,
            interval=interval)
        if raw_data is None:
            log.warning("No raw data to process")
            return {'prices': pd.DataFrame(), 'dividends': pd.DataFrame()}
        return self.process_data(
            raw_data=raw_data,
            symbol=symbol,
            interval=interval,
            dividend_period=dividend_period)

    def process_data(self, raw_data, symbol, interval, dividend_period):
        """Convert the raw JSON time series data into pandas dataframe
        """
        log.info("Processing raw data from API")
        tries = 0
        while True:
            try:
                price_data = pd.DataFrame.from_dict(
                    raw_data[list(
                        filter(lambda x: x != "Meta Data",
                               raw_data.keys()))[0]]).transpose()
            except (IndexError, TypeError) as e:
                log.warn("Error, raw data not accessed: ", e)
            else:
                break
            if tries > 4:
                break
            tries += 1

        price_data.columns = self.format_column_names(price_data.columns)
        price_data.reset_index(drop=False, inplace=True)
        price_data.rename(columns={"index": "datetime"}, inplace=True)
        price_data.loc[:, 'symbol'] = symbol
        if interval is not None:
            price_data.loc[:, 'interval'] = interval
        price_data = convert_df_dtypes(
            df=price_data,
            dtypes=self.DTYPES)
        if 'dividend_amount' in price_data.columns:
            dividend_data = price_data.loc[:, ['symbol', 'datetime', 'dividend_amount']]
            dividend_data = dividend_data.loc[dividend_data["dividend_amount"] > 0]
            if len(dividend_data) > 0:
                dividend_data.loc[:, 'period'] = dividend_period
                dividend_data = dividend_data. \
                    drop_duplicates(). \
                    sort_values('datetime'). \
                    reset_index(drop=True)
            price_data = price_data.drop('dividend_amount', axis=1)
        else:
            dividend_data = None
        price_data = price_data[price_data['datetime'].map(
            pd.tseries.offsets.BDay().is_on_offset)]
        log.info(f"Time series data created with columns: {str(list(price_data.columns))}")
        return {'prices': price_data, 'dividends': dividend_data}

    def call_server(self, symbol, function, interval):
        """Get fresh data from API
        """
        log.info('Getting data from server')
        api_parameters = {"function": function,
                          "symbol": symbol,
                          "interval": interval,
                          "outputsize": "full",
                          "datatype": "json"}
        print(api_parameters)
        tries = 0
        proxy_pool = cycle(self.proxies)
        api_keys = cycle(self._keys)

        while True:
            if (tries == 0) | (not self.use_proxy):
                proxies = None
            else:
                if tries > 20:
                    proxy_pool = cycle(self.get_proxy_list())
                    tries = 0
                try:
                    next_proxy = next(proxy_pool)
                except StopIteration as e:
                    print(e)
                else:
                    next_proxy = random.choice(self.get_proxy_list())
                proxies = {'http': next_proxy, 'https': next_proxy}
            api_parameters['apikey'] = next(api_keys)
            print('Using key: ', api_parameters['apikey'])
            print('Using proxy: ', proxies)
            try:
                response = self.session.get(
                    url=self.AV_URL, params=api_parameters,
                    proxies=proxies, timeout=(5, 5))
            except requests.RequestException as e:
                log.info("Data grab failed. Exiting!")
                log.warn(e)
            else:
                print('status_code:', response.status_code)

                if response.status_code == 200:
                    log.info("Object loaded with raw data")
                    raw_data = response.json()
                    if self.ERR_KEY in raw_data.keys():
                        msg = raw_data[self.ERR_KEY]
                        print('ERR_KEY is in json')
                        print(msg)
                    elif self.NOTE_KEY in raw_data.keys():
                        msg = raw_data[self.NOTE_KEY]
                        print('NOTE_KEY is in json')
                        print(msg)
                    elif self.INFO_KEY in raw_data.keys():
                        msg = raw_data[self.INFO_KEY]
                        print('INFO KEY is in json')
                        print(msg)
                    else:
                        log.info(f"API call successful: {symbol} {function} {interval} ...")
                        return response.json()
                    log.warn(f"API call error: {msg}. Trying again ...")
                if tries >= self.max_retries:
                    return None
                tries += 1

#
#
# av = AlphaVantage(keys=['5V11PBP7KPJDDNUP', 'LJLPYWJ9BW6A16TV', 'BCBU9ER8HOV78Z02'])
# new_data = []
# for i in ['SPY', 'AMAT', 'SPYG', 'REGN', 'USRT', 'VNQ', 'PLAY', 'QQQ', 'LUV']:
#     data = av.get_data_from_api(
#         symbol=i,
#         function='TIME_SERIES_INTRADAY',
#         interval='30min',
#         # function='TIME_SERIES_MONTHLY_ADJUSTED',
#         # interval=None,
#         dividend_period='month')
#     print(data['prices'].head(5))
