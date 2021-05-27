'''
Module for getting market symbols
'''
import json
import ssl
import urllib.request as ur
from time import sleep
import pandas as pd
import requests
from http.client import RemoteDisconnected
from requests.exceptions import *
from urllib3.exceptions import *

from .logger import *
from .utilities import *

log = get_logger(__name__)

class MarketSymbols:
    FINANCIALS_TABLE = 'FINANCIALS'
    DATATYPES = {'symbol': str, 'name': str, 'type': str,
                 'sector': str, 'industry': str, 'marketCap': float,
                 'sharesOutstanding': float, 'float': float}
    SYMBOL_URLS = dict(
        mutual_funds='ftp://ftp.nasdaqtrader.com/SymbolDirectory/mfundslist.txt',
        nasdaq_stocks='ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt',
        other_stocks='ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt')

    def __init__(self, con, cfg, refresh=False):
        self.con = con
        self.stocks = convert_df_dtypes(
            df=pd.read_sql_table(
                table_name=self.FINANCIALS_TABLE,
                con=self.con.engine,
                index_col=None),
            dtypes=self.DATATYPES)

        if refresh | (len(self.stocks) == 0):
            # self.mutual_funds = self.get_mutual_funds(self.SYMBOL_URLS['mutual_funds'])
            self.nasdaq_stocks = self.get_nasdaq_stocks(
                self.SYMBOL_URLS['nasdaq_stocks'])
            self.stock_info = self.get_stock_info(cfg=cfg, symbols=self.nasdaq_stocks['symbol'].values)
            all_stocks = self.nasdaq_stocks.merge(self.stock_info, how='left', on='symbol')
            self.stocks = convert_df_dtypes(
                df=all_stocks
                    .sort_values("symbol")
                    .drop_duplicates()
                    .reset_index(drop=True),
                dtypes=self.DATATYPES)
            self.save_table()

    def get_mutual_funds(self):
        log.info('<<< Getting list of mutual funds >>>')
        cols = ['Fund Symbol', 'Fund Name', 'Fund Family Name']
        url = self.SYMBOL_URLS['mutual_funds']
        data = self.pull_data(url=url, cols=cols)
        data = data[data['Fund Family Name'] != 'NASDAQ Test Funds']
        data.loc[:, 'type'] = 'Mutual fund'
        data = data.loc[:, ['Fund Symbol', 'Fund Name', 'type']]
        data = data.rename(columns={'Fund Symbol': 'symbol', 'Fund Name': 'name'})
        return data.drop_duplicates().sort_values('symbol').reset_index(drop=True)

    def get_nasdaq_stocks(self, url, cols=['Symbol', 'Security Name', 'ETF']):
        log.info('<<< Getting list of stocks and etfs >>>')
        data = self.pull_data(url=url, cols=cols)
        data.loc[data['ETF'] == 'Y', 'type'] = 'etf'
        data.loc[data['ETF'] == 'N', 'type'] = 'stock'
        data.loc[data['ETF'] == '', 'type'] = 'other'
        data = data[data['ETF'] != ' ']
        data = data.drop('ETF', axis=True)
        data = data.rename(columns={'Symbol': 'symbol', 'Security Name': 'name'})
        return data.drop_duplicates().sort_values('symbol').reset_index(drop=True)

    @staticmethod
    def pull_data(url, cols):
        while True:
            try:
                req = ur.Request(url)
                opened_url = ur.urlopen(req, timeout=20)
            except socket.timeout:
                log.info("timeout error")
            except socket.error:
                log.info("socket error occurred: ")
            else:
                log.info('Access successful.')
                break

        return pd.read_table(
            filepath_or_buffer=opened_url,
            sep='|',
            header=0,
            usecols=cols,
            index_col=None).dropna(axis=0)

    def show_all(self):
        log.info('<<< Returning all stocks and etfs >>>')
        return self.stocks

    def save_table(self):
        """Save the data status dataframe to a table
        """
        log.info("Saving data status table to datebase")
        self.con.update_table(self.show_all(), self.FINANCIALS_TABLE, "replace")

    def get_stock_info(self, cfg, symbols):
        error_messages = ['Unknown symbol',
                          'You have exceeded your allotted message quota. Please enable pay-as-you-go to regain access',
                          'forbidden',
                          'The API key provided is not valid.',
                          'An API key is required to access this data and no key was provided']
        stock_responses = []
        symbols = [i for i in symbols if ('$' not in i) and ('.' not in i)]
        timeout = (30, 45)
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=80)
        sess.mount('http://', adapter)

        for symbol in symbols:
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=80)
            sess.mount('http://', adapter)

            print(f'Symbol: {symbol}')
            attempt = 0
            # print('Symbol: ', symbol)
            url = cfg.view_stock_url()
            params = {'token': cfg.view_stock_apikey()}
            while True:
                is_error = False
                print('Attempt: ', attempt)
                try:
                    sector_response = sess.get(url + f'stock/{symbol}/company/', params=params, timeout=timeout)
                    if sector_response.text in error_messages:
                        is_error = True
                    else:
                        shares_response = sess.get(url + f'stock/{symbol}/stats/sharesOutstanding/',
                                                   params=params,
                                                   timeout=timeout)
                        if shares_response.text == 'Not found':
                            marketcap_response = None
                            float_response = None
                        else:
                            marketcap_response = sess.get(url + f'stock/{symbol}/stats/marketcap/',
                                                          params=params,
                                                          timeout=timeout)
                            float_response = sess.get(url + f'stock/{symbol}/stats/float/',
                                                      params=params,
                                                      timeout=timeout)

                except (requests.exceptions.RequestException,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.HTTPError,
                        ConnectionResetError,
                        NameError,
                        ssl.SSLError,
                        ssl.SSLEOFError,
                        requests.exceptions.Timeout,
                        requests.exceptions.SSLError,
                        requests.exceptions.ConnectTimeout,
                        ProtocolError, ReadTimeout,
                        RemoteDisconnected,
                        Exception) as error:
                    log.info("An attempt to get data failed!")
                    log.warn(error)
                    is_error = True
                else:
                    log.info('Data retrieval successful.')
                if not is_error:
                    try:
                        sector_data = sector_response.json()
                        sector_data = {key: val for key, val in sector_data.items() if key in ['industry', 'sector']}
                    except (json.decoder.JSONDecodeError, NameError, Exception) as error:
                        log.info(error)
                        sector_data = {'industry': None, 'sector': None}

                    try:
                        marketcap_data = {'marketCap': marketcap_response.text}
                    except (NameError, ValueError, AttributeError, Exception) as error:
                        log.info(error)
                        marketcap_data = {'marketCap': None}

                    try:
                        float_data = {'float': float_response.text}
                    except (NameError, ValueError, AttributeError, Exception) as error:
                        log.info(error)
                        float_data = {'float': None}

                    try:
                        shares_data = {'sharesOutstanding': shares_response.text}
                    except (NameError, ValueError, AttributeError, Exception) as error:
                        log.info(error)
                        shares_data = {'sharesOutstanding': None}

                    stock_responses.append(
                        {'symbol': symbol, **sector_data, **shares_data, **float_data, **marketcap_data})
                    sleep(1)
                    break
                if attempt > 10:
                    break
                sleep(1)
                attempt += 1
        print('Finished getting responses')
        return pd.DataFrame(stock_responses) \
            .sort_values('symbol') \
            .drop_duplicates() \
            .replace({'Not found': None})
        # df.to_csv('stock_output.tsv', index=False, sep='\t')
        # print(df)
        # return(df)
