'''
Module for getting market symbols
'''
import json
import ssl
import urllib.request as ur
from time import sleep

import logger
import numpy as np
import pandas as pd
import requests

log = logger.get_logger(__name__)


class MarketSymbols:
    financials_table_name = 'FINANCIALS'
    dtypes = {'symbol': str, 'marketCap': np.int64,
              'sharesOutstanding': np.int64,  # 'price': float,
              'float': np.int64, 'peRatio': float, 'beta': float}
    symbol_lists = dict(
        mutual_funds='ftp://ftp.nasdaqtrader.com/SymbolDirectory/mfundslist.txt',
        nasdaq_stocks='ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt',
        other_stocks='ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt')

    def __init__(self, con, cfg, refresh=False):
        self.con = con
        self.stocks = pd.read_sql_table(
            table_name=self.financials_table_name,
            con=self.con.engine,
            index_col=None)
        if refresh | (len(self.stocks) == 0):
            self.mutual_funds = None  # self.get_mutual_funds()
            self.nasdaq_stocks = self.get_nasdaq_stocks()
            self.stock_info = self.get_stock_info(cfg=cfg,
                                                  symbols=self.nasdaq_stocks['symbol'].values)
            all_stocks = self.nasdaq_stocks.merge(self.stock_info, how='left', on='symbol')
            self.stocks = all_stocks \
                .sort_values("symbol") \
                .drop_duplicates() \
                .reset_index(drop=True)  # \
            # .astype(self.dtypes)
            self.save_table()

    def get_mutual_funds(self):
        log.info('<<< Getting list of mutual funds >>>')
        cols = ['Fund Symbol', 'Fund Name', 'Fund Family Name']
        url = self.symbol_lists['mutual_funds']
        data = self.pull_data(url=url, cols=cols)
        data = data[data['Fund Family Name'] != 'NASDAQ Test Funds']
        data.loc[:, 'type'] = 'Mutual fund'
        data = data.loc[:, ['Fund Symbol', 'Fund Name', 'type']]
        data = data.rename(columns={'Fund Symbol': 'symbol', 'Fund Name': 'name'})
        return data.drop_duplicates().sort_values('symbol').reset_index(drop=True)

    def get_nasdaq_stocks(self):
        log.info('<<< Getting list of stocks and etfs >>>')
        cols = ['Symbol', 'Security Name', 'ETF']
        url = self.symbol_lists['nasdaq_stocks']
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
        self.con.update_table(self.show_all(), self.financials_table_name, "replace")

    def get_stock_info(self, cfg, symbols):
        error_messages = ['Unknown symbol',
                          'You have exceeded your allotted message quota. Please enable pay-as-you-go to regain access']
        stock_responses = []
        symbols = [i for i in symbols if ('$' not in i) and ('.' not in i)]
        timeout = (15, 25)
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=20)
        sess.mount('http://', adapter)

        for symbol in symbols:
            attempt = 0
            # print('Symbol: ', symbol)
            url = cfg.view_stock_url()
            params = {'token': cfg.view_stock_apikey()}
            while True:
                is_error = False
                print('Attempt: ', attempt)
                try:
                    sector_response = sess.get(url + f'stock/{symbol}/company/',
                                               params=params,
                                               timeout=timeout)
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
                        ssl.SSLError, ssl.SSLEOFError,
                        requests.exceptions.Timeout,
                        requests.exceptions.SSLError,
                        requests.exceptions.ConnectTimeout,
                        socket.timeout,
                        requests.exceptions.ReadTimeout) as error:
                    log.info("An attempt to get data failed!")
                    log.warn(error)
                    is_error = True
                else:
                    log.info('Data retrieval successful.')
                if not is_error:
                    try:
                        sector_data = sector_response.json()
                        sector_data = {key: val for key, val in sector_data.items() if key in ['industry', 'sector']}
                    except (json.decoder.JSONDecodeError) as error:
                        log.info(error)
                        sector_data = {'industry': None, 'sector': None}

                    try:
                        marketcap_data = {'marketCap': marketcap_response.text}
                    except (NameError, ValueError, AttributeError) as error:
                        log.info(error)
                        marketcap_data = {'marketCap': None}

                    try:
                        float_data = {'float': float_response.text}
                    except (NameError, ValueError, AttributeError) as error:
                        log.info(error)
                        float_data = {'float': None}

                    try:
                        shares_data = {'outstandingShares': shares_response.text}
                    except (NameError, ValueError, AttributeError) as error:
                        log.info(error)
                        shares_data = {'outstandingShares': None}

                    stock_responses.append(
                        {'symbol': symbol, **sector_data, **shares_data, **float_data, **marketcap_data})
                    break
                if attempt > 10:
                    break
                sleep(1)
                attempt += 1

        return pd.DataFrame(stock_responses) \
            .sort_values('symbol') \
            .drop_duplicates() \
            .replace({'Not found': None})
        # df.to_csv('stock_output.tsv', index=False, sep='\t')
        # print(df)
        # return(df)
