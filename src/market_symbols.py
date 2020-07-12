'''
Module for getting market symbols
'''
import urllib.request as ur

import logger
import pandas as pd

log = logger.get_logger(__name__)

class MarketSymbols:
    symbol_lists = dict(
        mutual_funds='ftp://ftp.nasdaqtrader.com/SymbolDirectory/mfundslist.txt',
        nasdaq_stocks='ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt',
        other_stocks='ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt')

    def __init__(self):
        self.con = None
        self.mutual_funds = self.get_mutual_funds()
        self.nasdaq_stocks = self.get_nasdaq_stocks()

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
        req = ur.Request(url)
        opened_url = ur.urlopen(req)
        return pd.read_table(
            filepath_or_buffer=opened_url,
            sep='|',
            header=0,
            usecols=cols,
            index_col=None).dropna(axis=0)

    def show_all(self):
        log.info('<<< Returning all stocks and mutual funds >>>')
        return pd.concat([self.mutual_funds, self.nasdaq_stocks]) \
            .sort_values("symbol") \
            .drop_duplicates() \
            .reset_index(drop=True)

# new_file = MarketSymbols(con=None)
# # print(new_file.mutual_funds)
# # print(new_file.nasdaq_stocks)
# print(new_file.show_all())
