""" Timeseries module

This module controls the request of data from the server and formatting of timeseries data
"""

import datetime as dt
import sys
import pandas as pd
import requests
from src.alphavantage import AlphaVantage
from src.logger import *
from src.utilities import *

log = get_logger(__name__)


def convert_string_to_datetime(string, str_format):
    """Convert a string of values to datetime format
    """
    return dt.datetime.strptime(string, str_format)


def convert_datetime_to_string(old_datetime, str_format):
    """Convert a datetime object to string
    """
    return dt.datetime.strftime(old_datetime, str_format)


def check_date_format(old_datetime, convert=True):
    """Check the datetime format, and change if necessary
    """
    new_datetime = old_datetime.split(" ")
    if len(new_datetime) == 1:
        new_datetime = str(old_datetime) + " 16:00:00"
    else:
        new_datetime = old_datetime
    if convert:
        new_datetime = convert_string_to_datetime(new_datetime, utilities.DATETIME_FORMAT)
    return new_datetime


class TimeSeries():
    def __init__(self, con, symbol, function, interval=None):
        self.con = con
        self.params = dict(
            symbol=symbol,
            function=function,
            interval=interval)
        self.params['period'] = self.get_dividend_period()
        self.local_data = None
        self.remote_data = None

    def get_dividend_period(self):
        """Get the dividend period based on the function
        """
        if 'DAILY' in self.params['function']:
            return 'day'
        if 'WEEKLY' in self.params['function']:
            return 'week'
        if 'MONTHLY' in self.params['function']:
            return 'month'
        return None

    def get_local_data(self):
        """Get the data from the local database
        """
        self.local_data = self.DatabaseData(
            con=self.con, params=self.params)

    def get_remote_data(self, cfg):
        """Get the data from the API
        """
        self.remote_data = self.RemoteData(
            cfg=cfg,
            local_data={
                'prices': self.local_data.prices,
                'dividends': self.local_data.dividends},
            params=self.params)
        self.local_data.get_obsolete_data(
            new_dividends=self.remote_data.dividends,
            new_prices=self.remote_data.prices)
        self.local_data.remove_obsolete_data()
        self.delete_obsolete_data_from_database()
        if len(self.remote_data.prices) > 0:
            self.save_new_data(new_data=self.remote_data.prices, table=self.params['function'])
        if len(self.remote_data.dividends) > 0:
            self.save_new_data(new_data=self.remote_data.dividends, table='DIVIDEND')

    def delete_obsolete_data_from_database(self):
        """Delete obsolete price and dividend data from database
        """
        if len(self.local_data.obsolete_prices) > 0:
            self.delete_dataframe_from_database(
                self.local_data.obsolete_prices, table=self.params['function'])
        if self.local_data.incomplete_periods is not None:
            print('Removing incomplete periods from database: ', self.local_data.incomplete_periods)
            self.delete_dataframe_from_database(
                self.local_data.incomplete_periods, table=self.params['function'])
        if len(self.local_data.obsolete_dividends) > 0:
            self.delete_dataframe_from_database(
                self.local_data.obsolete_dividends, table='DIVIDEND')

    def delete_dataframe_from_database(self, df, table):
        """Delete a dataframe from the database using the current function table
        """
        records = []
        for index, row in df.iterrows():
            where = dict()
            where['symbol'] = row['symbol']
            where['datetime'] = row['datetime'].to_pydatetime()
            if 'interval' in df.columns:
                where['interval'] = row['interval']
            if 'period' in df.columns:
                where['period'] = row['period']
            # values = {'table': table,
            #           'where': where}
            records.append(where)
            log.info(f"Obsolete row: {str(where)}")
        log.info(f'{str(len(records))} records ready for deletion from table {table}')
        self.con.delete_records(records=records, table=table)

    def save_new_data(self, new_data, table):
        """Save the new data to the database table
        """
        log.info("Saving new data to database")
        self.con.update_table(dataframe=new_data,
                              table=table,
                              if_exists="append")

    class DatabaseData:
        def __init__(self, con, params):
            self.params = params
            self.prices = self.get_price_data_from_database(con=con)
            self.dividends = self.get_dividend_data_from_database(con=con)
            self.obsolete_prices = None
            self.obsolete_dividends = None
            self.incomplete_periods = None

        def get_price_data_from_database(self, con):
            """Retrieve data from database table
            """
            log.info("Getting data from database ...")
            query = {'table': self.params['function'], 'where': {'symbol': self.params['symbol']}}
            if self.params['interval'] is not None:
                query['where']['interval'] = self.params['interval']
            try:
                price_data = con.query_to_pandas(
                    where_dict=query, has_dt=True).sort_values('datetime').reset_index(drop=True)
                if isinstance(price_data, pd.DataFrame):
                    log.info("Object loaded with price data from database")
            except requests.ConnectionError as error:
                log.warn("Cannot get data from database. Exiting!")
                log.info(error)
                sys.exit()
            return price_data

        def get_dividend_data_from_database(self, con):
            """Retrieve data from database table
            """
            log.info("Getting dividend data from database ...")
            query = {'table': 'DIVIDEND',
                     'where': {'symbol': self.params['symbol'],
                               'period': self.params['period']}}
            try:
                dividend_data = con.query_to_pandas(
                    where_dict=query, has_dt=True).sort_values('datetime').reset_index(drop=True)
                if isinstance(dividend_data, pd.DataFrame):
                    log.info("Object loaded with price data from database")
            except requests.ConnectionError as error:
                log.warn("Cannot get dividend data from database. Exiting!")
                log.info(error)
                sys.exit()
            return dividend_data

        def get_obsolete_data(self, new_prices, new_dividends):
            self.obsolete_prices = self.get_obsolete_prices(new_data=new_prices) \
                if len(new_prices) > 0 else pd.DataFrame()
            self.obsolete_dividends = self.get_obsolete_dividends(new_data=new_dividends) \
                if len(new_dividends) > 0 else pd.DataFrame()
            if ('WEEK' in self.params['function']) or ('MONTH' in self.params['function']):
                self.incomplete_periods = self.get_incomplete_periods(new_data=new_prices) \
                    if len(new_prices) > 0 else pd.DataFrame()
                if len(self.incomplete_periods) > 0:
                    print('Removing the incomplete data from the self.prices')
                    self.prices = self.prices.merge(
                        self.incomplete_periods.loc[:, self.prices.columns],
                        how='left', indicator=True).query('_merge == "left_only"').drop(['_merge'], axis=1)
            print("Obsolete Data: ")
            print(self.obsolete_prices)
            print(self.obsolete_dividends)

        def get_obsolete_prices(self, new_data):
            """Remove price data already in database and delete obsolete rows
            """
            merge_cols = ['symbol', 'datetime']
            if self.params['interval'] is not None:
                merge_cols.append('interval')
            value_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'ADJUSTED' in self.params['function']:
                value_cols.append('adjusted_close')
            if 'DAILY_ADJUSTED' in self.params['function']:
                value_cols.append('split_coefficient')

            print("merge cols: ", merge_cols)
            print("value cols: ", value_cols)
            print("local cols: ", self.prices.columns)
            # print("new cols: ", new_data.columns)
            overlapping_data = self.overlapping_data_merge(
                old_data=self.prices, new_data=new_data, merge_cols=merge_cols)
            print("overlapping_data: ", overlapping_data.loc[:, sorted(overlapping_data.columns)])
            if len(overlapping_data) == 0:
                print("no overlapping data found")
                log.info("No overlapping rows found")
                return pd.DataFrame()
            query = ' | '.join(['(' + i + '_new' + ' != ' + i + ')' for i in value_cols])
            print(query)
            conflicting_data = overlapping_data.query(query)
            if len(conflicting_data) == 0:
                log.info("No rows with conflicting data")
                return pd.DataFrame()
            return conflicting_data.loc[:, sorted(value_cols + merge_cols)]

        def get_incomplete_periods(self, new_data):
            """Remove price data from partial weeks and months
            """
            cols = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            if 'ADJUSTED' in self.params['function']:
                cols.append('adjusted_close')
            if 'WEEK' in self.params['function']:
                period = 'week'
            if 'MONTH' in self.params['function']:
                period = 'month'

            print("value cols: ", cols)
            price_data = pd.concat([self.prices.loc[:, cols], new_data.loc[:, cols]])
            print('price_data: ', price_data.dtypes)
            price_data['day'] = price_data['datetime'].dt.day
            price_data['week'] = price_data['datetime'].dt.week
            price_data['month'] = price_data['datetime'].dt.month
            price_data['year'] = price_data['datetime'].dt.year
            price_data = price_data.sort_values(['year', 'datetime', 'month', 'week', 'day']).reset_index(drop=True)
            incomplete_price_index = price_data.duplicated(subset=['year', period], keep='last')
            incomplete_prices = price_data.loc[incomplete_price_index, :].reset_index(drop=True) \
                .drop(['year', 'month', 'week', 'day'], axis=1)
            print('incomplete prices - ', period, '-', incomplete_prices)

            return incomplete_prices

        def get_obsolete_dividends(self, new_data):
            """Remove price data already in database and delete obsolete rows
            """
            merge_cols = ['symbol', 'datetime', 'period']
            value_cols = ['dividend_amount']

            print("merge cols: ", merge_cols)
            print("value cols: ", value_cols)
            print("local cols: ", self.prices.columns)
            # print("new cols: ", new_data.columns)
            overlapping_data = self.overlapping_data_merge(
                old_data=self.dividends, new_data=new_data, merge_cols=merge_cols)
            print("overlapping_data: ", overlapping_data.loc[:, sorted(overlapping_data.columns)])
            if len(overlapping_data) == 0:
                print("no overlapping data found")
                log.info("No overlapping rows found")
                return pd.DataFrame()
            query = ' | '.join(['(' + i + '_new' + ' != ' + i + ')' for i in value_cols])
            print(query)
            conflicting_data = overlapping_data.query(query)
            if len(conflicting_data) == 0:
                log.info("No rows with conflicting data")
                return pd.DataFrame()
            return conflicting_data.loc[:, sorted(value_cols + merge_cols)]

        def remove_obsolete_data(self):
            if (len(self.prices) > 0) & (len(self.obsolete_prices) > 0):
                print("price columns: ", self.prices.columns)
                print("obsolete price columns: ", self.obsolete_prices.columns)
                print("prices: ", self.prices)
                print("obsolete prices: ", self.obsolete_prices)
                self.prices = self.prices.merge(
                    self.obsolete_prices,
                    how='left',
                    on=list(self.prices.columns),
                    indicator=True).query('_merge == "left_only"')

            if (len(self.dividends) > 0) & (len(self.obsolete_dividends) > 0):
                print("obsolete dividend columns: ", self.obsolete_dividends.columns)
                self.dividends = self.dividends.merge(
                    self.obsolete_dividends,
                    how='left',
                    on=list(self.dividends.columns),
                    indicator=True).query('_merge == "left_only"')

        @staticmethod
        def overlapping_data_merge(old_data, new_data, merge_cols):
            print("old data: ", old_data)
            print("new data: ", new_data)
            if (len(old_data) == 0) | (len(new_data) == 0):
                return pd.DataFrame()
            print("Getting overlapping data ......")
            return new_data.merge(
                old_data,
                how="outer",
                on=merge_cols,
                indicator=True,
                suffixes=['_new', '']
            ).query('_merge == "both"')

        def remove_obsolete_price_data(self, obsolete_price_data):
            self.prices = self.prices.merge(
                obsolete_price_data,
                how="left",
                on=self.prices.columns,
                indicator=True,
                suffixes=None) \
                .query('_merge == "left_only"') \
                .drop('_merge', axis=1)

        def remove_obsolete_dividend_data(self, obsolete_dividend_data):
            self.dividends = self.dividends.merge(
                obsolete_dividend_data,
                how="left",
                on=self.dividends.columns,
                indicator=True,
                suffixes=None) \
                .query('_merge == "left_only"') \
                .drop('_merge', axis=1)

    def view_data(self):
        """Return the price and dividend data
        """
        if self.remote_data is None:
            return dict(
                prices=self.local_data.prices,
                dividends=self.local_data.dividends)

        return dict(
            prices=pd.concat([self.local_data.prices,
                              self.remote_data.prices]) \
                .sort_values('datetime').reset_index(drop=True),
            dividends=pd.concat([self.local_data.dividends,
                                 self.remote_data.dividends]) \
                .sort_values('datetime').reset_index(drop=True))

    class RemoteData:
        """Manage time series data
        """
        def __init__(self, cfg, local_data, params):
            log.info("Creating TimeSeries object")

            av = AlphaVantage(keys=cfg.view_price_apikey())
            processed_data = av.get_data_from_api(
                symbol=params['symbol'],
                function=params['function'],
                interval=params['interval'],
                dividend_period=params['period'])

            self.prices = self.get_unique_data(
                old_data=local_data['prices'],
                new_data=processed_data['prices'],
                sort_col='datetime')
            self.dividends = self.get_unique_data(
                old_data=local_data['dividends'],
                new_data=processed_data['dividends'],
                sort_col='datetime')

        def get_data_from_server(self, cfg, symbol, function, interval):
            """Get fresh data from API
            """
            log.info('Getting data from server')


        @staticmethod
        def get_unique_data(old_data, new_data, sort_col):
            if new_data is None:
                return pd.DataFrame()
            if len(new_data) == 0:
                # print("newdata is none")
                return new_data
            return new_data \
                .merge(
                old_data,
                how="outer",
                on=list(new_data.columns),
                indicator=True) \
                .query('_merge == "left_only"') \
                .drop('_merge', axis=1) \
                .drop_duplicates(ignore_index=True) \
                .sort_values(sort_col) \
                .reset_index(drop=True)
