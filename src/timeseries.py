""" Timeseries module

This module controls the request of data from the server and formatting of timeseries data
"""

import datetime as dt
import logging

import pandas as pd
import requests
import utilities
from pandas.tseries.offsets import BDay

# API_KEY = "5V11PBP7KPJDDNUP"
# URL = "https://www.alphavantage.co/query?"
# RETURN_FORMAT = "json"

DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def convert_string_to_datetime(string, str_format):
    return dt.datetime.strptime(string, str_format)


def convert_datetime_to_string(old_datetime, str_format):
    return dt.datetime.strftime(old_datetime, str_format)


def check_date_format(old_datetime, convert=True):
    new_datetime = old_datetime.split(" ")
    if len(new_datetime) == 1:
        new_datetime = str(old_datetime) + " 16:00:00"
    else:
        new_datetime = old_datetime
    if convert:
        new_datetime = convert_string_to_datetime(new_datetime, DATETIME_FORMAT)
    return new_datetime


def format_column_names(names):
    logging.info("Formatting column names ...")
    return [i.split(".")[1].replace(" ", "_")[1:] for i in names]


class TimeSeries:

    def __init__(self, cfg):
        logging.info("Creating TimeSeries object ...")
        self.function = None
        self.symbol = None
        self.interval = None
        self.raw_data = None
        self.new_data = None
        self.db_data = None
        self.meta_data = None
        self.has_data = False
        self.api_cfg = cfg
        self.isBusinessDay = BDay().onOffset

    def get_data_from_database(self, con, function, symbol, has_dt=False, interval=None):
        self.function = function
        self.symbol = symbol
        self.interval = interval
        logging.info("Getting data from database ...")
        where_dict = {'table': function, 'where': {'symbol': symbol}}
        if interval is not None:
            where_dict['where']['interval'] = interval
        try:
            self.db_data = con.query_to_pandas(
                where_dict={'table': function, 'where': {'symbol': symbol}}, has_dt=has_dt)
        except requests.ConnectionError as error:
            logging.debug("Cannot get data from database!")
            logging.info(error)
            exit(2)
        finally:
            if isinstance(self.db_data, pd.DataFrame):
                self.has_data = True
                logging.info("Object loaded with data from database")

    def get_data_from_server(self, function, symbol, interval=None):
        self.function = function
        self.symbol = symbol
        self.interval = interval

        parameters = {"function": self.function,
                      "symbol": self.symbol,
                      "interval": self.interval,
                      "outputsize": self.api_cfg.view_outputsize(),
                      "apikey": self.api_cfg.view_apikey(),
                      "datatype": self.api_cfg.view_format()
                      }

        logging.info("Getting data with parameters: %s", str(parameters))
        try:
            self.raw_data = requests.get(
                url=self.api_cfg.view_url(),
                params=parameters).json()
        except requests.RequestException as error:
            logging.debug("Data grab failed!")
            logging.info(error)
            exit(2)
        else:
            if isinstance(self.raw_data, dict):
                # print("setting has_data to TRUE")
                self.has_data = True
                logging.info("Object loaded with raw data")

    def process_meta_data(self):
        print("Processing metadata ........")
        if not self.has_data:
            return
        meta_data = self.raw_data["Meta Data"]

        self.meta_data = pd.DataFrame.from_dict(
            {'symbol': [self.symbol],
             'function': [self.function],
             'interval': [self.interval],
             'datetime': [utilities.convert_between_timezones(
                 check_date_format(meta_data['3. Last Refreshed']),
                 utilities.MARKET_TZ,
                 utilities.UTC_TZ)]},
            orient='columns')

    def process_data(self):
        if not self.has_data:
            return
        # self.meta_data = raw_result["Meta Data"]
        # logging.info("Request meta-data: %s", str(self.meta_data))
        results_df = pd.DataFrame.from_dict(
            self.raw_data[list(
                filter(lambda x: x != "Meta Data", self.raw_data.keys())
            )[0]]).transpose()

        results_df.columns = format_column_names(results_df.columns)
        results_df.reset_index(drop=False, inplace=True)
        results_df.rename(columns={"index": "datetime"}, inplace=True)
        results_df['symbol'] = self.symbol
        if self.interval is not None:
            results_df['interval'] = self.interval
        results_df = utilities.set_column_dtypes(dataframe=results_df,
                                                 dtypes=utilities.DTYPES)
        if 'dividend_amount' in results_df.columns:
            results_df = results_df[
                (results_df['datetime'].map(self.isBusinessDay)) |
                (results_df['dividend_amount'] > 0)]
        else:
            results_df = results_df[results_df['datetime'].map(self.isBusinessDay)]

        results_df = results_df[utilities.time_series_column_order(results_df.columns)]

        logging.info("Dataframe created with these columns: %s", str(list(results_df.columns)))

        results_df = results_df.drop_duplicates(ignore_index=True).merge(
            self.db_data,
            how="outer",
            on=list(results_df.columns),
            indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

        self.new_data = results_df \
            .sort_values('datetime') \
            .reset_index(drop=True)

    def view_data(self):
        return pd.concat(
            [self.db_data, self.new_data]) \
            .drop_duplicates(ignore_index=True) \
            .sort_values('datetime') \
            .reset_index(drop=True)
