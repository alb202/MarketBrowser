""" Timeseries module

This module controls the request of data from the server and formatting of timeseries data
"""
import logging
from datetime import datetime

import pandas as pd
import requests
import utils

API_KEY = "5V11PBP7KPJDDNUP"
URL = "https://www.alphavantage.co/query?"
RETURN_FORMAT = "json"

DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


# TIME_FORMAT = '%H:%M:%S'


def convert_string_to_datetime(string, str_format):
    return datetime.strptime(string, str_format)


def convert_datetime_to_string(dt, str_format):
    return datetime.strftime(dt, str_format)


def check_date_format(dt, convert=True):
    if len(dt.split(" ")) == 1:
        dt = str(dt) + " 16:00:00"
    if convert:
        dt = convert_string_to_datetime(dt, DATETIME_FORMAT)
    return dt


def format_column_names(names):
    logging.info("Formatting column names ...")
    return [i.split(".")[1].replace(" ", "_")[1:] for i in names]


class TimeSeries:

    # def __init__(self, function=None, symbol=None, interval=None):
    def __init__(self):
        logging.info("Creating TimeSeries object ...")
        self.function = ''
        self.symbol = ''
        self.interval = ''
        self.raw_data = None
        self.data = None
        self.meta_data = None
        self.has_data = False

    def get_data(self, function, symbol, interval=None):
        self.function = function
        self.symbol = symbol
        self.interval = interval

        parameters = {"function": self.function,
                      "symbol": self.symbol,
                      "interval": self.interval,
                      "outputsize": "full",
                      "apikey": API_KEY,
                      "datatype": RETURN_FORMAT
                      }

        logging.info("Getting data with parameters: %s", str(parameters))
        try:
            self.raw_data = requests.get(url=URL, params=parameters).json()
        except:
            logging.debug("Data grab failed!")
        finally:
            # print("Finally")
            if (type(self.raw_data) == dict):
                # print("setting has_data to TRUE")
                self.has_data = True
            logging.info("Object loaded with raw data")

    def process_meta_data(self):
        if self.has_data == False:
            return
        meta_data = self.raw_data["Meta Data"]

        # if (len(meta_data['3. Last Refreshed'].split(" ")) == 2):
        #     # date = convert_string_to_datetime(date, DATE_FORMAT)
        #     time = convert_string_to_datetime(meta_data['3. Last Refreshed'].split(" ")[1], TIME_FORMAT)
        # else:
        #     time = None
        self.meta_data = pd.DataFrame.from_dict(
            {'symbol': [self.symbol],
             'function': [self.function],
             'interval': [self.interval],
             'datetime': [check_date_format(meta_data['3. Last Refreshed'])]},
            orient='columns')

    def process_data(self):
        if self.has_data == False:
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
        results_df = utils.set_column_dtypes(dataframe=results_df,
                                             dtypes=utils.DTYPES)

        if "split_coefficient" in results_df.columns:
            column_order = utils.DATA_COLUMNS3
        elif "adjusted_close" in results_df.columns:
            column_order = utils.DATA_COLUMNS2
        else:
            column_order = utils.DATA_COLUMNS1

        results_df = results_df[column_order]

        logging.info("Dataframe created with these columns: %s", str(list(results_df.columns)))

        self.data = results_df
