""" Timeseries module

This module controls the request of data from the server and formatting of timeseries data
"""
import logging
import pandas as pd
import requests
import utils

API_KEY = r"5V11PBP7KPJDDNUP"
URL = r"https://www.alphavantage.co/query?"
RETURN_FORMAT = 'json'

def format_column_names(names):
    logging.info("Formatting column names ...")
    return [i.split(".")[1].replace(" ", "_")[1:] for i in names]

class TimeSeries:

    # def __init__(self, function=None, symbol=None, interval=None):
    def __init__(self, function, symbol, interval=None):
        """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        file_loc : str
            The file location of the spreadsheet
        print_cols : bool, optional
            A flag used to print the columns to the console (default is
            False)

        Returns
        -------
        list
            a list of strings used that are the header columns
        """
        logging.info("Creating TimeSeries object ...")
        self._function = function
        self._symbol = symbol
        self._interval = interval
        self.data = None
        self.meta_data = None

    def get_data(self):

        parameters = {"function": self._function,
                      "symbol": self._symbol,
                      "interval": self._interval,
                      "outputsize": "full",
                      "apikey": API_KEY,
                      "datatype": RETURN_FORMAT
                      }

        logging.info("Getting data with parameters: %s", str(parameters))
        raw_result = requests.get(url=URL, params=parameters).json()
        self.meta_data = raw_result["Meta Data"]
        logging.info("Request meta-data: %s", str(self.meta_data))
        results_df = pd.DataFrame.from_dict(
            raw_result[list(filter(lambda x: x != "Meta Data", raw_result.keys()))[0]]).transpose()
        results_df.columns = format_column_names(results_df.columns)
        results_df.reset_index(drop=False, inplace=True)
        results_df.rename(columns={"index": "datetime"}, inplace=True)
        results_df['symbol'] = self._symbol
        results_df = utils.set_column_dtypes(dataframe=results_df, dtypes=utils.DTYPES)

        if "split_coefficient" in results_df.columns:
            column_order = utils.DATA_COLUMNS3
        elif "adjusted_close" in results_df.columns:
            column_order = utils.DATA_COLUMNS2
        else:
            column_order = utils.DATA_COLUMNS1

        results_df = results_df[column_order]

        logging.info("Dataframe created with these columns: %s", str(list(results_df.columns)))

        self.data = results_df
