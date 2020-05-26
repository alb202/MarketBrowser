""" Timeseries module

This module controls the request of data from the server and formatting of timeseries data
"""

import datetime as dt
import sys

import logger
import pandas as pd
import requests
import utilities

log = logger.get_logger(__name__)

DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


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
        new_datetime = convert_string_to_datetime(new_datetime, DATETIME_FORMAT)
    return new_datetime


def format_column_names(names):
    """Format column names for pandas
    """
    log.info("Formatting column names")
    return [i.split(".")[1].replace(" ", "_")[1:] for i in names]


class TimeSeries:
    """Manage time series data
    """

    def __init__(self, cfg, function, symbol, interval=None):
        log.info("Creating TimeSeries object")
        self.function = function
        self.symbol = symbol
        self.interval = interval
        self.raw_data = None
        self.new_data = None
        self.db_data = None
        self.new_dividend_data = None
        self.db_dividend_data = None
        self.meta_data = None
        self.has_data = False
        self.cfg = cfg
        self.is_business_day = pd.tseries.offsets.BDay().is_on_offset

        self.dividend_period = self.get_dividend_period()

    def get_data_from_database(self, con, has_dt=False):
        """Retrieve data from database table
        """
        log.info("Getting data from database ...")
        where_dict = {'table': self.function, 'where': {'symbol': self.symbol}}
        if self.interval is not None:
            where_dict['where']['interval'] = self.interval
        try:
            self.db_data = con.query_to_pandas(
                where_dict=where_dict, has_dt=has_dt).sort_values('datetime').reset_index(drop=True)
        except requests.ConnectionError as error:
            log.warn("Cannot get data from database. Exiting!")
            log.info(error)
            sys.exit()
        finally:
            if isinstance(self.db_data, pd.DataFrame):
                self.has_data = True
                log.info("Object loaded with data from database")

    def get_dividend_data_from_database(self, con):
        """Retrieve data from database table
        """
        log.info("Getting dividend data from database ...")
        where_dict = {'table': 'DIVIDEND',
                      'where': {'symbol': self.symbol,
                                'period': self.dividend_period}}
        try:
            self.db_dividend_data = con.query_to_pandas(
                where_dict=where_dict, has_dt=True).sort_values('datetime').reset_index(drop=True)
        except requests.ConnectionError as error:
            log.warn("Cannot get dividend data from database. Exiting!")
            log.info(error)
            sys.exit()

    def get_data_from_server(self):
        """Get fresh data from API
        """
        log.info('Getting data from server')
        parameters = {"function": self.function,
                      "symbol": self.symbol,
                      "interval": self.interval,
                      "outputsize": self.cfg.view_outputsize(),
                      "apikey": self.cfg.view_apikey(),
                      "datatype": self.cfg.view_format()
                      }
        try:
            self.raw_data = requests.get(
                url=self.cfg.view_url(),
                params=parameters).json()
        except requests.RequestException as error:
            log.info("Data grab failed. Exiting!")
            log.warn(error)
            sys.exit()
        else:
            if isinstance(self.raw_data, dict):
                if "Error Message" in self.raw_data.keys():
                    log.warn(f"API call unsuccessful: {self.raw_data['Error Message']}")
                    sys.exit()
                else:
                    self.has_data = True
                    log.info("Object loaded with raw data")

    def process_meta_data(self):
        """Convert the raw JSON metadata into pandas dataframe
        """
        log.info('Processing metadata from server')
        if not self.has_data:
            return
        meta_data = self.raw_data["Meta Data"]
        self.meta_data = pd.DataFrame.from_dict(
            {'symbol': [self.symbol],
             'function': [self.function],
             'interval': [self.interval],
             'datetime': [utilities.convert_between_timezones(
                 check_date_format(meta_data['3. Last Refreshed']),
                 self.cfg.market_timezone(),
                 self.cfg.common_timezone())]},
            orient='columns')

    def process_data(self):
        """Convert the raw JSON time series data into pandas dataframe
        """
        log.info("Processing raw data from API")
        if not self.has_data:
            log.warning("No raw data to process")
            return
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
            dividend_df = results_df[['symbol', 'datetime', 'dividend_amount']]
            dividend_df = dividend_df.query("dividend_amount > 0")
            if len(dividend_df) > 0:
                dividend_df.loc[:, 'period'] = self.dividend_period
                if self.db_dividend_data is not None:
                    dividend_df = dividend_df.drop_duplicates(ignore_index=True).merge(
                        self.db_dividend_data,
                        how="outer",
                        on=list(dividend_df.columns),
                        indicator=True).query('_merge == "left_only"') \
                        .drop('_merge', axis=1)
                self.new_dividend_data = dividend_df. \
                    drop_duplicates(). \
                    sort_values('datetime'). \
                    reset_index(drop=True)
        results_df = results_df.drop('dividend_amount', axis=1)
        results_df = results_df[results_df['datetime'].map(self.is_business_day)]

        log.info(f"Time series data created with columns: {str(list(results_df.columns))}")

        results_df = results_df.drop_duplicates(ignore_index=True).merge(
            self.db_data,
            how="outer",
            on=list(results_df.columns),
            indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

        self.new_data = results_df \
            .sort_values('datetime') \
            .reset_index(drop=True)

    def view_data(self):
        """Return the database data and fresh API data
        """
        prices = pd.concat([self.db_data, self.new_data]) \
            .drop_duplicates(ignore_index=True) \
            .sort_values('datetime') \
            .reset_index(drop=True)
        dividends = pd.concat([self.db_dividend_data, self.new_dividend_data]) \
            .drop_duplicates(ignore_index=True) \
            .sort_values('datetime') \
            .reset_index(drop=True)
        return {'prices': prices, 'dividends': dividends}

    def save_new_data(self, database):
        """Save the new data to the database table
        """
        log.info("Saving new data to database")
        database.update_table(dataframe=self.new_data,
                              table=self.function,
                              if_exists="append")
        if self.new_dividend_data is not None:
            database.update_table(dataframe=self.new_dividend_data,
                                  table='DIVIDEND',
                                  if_exists="append")

    def remove_nonunique_rows(self, database):
        """Remove any non-unique rows from the database
        """
        merge_cols = ['symbol', 'datetime']
        if self.interval is not None:
            merge_cols.append('interval')
        value_cols = ['open', 'high', 'low', 'close', 'volume']
        df = self.new_data.merge(
            self.db_data,
            how="outer",
            on=merge_cols,
            indicator=True,
            suffixes=['', '_db']
        ).query('_merge == "both"')
        if len(df) == 0:
            log.info("No obsolete data found in database")
            return
        query = ' | '.join(['(' + i + ' != ' + i + '_db' + ')' for i in value_cols])
        df = df.query(query)
        if len(df) == 0:
            log.info("No rows with conflicting data")
            return
        self.delete_dataframe_from_database(database=database, dataframe=df, table=self.function)
        self.db_data = self.db_data.merge(
            df[merge_cols],
            how="left",
            on=merge_cols,
            indicator=True,
            suffixes=['', '_db']
        ).query('_merge == "left_only"').drop('_merge', axis=1)

    def remove_nonunique_dividend_rows(self, database):
        """Remove any non-unique dividends from the database
        """
        if self.new_dividend_data is None:
            return
        merge_cols = ['symbol', 'datetime', 'period']
        value_cols = ['dividend_amount']
        df = self.new_dividend_data.merge(
            self.db_dividend_data,
            how="outer",
            on=merge_cols,
            indicator=True,
            suffixes=['', '_db']
        ).query('_merge == "both"')
        if len(df) == 0:
            log.info("No obsolete dividend data found in database")
            return
        query = ' | '.join(['(' + i + ' != ' + i + '_db' + ')' for i in value_cols])
        df = df.query(query)
        if len(df) == 0:
            log.info("No rows with conflicting data")
            return
        self.delete_dataframe_from_database(database=database, dataframe=df, table='DIVIDEND')
        self.db_dividend_data = self.db_dividend_data.merge(
            df[merge_cols],
            how="left",
            on=merge_cols,
            indicator=True,
            suffixes=['', '_db']
        ).query('_merge == "left_only"').drop('_merge', axis=1)

    def delete_dataframe_from_database(self, database, dataframe, table):
        """Delete a dataframe from the database using the current function table
        """
        for index, row in dataframe.iterrows():
            where = dict()
            where['symbol'] = row['symbol']
            where['datetime'] = row['datetime'].to_pydatetime()
            if 'interval' in dataframe.columns:
                where['interval'] = row['interval']
            if 'period' in dataframe.columns:
                where['period'] = row['period']
            values = {'table': table,
                      'where': where}
            log.info(f"Obsolete row: {str(values)}")
            database.delete_record(values=values)

    def get_dividend_period(self):
        if 'DAILY' in self.function:
            return 'day'
        if 'WEEKLY' in self.function:
            return 'week'
        if 'MONTHLY' in self.function:
            return 'month'
        return None
