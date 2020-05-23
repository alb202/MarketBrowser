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
        self.meta_data = None
        self.has_data = False
        self.cfg = cfg
        self.is_business_day = pd.tseries.offsets.BDay().is_on_offset

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

    def get_data_from_server(self):
        """Get fresh data from API
        """
        log.info('Getting data from server')
        # self.function = function
        # self.symbol = symbol
        # self.interval = interval
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
            results_df = results_df[
                (results_df['datetime'].map(self.is_business_day)) |
                (results_df['dividend_amount'] > 0)]
        else:
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
        return pd.concat(
            [self.db_data, self.new_data]) \
            .drop_duplicates(ignore_index=True) \
            .sort_values('datetime') \
            .reset_index(drop=True)

    def save_new_data(self, database):
        """Save the new data to the database table
        """
        log.info("Saving new data to database")
        self.remove_nonunique_rows(database)
        # database.update_table(dataframe=self.new_data,
        #                       table=self.function,
        #                       if_exists="append")

    def remove_nonunique_rows(self, database):
        """Remove any non-unique rows from the database
        """
        merge_cols = ['symbol', 'datetime']
        if self.interval is not None:
            merge_cols.append('interval')
        value_cols = ['open', 'high', 'low', 'close', 'volume']
        print('merge_cols: ', merge_cols)
        print('value_cols: ', value_cols)
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
        print(query)
        df = df.query(query)
        if len(df) == 0:
            log.info("No rows with conflicting data")
            print(df)
            return
        print('Rows with conflicting data: ', df)
        self.delete_dataframe_from_database(database=database, dataframe=df)
        print("self.db_data", self.db_data)
        self.db_data = self.db_data.merge(
            df[merge_cols],
            how="outer",
            on=merge_cols,
            indicator=True,
            suffixes=['', '_db']
        ).query('_merge == "left_only"').drop('_merge', axis=1)
        print("self.db_data new", self.db_data)

    def delete_dataframe_from_database(self, database, dataframe):
        """Delete a dataframe from the database using the current function table
        """
        for index, row in dataframe.iterrows():
            where = dict()
            where['symbol'] = row['symbol']
            where['datetime'] = row['datetime'].to_pydatetime()
            if self.interval is not None:
                where['interval'] = row['interval']
            values = {'table': self.function,
                      'where': where}
            log.info(f"Obsolete row: {str(values)}")
            database.delete_record(values=values)

    # def remove_last_entry(self, delete_from_db=None):
    #     """Remove the last entry from the database data
    #     """
    #     log.info("Removing last entry from db data")
    #     if delete_from_db is not None:
    #         sql = dict()
    #         sql['table'] = self.function
    #         sql['where'] = {key:value[0] for key, value in
    #                         self.db_data[-1:].reset_index(drop=True).to_dict(orient='list').items()}
    #         if "INTRADAY" not in self.function:
    #             sql['where']['datetime'] = sql['where']['datetime'].date()
    #         delete_from_db.delete_record(values=sql)
    #         log.info(f"Removing data from database: {sql}")
    #     self.db_data = self.db_data[:-1]

    # def delete_obsolete_data(self, database, last_complete):
    #     """Remove obsolete rows from database
    #     """
    #     log.info("Deleting obsolete data from database")
    #     if 'INTRADAY' not in self.function:
    #         obsolete_rows = self.db_data[self.db_data['datetime'].dt.date > last_complete.date()]
    #     else:
    #         obsolete_rows = self.db_data[self.db_data['datetime'] > last_complete]
    #     print(obsolete_rows)
    #
    #     if obsolete_rows is not None:
    #         for index, row in obsolete_rows.iterrows():
    #             where = dict()
    #             where['symbol'] = row['symbol']
    #             where['datetime'] = row['datetime'].to_pydatetime()
    #             if self.interval is not None:
    #                 where['interval'] = row['interval']
    #             values = {'table': self.function,
    #                       'where': where}
    #             log.info(f"Obsolete row: {str(values)}")
    #             database.delete_record(values=values)
    #         self.db_data = self.db_data[self.db_data['datetime'].dt.date <= last_complete.date()]

    #
    # merge_cols = ['symbol', 'datetime']
    # if self.interval is not None:
    #     merge_cols.append('interval')
    # value_cols = ['open', 'high', 'low', 'close', 'volume']
    #
    # df = self.new_data.merge(
    #     self.db_data,
    #     how="outer",
    #     on=merge_cols,
    #     indicator=True,
    #     suffixes=['', '_db']
    # ).query('_merge == "both"')
    # if df is None:
    #     log.info("No obsolete data found in database")
    #     return None
    # query = ' | '.join(['(' + i + ' != ' + i + '_db' + ')' for i in value_cols])
    # print(query)
    # df = df.query(query)
    # if df is None:
    #     return None
    # for index, row in df.iterrows():
    #     where = dict()
    #     where['symbol'] = row['symbol']
    #     where['datetime'] = row['datetime']
    #     if "INTRADAY" not in self.function:
    #         where['datetime'] = row['datetime'].date()
    #     if 'interval' in merge_cols:
    #         where['interval'] = row['interval']
    #     values = {'table': self.function, 'where': where}
    #     log.info(f"Obsolete data values: {str(values)}")
    #     database.delete_record(values=values)
    #
