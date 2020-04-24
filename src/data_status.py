"""This is the refresh data module

It contains classes to control the data that tracks the status of the API calls
"""
import logging

# import pandas as pd
# import datetime
import utils


class DataStatus:

    def __init__(self, db):
        self.last_refreshed = None
        self.table = "DATA_STATUS"
        self.data = db.load_data_table(self.table, has_dt=True)
        self.db = db
        logging.info("Created DataStatus object with %s rows" % str(len(self.data)))

    def update_row(self, symbol, function, interval=None):
        query = utils.make_query(symbol, function, interval)
        self.data.loc[self.data.eval(query), 'last_update'] = utils.get_current_time()

    def add_status(self, symbol, function, interval=None):
        self.data.loc[len(self.data), :] = [symbol, function, interval, utils.get_current_time()]

    def get_last_update(self, symbol, function, interval=None):
        logging.info("Getting the data status table")
        utils.make_query(symbol, function, interval)
        return self.data.query(query)['datetime'][0]

    def save_table(self):
        self.db.replace_table(self.data, "DATA_STATUS")

    #
    # def does_table_exist(self, table):
    #     """
    #
    #     :param table:
    #     :return:
    #     """
    #     cursor = self.cur
    #     tables = [i[0] for i in list(
    #         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"))]
    #     logging.info("Checking if table %s exists: %s", table, str(table in tables))
    #     return table in tables
    #
    # def create_ts_table(self, table):
    #     logging.info("Creating table %s ...", table)
    #     cursor = self.cur
    #     cursor.execute(f"CREATE TABLE IF NOT EXISTS {table}"
    #                    f"(symbol TEXT NOT NULL, datetime TEXT NOT NULL, open REAL, "
    #                    f"high REAL, low REAL, close REAL, volume INTEGER);")
    #     logging.info("Created table %s", table)
    #
    # def view_table_info(self, table):
    #     cursor = self.cur
    #     cursor = cursor.execute(f'select * from {table}')
    #     columns = [i[0] for i in cursor.description]
    #     logging.info("Table %s has columns: %s", table, str(columns))
    #     return columns
    #
    #
    # def append_to_table(self, dataframe, table):
    #     logging.info("Adding dataframe to table %s", table)
    #     cursor = self.cur
    #     dataframe.to_sql(name=table,
    #                      con=cursor.connection,
    #                      if_exists='append',
    #                      index=False,
    #                      method=None)
    #
    # def create_tables(self, db_schema):
    #     logging.info("Creating database tables with SQL schema ...")
    #     cursor = self.cur
    #     schema = open(db_schema)
    #     line = schema.readline().strip('\n')
    #     while line:
    #         logging.info("Running SQL: %s", line)
    #         cursor.execute(line)
    #         line = schema.readline()
    #     schema.close()
    #
    # def load_data_table(self, table, has_dt=False):
    #     logging.info("Loading all data from table %s", table)
    #     sql = "SELECT * FROM %s" % table
    #     return self.table_to_pandas(sql=sql,
    #                                 has_dt=has_dt)
    #
    # def table_to_pandas(self, sql, has_dt=False):
    #     logging.info("Creating database with SQL call ...")
    #     has_dt = {"datetime": timeseries.DATETIME_FORMAT} if has_dt else None
    #     return pd.read_sql_query(con=self.__db_connection,
    #                              sql=sql,
    #                              parse_dates=has_dt)
