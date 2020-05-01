"""This is the database module.

It contains classes to control the database connection and database reading and writing
"""

import logging
import sqlite3

import pandas as pd
import timeseries


# Note: to get sqlite3 working, I copied the DLL from
# https://www.sqlite.org/download.html to the folder {Miniconda folder}\DLLs

class Database:

    def __init__(self, db_location, db_schema):
        self.__db_connection = sqlite3.connect(db_location)
        self.cur = self.__db_connection.cursor()

        if len(self.view_tables()) == 0:
            logging.info("Database has no tables, so they need to be created ...")
            self.create_tables(db_schema)
        else:
            logging.info("Database has already been initialized ...")

    def __del__(self):
        self.__db_connection.close()

    def view_tables(self):

        cursor = self.cur
        tables = [i[0] for i in list(
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"))]
        logging.info("Viewing tables in database ...")
        return tables

    def does_table_exist(self, table):
        """

        :param table:
        :return:
        """
        cursor = self.cur
        tables = [i[0] for i in list(
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"))]
        logging.info("Checking if table %s exists: %s", table, str(table in tables))
        return table in tables

    def create_ts_table(self, table):
        """

        :param table:
        :type table:
        """
        logging.info("Creating table %s ...", table)
        cursor = self.cur
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table}"
                       f"(symbol TEXT NOT NULL, datetime TEXT NOT NULL, open REAL, "
                       f"high REAL, low REAL, close REAL, volume INTEGER);")
        logging.info("Created table %s", table)

    def view_table_info(self, table: str) -> list:
        cursor = self.cur
        cursor = cursor.execute(f'select * from {table}')
        columns = [i[0] for i in cursor.description]
        logging.info("Table %s has columns: %s", table, str(columns))
        return columns


    def create_tables(self, db_schema):
        logging.info("Creating database tables with SQL schema ...")
        cursor = self.cur
        schema = open(db_schema)
        line = schema.readline().strip('\n')
        while line:
            logging.info("Running SQL: %s", line)
            cursor.execute(line)
            line = schema.readline()
        schema.close()

    def load_data_table(self, table, has_dt=False):
        logging.info("Loading all data from table %s", table)
        sql = "SELECT * FROM %s" % table
        return self.table_to_pandas(sql=sql,
                                    has_dt=has_dt)

    def table_to_pandas(self, sql, has_dt=False):
        logging.info("Reading from database ...")
        has_dt = {"datetime": timeseries.DATETIME_FORMAT} if has_dt else None
        return pd.read_sql_query(con=self.__db_connection,
                                 sql=sql,
                                 parse_dates=has_dt)

    def update_table(self, dataframe, table, if_exists='append'):
        logging.info("Adding dataframe to table %s", table)
        cursor = self.cur
        dataframe.to_sql(name=table,
                         con=cursor.connection,
                         if_exists=if_exists,
                         index=False,
                         method=None)
