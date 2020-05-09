"""This is the database module.

It contains classes to control the database connection and database reading and writing
"""

# import sqlite3
import pandas as pd
import sqlalchemy as sa
import timeseries
from models import (
    Base)


# from models import DataStatus

# Note: to get sqlite3 working, I copied the DLL from
# https://www.sqlite.org/download.html to the folder {Miniconda folder}\DLLs

# data_status = DataStatus()

class Database:
    def __init__(self, db_location):
        self.Base = Base
        # self.data_status = DataStatus()
        # self.timeseries_intraday = TimeSeriesIntraday()
        # self.timeseries_daily = TimeSeriesDaily()
        # self.timeseries_daily_adjusted = TimeSeriesDailyAdjusted()
        # self.timeseries_weekly = TimeSeriesWeekly()
        # self.timeseries_weekly_adjusted = TimeSeriesWeeklyAdjusted()
        # self.timeseries_monthly = TimeSeriesMonthly()
        # self.timeseries_monthly_adjusted = TimeSeriesMonthlyAdjusted()
        self.engine = sa.create_engine('sqlite:///' + db_location, echo=True)
        print('self.engine')
        self.Session = sa.orm.sessionmaker(bind=self.engine)
        print('self.Session')
        self.meta = sa.MetaData()
        print('self.meta')
        # self.current_session/ = self.Session()
        self.Base.metadata.create_all(bind=self.engine, checkfirst=True)
        self.meta.reflect(bind=self.engine)

        print('self.Base.metadata')
        print(self.meta.tables['DATA_STATUS'])
        # if sqlalchemy_utils.database_exists(self.engine.url):
        #     logging.info("Database has already been initialized ...")
        # else:
        #     logging.info("Database has no tables, so they need to be created ...")
        #     ds = models.DataStatus()
        #
        #     self.base.metadata.create_all(
        #         bind=self.engine,
        #         checkfirst=True,
        #         tables=[ds])
        #     # DataStatus = models.DataStatus()
        #     # DataStatus.create(bind=engine, checkfirst=True)
        # self.meta.reflect(bind=self.engine, )
        # # self.metadata = MetaData(engine)
        # print(self.metadata)

    # def __del__(self):
    #     self.__db_connection.close()

    def new_session(self):
        return self.Session()

    def check_database_connection(self):
        return pd.io.sql._is_sqlalchemy_connectable(self.engine)

    def view_tables(self):
        return self.meta.tables

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

    # def load_data_table(self, table, has_dt=False):
    #     logging.info("Loading all data from table %s", table)
    #     sql = "SELECT * FROM %s" % table
    #     return self.table_to_pandas(sql=sql, has_dt=has_dt)

    def table_to_pandas(self, table, has_dt=False):
        has_dt = {"datetime": timeseries.DATETIME_FORMAT} if has_dt else None
        df = pd.read_sql_table(con=self.engine, table_name=table, parse_dates=has_dt)
        return df

    def query_to_pandas(self, where_dict, has_dt=False):
        has_dt = {"datetime": timeseries.DATETIME_FORMAT} if has_dt else None
        where_statement = self.create_sql(where_dict)
        df = pd.read_sql_query(con=self.engine, sql=where_statement, parse_dates=has_dt)
        return df

    def create_sql(self, values):
        tbl = self.meta.tables[values['table']]
        sql_statement = sa.sql.select([tbl])
        if 'where' in values.keys():
            for i, j in values['where'].items():
                sql_statement = sql_statement.where(tbl.c[i] == j)
        return sql_statement

    def update_table(self, dataframe, table, if_exists='append'):
        dataframe.to_sql(table, con=self.engine, if_exists=if_exists, index=False)

    def update_value(self, table, where, values):
        session = self.Session()
        tbl = self.meta.tables[table]
        value_dict = {tbl.c[i]: j for i, j in values.items()}
        sql = tbl.update().values(value_dict).where(tbl.c[where[0]] == where[1])
        session.execute(sql)
        session.commit()
        session.close()

        # dataframe.to_sql(table, con=self.engine, if_exists=if_exists, index=False)
