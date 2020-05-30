"""This is the database module.

It contains classes to control the database connection and database reading and writing
"""
import logger
import pandas as pd
import sqlalchemy as sa
import utilities
from models import Base
from sqlalchemy.exc import OperationalError

log = logger.get_logger(__name__)


# Note: to get sqlite3 working, I copied the DLL from
# https://www.sqlite.org/download.html to the folder {Miniconda folder}\DLLs

class Database:
    """Database connection class
    """

    def __init__(self, db_location):
        """Initializing a database connection object
        """
        log.info("Creating database object")
        self.Base = Base
        self.engine = sa.create_engine('sqlite:///' + db_location, echo=False)
        self.Session = sa.orm.sessionmaker(bind=self.engine)
        self.meta = sa.MetaData()
        self.Base.metadata.create_all(bind=self.engine, checkfirst=True)
        self.meta.reflect(bind=self.engine)

    def view_tables(self):
        """View database tables
        """
        return self.meta.tables

    def table_to_pandas(self, table, has_dt=False):
        """Read database table into pandas dataframe
        """
        log.info("Writing table to database")
        has_dt = {"datetime": utilities.DATETIME_FORMAT} if has_dt else None
        return pd.read_sql_table(con=self.engine, table_name=table, parse_dates=has_dt)

    def query_to_pandas(self, where_dict, has_dt=False):
        """Read database query into pandas dataframe
        """
        log.info("Reading query from database")
        has_dt = {"datetime": utilities.DATETIME_FORMAT} if has_dt else None
        where_statement = self.create_sql_for_selection(where_dict)
        return pd.read_sql_query(con=self.engine, sql=where_statement, parse_dates=has_dt)

    def create_sql_for_selection(self, values):
        """Creates SQLAlchemy selectable statement from dictionary
        """
        log.info("Creating sql statement")
        log.info(str(values))
        tbl = self.meta.tables[values['table']]
        sql_statement = sa.sql.select([tbl])
        if 'where' in values.keys():
            sql_statement = self.make_where_statement(
                table=tbl, sql=sql_statement, where=values['where'])
        return sql_statement

    def create_sql_for_deletion(self, values):
        """Creates SQLAlchemy selectable statement from dictionary
        """
        log.info("Creating sql statement")
        log.info(str(values))
        tbl = self.meta.tables[values['table']]
        sql_statement = tbl.delete()
        if 'where' in values.keys():
            sql_statement = self.make_where_statement(
                table=tbl, sql=sql_statement, where=values['where'])
        log.info(f"SQL statement: {sql_statement}")
        return sql_statement

    @staticmethod
    def make_where_statement(table, sql, where):
        for i, j in where.items():
            # print('i: ', i)
            # print('j: ', j)
            sql = sql.where(table.c[i] == j)
        # print(sql)
        return sql

    def update_table(self, dataframe, table, if_exists='append'):
        """Update an existing table with a pandas dataframe
        """
        log.info(f"Updating table {table} in database")
        dataframe.to_sql(table, con=self.engine, if_exists=if_exists, index=False)

    def delete_record(self, values):
        """Update a specific value in a dataframe
        """
        log.info(f"Deleting record in table {values['table']} "
                 f"where {str(values['where'])} from database")
        session = self.Session()
        sql = self.create_sql_for_deletion(values)
        session.execute(sql)
        session.commit()
        session.close()

    def get_record(self, values):
        """Getting a record based on a query
        """
        log.info(f"Getting record in table {values['table']} "
                 f"where {str(values['where'])} from database")
        session = self.Session()
        sql = self.create_sql_for_selection(values)
        results = session.execute(sql).fetchall()
        session.commit()
        session.close()
        return results

    def check_database(self):
        """Check if the database is connectable
        """
        try:
            self.engine.execute("SELECT 1")
        except OperationalError:
            log.warn("Database check failed")
            return False
        log.info("Database check succeeded")
        return True
