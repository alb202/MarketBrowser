"""

"""
import logging

import utilities

STATUS_TABLE_NAME = 'DATA_STATUS'


class DataStatus:
    """
    Class DataStatus holds the most recent update time for each query
    ...

    Attributes
    ----------
    table : str
        Name of database table holding the data status

    data : Pandas DataFrame
        DataFrame containing data status table

    database : Sqlite Connection
        Sqlite database connection passed to instantiation

    Methods
    -------
    update_status_entry(symbol, function, interval=None)
        Update the data status for a single symbol and function
    add_status_entry(symbol, function, interval=None)
        Add a row to the data status table for a unique symbol and function
    get_last_update(symbol, function, interval=None):
        Get the last update time for a symbol and function
    save_table():
        Save the data status table back to the database
    """
    def __init__(self, database_connection):
        self.table = STATUS_TABLE_NAME
        self.data = database_connection.load_data_table(self.table, has_dt=True)
        self.database = database_connection
        logging.info("Created DataStatus object with %s rows", str(len(self.data)))

    def update_status_entry(self, symbol, function, interval=None):
        query = utilities.make_query(symbol, function, interval)
        self.data.loc[self.data.eval(query), 'datetime'] = utilities.get_current_time()

    def add_status_entry(self, symbol, function, interval=None):
        self.data.loc[len(self.data), :] = [symbol,
                                            function,
                                            interval,
                                            utilities.get_current_time()]

    def get_last_update(self, symbol, function, interval=None):
        logging.info("Getting the data status table")
        query = utilities.make_query(symbol, function, interval)
        print("get last update query: ", query)
        results = self.data.query(query).reset_index(drop=True)
        print("Get last update results: ", results)
        print("Get last update results type: ", type(results))
        print("Get last update results length: ", len(results))
        print("Last update datetime column: ", results['datetime'])
        return results['datetime'][0] if len(results) > 0 else None

    def save_table(self):
        self.database.update_table(self.data, "DATA_STATUS", "replace")
