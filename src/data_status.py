"""This is the refresh data module

It contains classes to control the data that tracks the status of the API calls
"""
import logging

import utilities


class DataStatus:

    def __init__(self, database_connection):
        self.last_refreshed = None
        self.table = "DATA_STATUS"
        self.data = database_connection.load_data_table(self.table, has_dt=True)
        self.database = database_connection
        logging.info("Created DataStatus object with %s rows", str(len(self.data)))

    def update_row(self, symbol, function, interval=None):
        query = utilities.make_query(symbol, function, interval)
        self.data.loc[self.data.eval(query), 'last_update'] = utilities.get_current_time()

    def add_status(self, symbol, function, interval=None):
        self.data.loc[len(self.data), :] = [symbol,
                                            function,
                                            interval,
                                            utilities.get_current_time()]

    def get_last_update(self, symbol, function, interval=None):
        logging.info("Getting the data status table")
        query = utilities.make_query(symbol, function, interval)
        print("get last update query: ", query)
        results = self.data.query(query)
        print("Get last update results: ", results)
        print("Get last update results length: ", len(results))
        return results['datetime'][0] if len(results) > 0 else None

    def save_table(self):
        self.database.replace_table(self.data, "DATA_STATUS")
