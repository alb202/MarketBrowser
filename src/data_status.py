"""Data status tracking
"""

import pandas as pd

from .utilities import *
from .database import *
from .logger import *
from .models import *

log = get_logger(__name__)


class Status:
    """Class for tracking data status
    """
    status_table_name = 'DATA_STATUS'
    datetime_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, cfg):
        log.info("Creating data status object")
        self.table = DataStatus()
        self.data = None
        self.cfg = cfg

    def get_data_status(self, db):
        """Get data statuses from database
        """
        log.info("Getting data status table from database")
        has_dt = {"datetime": self.datetime_format}
        datastatus = pd.read_sql_table(
            table_name=self.status_table_name,
            con=db.engine,
            parse_dates=has_dt,
            index_col=None)
        datastatus['datetime'] = datastatus['datetime'].dt.tz_localize(self.cfg.common_timezone())
        self.data = datastatus

    def update_status_entry(self, symbol, function, interval=None):
        """Update the data status for a symbol/function/interval
        """
        log.info(f"Updating data status for symbol:{symbol}  "
                 f"function:{function}  interval:{interval}")
        query = make_pandas_query(symbol, function, interval)
        self.data.loc[self.data.eval(query), 'datetime'] = get_current_time(
            set_to_utc=True,
            old_timezone=self.cfg.market_timezone(),
            new_timezone=self.cfg.common_timezone())

    def add_status_entry(self, symbol, function, interval=None):
        """Add the data status for a symbol/function/interval
        """
        log.info(f"Adding data status for symbol:{symbol}  "
                 f"function:{function}  interval:{interval}")
        self.data.loc[len(self.data), :] = [symbol,
                                            function,
                                            interval,
                                            get_current_time(
                                                set_to_utc=True,
                                                old_timezone=self.cfg.market_timezone(),
                                                new_timezone=self.cfg.common_timezone())]

    def get_last_update(self, symbol, function, interval=None):
        """Get the data status for a symbol/function/interval
        """
        log.info(f"Get last update for symbol:{symbol}  function:{function}  interval:{interval}")
        query = make_pandas_query(symbol, function, interval)
        results = self.data.query(query).reset_index(drop=True)
        return results['datetime'][0] if len(results) > 0 else None

    def save_table(self, database):
        """Save the data status dataframe to a table
        """
        log.info("Saving data status table to datebase")
        database.update_table(self.data.drop_duplicates(), "DATA_STATUS", "replace")
