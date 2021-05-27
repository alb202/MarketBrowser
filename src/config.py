"""Module for loading configuration settings
"""

import datetime as dt
import random
import sys
import pytz
from .logger import *

log = get_logger(__name__)

class Config:
    """Class for loading and accessing configuration settings
    """

    def __init__(self, path):
        log.info(f"Creating config object with config file {path}")
        self._config = self.load_config(path)
        self._price_key = self.load_key(self.view_price_apikey_path())
        self._stock_key = self.load_key(self.view_stock_apikey_path())

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            return f"Parameter {name} not found! Exiting."

    @staticmethod
    def load_key(path):
        """Load the api key from file
        """
        log.info("Loading API key")
        try:
            with open("./" + path, 'r') as file:
                data = file.read().split('\n')
        except (IOError, FileNotFoundError):
            log.warning("API key file not found. Exiting!")
            sys.exit()
        else:
            log.info("API key loaded successfully")
            return data

    @staticmethod
    def load_config(path):
        """Load the configuration settings from file
        """
        log.info("Loading config file")
        try:
            with open(path, 'r') as file:
                cfg = dict()
                for line in file:
                    splt_line = line.split("=")
                    cfg[splt_line[0].strip()] = splt_line[1].strip()
        except (IOError, FileNotFoundError):
            log.warning("Config file not found. Exiting!")
            sys.exit()

        return cfg

    def view_price_url(self):
        return self._config['price_api_url']

    def view_stock_url(self):
        return self._config['stock_api_url']

    def view_price_apikey(self):
        return random.sample(self._price_key, len(self._price_key))

    def view_price_apikey_path(self):
        return "../" + self._config['price_api_key_path']

    def view_stock_apikey(self):
        return random.sample(self._stock_key, len(self._stock_key))

    def view_stock_apikey_path(self):
        return "../" + self._config['stock_api_key_path']

    def view_db_location(self):
        return "../" + self._config['db_location']

    def user_timezone(self):
        return pytz.timezone(self._config['user_timezone'])

    def market_timezone(self):
        return pytz.timezone(self._config['market_timezone'])

    def common_timezone(self):
        return pytz.timezone(self._config['common_timezone'])

    def market_open(self):
        return dt.time(*[int(i) for i in self._config['market_open'].split(',')])

    def market_close(self):
        return dt.time(*[int(i) for i in self._config['market_close'].split(',')])