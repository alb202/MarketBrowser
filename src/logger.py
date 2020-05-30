"""Logging module
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "../app.log"


def get_console_handler():
    """Create the console handler
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    """Create the log file handler
    """
    file_handler = RotatingFileHandler(LOG_FILE, backupCount=0, maxBytes=100000)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name):
    """Create the logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
