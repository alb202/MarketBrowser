import sqlite3

DB_PATH = ./db/

class Database(sqlite3):

    def __init__(self, path):
        self._path = path
        return(None)

