from sqlalchemy import Column, Text, DateTime, Float, BigInteger, Date
from sqlalchemy.ext.declarative import declarative_base

# import sqlalchemy_utils
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class TimeSeries():
    symbol = Column(Text, primary_key=True, nullable=False)
    datetime = Column(Date, primary_key=True, nullable=False)
    open = Column(Float, primary_key=False, nullable=False)
    high = Column(Float, primary_key=False, nullable=False)
    low = Column(Float, primary_key=False, nullable=False)
    close = Column(Float, primary_key=False, nullable=False)
    volume = Column(BigInteger, primary_key=False, nullable=False)


class TimeSeriesAdjusted():
    adjusted_close = Column(Float, primary_key=False, nullable=False)
    dividend_amount = Column(Float, primary_key=False, nullable=False)


class TimeSeriesIntraday(TimeSeries, Base):
    __tablename__ = 'TIME_SERIES_INTRADAY'
    datetime = Column(DateTime, primary_key=True, nullable=False)
    interval = Column(Text, primary_key=True, nullable=False)


class TimeSeriesDaily(TimeSeries, Base):
    __tablename__ = 'TIME_SERIES_DAILY'


class TimeSeriesDailyAdjusted(TimeSeries, TimeSeriesAdjusted, Base):
    __tablename__ = 'TIME_SERIES_DAILY_ADJUSTED'
    split_coefficient = Column(Float, primary_key=False, nullable=False)


class TimeSeriesWeekly(TimeSeries, Base):
    __tablename__ = 'TIME_SERIES_WEEKLY'


class TimeSeriesWeeklyAdjusted(TimeSeries, TimeSeriesAdjusted, Base):
    __tablename__ = 'TIME_SERIES_WEEKLY_ADJUSTED'


class TimeSeriesMonthly(TimeSeries, Base):
    __tablename__ = 'TIME_SERIES_MONTHLY'


class TimeSeriesMonthlyAdjusted(TimeSeries, TimeSeriesAdjusted, Base):
    __tablename__ = 'TIME_SERIES_MONTHLY_ADJUSTED'


class DataStatus(Base):
    __tablename__ = 'DATA_STATUS'
    # name = 'DataStatus'
    symbol = Column(Text, primary_key=True, nullable=False)
    function = Column(Text, primary_key=False, nullable=False)
    interval = Column(Text, primary_key=False, nullable=True)
    datetime = Column(DateTime, primary_key=True, nullable=False)


class Dividend(Base):
    __tablename__ = 'DIVIDEND'

    symbol = Column(Text, primary_key=True, nullable=False, unique=False)
    dividend = Column(Float, primary_key=False, nullable=False, unique=False)
    date = Column(Date, primary_key=True, nullable=False, unique=False)

    def __init__(self, symbol, dividend, date):
        self.symbol = symbol
        self.dividend = dividend
        self.date = date

    def __repr__(self):
        return '<{} model>'.format(self.__tablename__)


'''

DB_LOCATION = "../db/database7.sqlite"
engine = sa.create_engine('sqlite:///'+DB_LOCATION, echo=True)
Session = sa.orm.sessionmaker(bind=engine)
meta = sa.MetaData()
Base.metadata.create_all(bind=engine, checkfirst=True)
meta.reflect(bind=engine)
session = Session()

# print(type(dt.datetime(2020, 5, 4)))#, 21, 55, 30)))

new_symbol1 = Dividend("USRT", 1.5, dt.datetime(2020,5,2).date())
new_symbol2 = Dividend("SPY", 0.214, dt.datetime(2020,5,3).date())
new_symbol3 = Dividend("AGGY", 4.098, dt.datetime(2020,5,4).date())
print(new_symbol1)
# session.add(new_symbol1)
# session.add(new_symbol2)
# session.add(new_symbol3)


session.commit()
session.close()
session = Session()
# new_symbol4 = session.query(Dividends).all()
# print("4: ", new_symbol4)

table_name = 'DIVIDEND'
model_name = 'Dividend'
symbol_to_filter = 'SPY'
filter_column = 'symbol'
column_to_update = 'datetime'

# print(meta)
table = meta.tables[table_name]
# table.update()
# print(f"Table {table} exists: ", table.exists(bind=engine))
# if not table.exists(bind=engine):
#     table.create(bind=engine, checkfirst=True)
# where_statement = session.query(table).filter(table.c['symbol'] == 'USRT').all()  #.query([table]).filter(table.c['symbol'] == 'SPY').all()
# where_statement = table.select().where(table.filter(table.c['symbol'] == 'USRT').all()  #.query([table]).filter(table.c['symbol'] == 'SPY').all()
where_statement = sa.sql.select([table])#.fr (table_name)
# where_statement = where_statement.where(table.c['symbol'] == symbol_to_filter)
print(where_statement)
# where_statement = table.query.all()

# .text("SELECT * FROM :table WHERE symbol == :symbol").bindparams(table=table_name, symbol=symbol_to_filter)
print(pd.read_sql_query(con=engine, sql=where_statement))
# print(type(table))
# print(help(table))
# print(session.execute(select([table])).__dict__)
# value_dict = {table.c[i]: j for i, j in {table.c[column_to_update]:dt.datetime.now()}.items()}
# value_dict = {table.c[i]: j for i, j in {column_to_update: dt.datetime.now().date()}.items()}

# def make_where(conditions, operators=None):
#     for i in conditions:

print(value_dict)
stmt = table.update().values(value_dict).where((table.c[filter_column]==symbol_to_filter) | (table.c['dividend']>3))
session.execute(stmt)
session.commit()
session.close()
session = Session()
a = pd.read_sql_table(con=engine, table_name=table_name)
session.commit()
session.close()
session = Session()
b = pd.DataFrame(
    {'symbol':['AGGY', 'MSFT'],
     'dividend':[0.233, 4.432],
     'datetime':[dt.datetime.now().date(), dt.datetime.now().date()]})
print(a)
print(b)
print("is database connected: ", pd.io.sql._is_sqlalchemy_connectable(engine))
# print("is session connected: ", pd.io.sql._is_sqlalchemy_connectable(session))
b.to_sql(table_name, con=engine, if_exists='append', index=False)
session.commit()
session.close()
session=Session()

print("is database connected: ", pd.io.sql._is_sqlalchemy_connectable(engine))

c = pd.read_sql_table(con=engine, table_name=table_name)
print(c)
session.commit()
session.close()

# stmt = table.append().values({table.c[column_to_update]:dt.datetime.now()}).where((table.c[filter_column]==symbol_to_filter) | (table.c['dividend']>3))
# session.execute(stmt)
# print(table.c[column_to_update])
# table..query(Dividends).filter(Dividends.symbol=='SPY').update({Dividends.datetime: dt.datetime.now()})
# print(table)
# t = Table(table_name, )
# stmt = select([t])#.where(column())
# # stmt = sa.text('SELECT * FROM :table WHERE symbol=:symbol').params(table='Dividends', symbol='SPY')
# print(stmt)
# # stmt = stmt.bindparams(sa.bindparam("x", type_=String),sa.bindparam("y", type_=String))
# # print(stmt)
# engine.connect().execute(stmt)
# session.query(Dividends).filter(Dividends.symbol=='SPY').update({Dividends.datetime: dt.datetime.now()})
# stmt =
# new_symbol4 = session.query(Dividends).all()

# print("4: ", new_symbol4)
# print(help(session))

#
# session = Session()
# # print(engine.connect())
# sql = "SELECT * FROM Dividend"
# data_model = 'Dividends'
# table_name = 'Dividend'
# symbol = 'SPY'
# # sql = sa.sql.select([data_model]).where(Dividends.column('symbol')=='SPY')
# stmt = sa.text('SELECT * FROM :x WHERE symbol = :y')
# stmt = stmt.bindparam(x=table_name, y=symbol)
# print(stmt)
# df = pd.read_sql_query(
#     con=engine,
#     sql=stmt)
# print(df)
# session.close()


# connection = engine.connect()
# # metadata = sa.MetaData()
# print(sqlalchemy_utils.database_exists(engine.url))
'''
