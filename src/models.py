"""SQLAlchemy data table models
"""
from sqlalchemy import Column, Text, DateTime, Float, BigInteger, Date
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class TimeSeries():
    """Basic time series model
    """
    symbol = Column(Text, primary_key=True, nullable=False)
    datetime = Column(Date, primary_key=True, nullable=False)
    open = Column(Float, primary_key=False, nullable=False)
    high = Column(Float, primary_key=False, nullable=False)
    low = Column(Float, primary_key=False, nullable=False)
    close = Column(Float, primary_key=False, nullable=False)
    volume = Column(BigInteger, primary_key=False, nullable=False)


class TimeSeriesAdjusted():
    """Basic time series model for adjusted prices
    """
    adjusted_close = Column(Float, primary_key=False, nullable=False)


class TimeSeriesIntraday(TimeSeries, Base):
    """Model for TimeSeriesIntraday table
    """
    __tablename__ = 'TIME_SERIES_INTRADAY'
    datetime = Column(DateTime, primary_key=True, nullable=False)
    interval = Column(Text, primary_key=True, nullable=False)


class TimeSeriesDaily(TimeSeries, Base):
    """Model for TimeSeriesDaily table
    """
    __tablename__ = 'TIME_SERIES_DAILY'


class TimeSeriesDailyAdjusted(TimeSeries, TimeSeriesAdjusted, Base):
    """Model for TimeSeriesDailyAdjusted table
    """
    __tablename__ = 'TIME_SERIES_DAILY_ADJUSTED'
    split_coefficient = Column(Float, primary_key=False, nullable=False)


class TimeSeriesWeekly(TimeSeries, Base):
    """Model for TimeSeriesWeekly table
    """
    __tablename__ = 'TIME_SERIES_WEEKLY'


class TimeSeriesWeeklyAdjusted(TimeSeries, TimeSeriesAdjusted, Base):
    """Model for TimeSeriesWeeklyAdjusted table
    """
    __tablename__ = 'TIME_SERIES_WEEKLY_ADJUSTED'


class TimeSeriesMonthly(TimeSeries, Base):
    """Model for TimeSeriesMonthly table
    """
    __tablename__ = 'TIME_SERIES_MONTHLY'


class TimeSeriesMonthlyAdjusted(TimeSeries, TimeSeriesAdjusted, Base):
    """Model for TimeSeriesMonthlyAdjusted table
    """
    __tablename__ = 'TIME_SERIES_MONTHLY_ADJUSTED'


class DataStatus(Base):
    """Model for data status table
    """
    __tablename__ = 'DATA_STATUS'
    symbol = Column(Text, primary_key=True, nullable=False)
    function = Column(Text, primary_key=False, nullable=False)
    interval = Column(Text, primary_key=False, nullable=True)
    datetime = Column(DateTime, primary_key=True, nullable=False)


class Dividend(Base):
    """Model for dividend class
    """
    __tablename__ = 'DIVIDEND'
    symbol = Column(Text, primary_key=True, nullable=False, unique=False)
    dividend_amount = Column(Float, primary_key=False, nullable=False, unique=False)
    datetime = Column(Date, primary_key=True, nullable=False, unique=False)
    period = Column(Text, primary_key=True, nullable=False, unique=False)


class Financials(Base):
    """Model for financials class
    """
    __tablename__ = 'FINANCIALS'
    symbol = Column(Text, primary_key=True, nullable=False, unique=True)
    name = Column(Text, primary_key=False, nullable=True, unique=False)
    type = Column(Text, primary_key=False, nullable=True, unique=False)
    sector = Column(Text, primary_key=False, nullable=True, unique=False)
    industry = Column(Text, primary_key=False, nullable=True, unique=False)
    rea = Column(BigInteger, primary_key=False, nullable=True, unique=False)
    sharesOutstanding = Column(BigInteger, primary_key=True, nullable=False, unique=False)
    float = Column(Float, primary_key=False, nullable=True, unique=False)
