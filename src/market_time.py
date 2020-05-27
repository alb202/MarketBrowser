"""This is the market time module

It is used for tracking the current time relative to the market hours

"""
import datetime as dt

import logger
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

log = logger.get_logger(__name__)


class MarketTime():
    business_calendar = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    business_calendar_month = pd.offsets.CustomBusinessMonthEnd(calendar=USFederalHolidayCalendar())

    def __init__(self, cfg, testdate=None, testtime=None):
        """Initialize the business hours object
        """
        log.info("Initializing LastBusinessHours object")
        self.cfg = cfg
        current_datetime = self.get_current_datetime(testdate, testtime)
        market_open = self.cfg.market_timezone().localize(
            dt.datetime.combine(current_datetime.date(), self.cfg.market_open()))
        market_close = self.cfg.market_timezone().localize(
            dt.datetime.combine(current_datetime.date(), self.cfg.market_close()))
        self.market_open_common = market_open.astimezone(self.cfg.common_timezone()).time()
        self.market_close_common = market_close.astimezone(self.cfg.common_timezone()).time()
        self.current_datetime_common = current_datetime.astimezone(self.cfg.common_timezone())

    def view_current_datetime(self):
        return self.current_datetime_common

    def get_current_datetime(self, testdate, testtime):
        """Get the current datetime in UTC timezone, or a test datetime
        """
        log.info("Get the current datetime")
        if (testdate is None) & (testtime is None):
            return self.cfg.user_timezone().localize(dt.datetime.now())
        if (testdate is None) & (testtime is not None):
            current_date = dt.datetime.now().date()
            current_time = dt.time(*testtime)
        elif (testdate is not None) & (testtime is None):
            current_date = dt.date(*testdate)
            current_time = dt.datetime.now().time()
        else:
            current_time = dt.time(*testtime)
            current_date = dt.date(*testdate)
        current_datetime = self.cfg.user_timezone().localize(
            dt.datetime.combine(current_date, current_time))
        log.info(f"The current datetime is: {current_datetime}")
        return current_datetime


class BusinessHours():

    def __init__(self, market_time):
        """Initialize BusinessHours object
        """
        self.market_time = market_time
        self.business_day = self.is_today_business_day()
        self.market_hours = self.is_market_hours()
        self.last_market_time = self.get_last_market_time()

    def is_today_business_day(self):
        """Is today a business day
        """
        return self.market_time.business_calendar.rollback(
            self.market_time.current_datetime_common) \
                   .date() == self.market_time.current_datetime_common.date()

    def is_market_hours(self):
        """Check the current time relative to market hours
        """
        log.info("Checking if currently market hours")
        if self.market_time.current_datetime_common.time() < self.market_time.market_open_common:
            log.info("It is before market hours")
            return 'before'
        if self.market_time.current_datetime_common.time() > self.market_time.market_close_common:
            log.info("It is after market hours")
            return 'after'
        log.info("It is currently market hours")
        return 'open'

    def get_last_market_time(self):
        """Get the last time the market was open
        """
        log.info("Get the last time the market was open")
        if not self.business_day:
            last_business_day = self.market_time.business_calendar.rollback(
                self.market_time.current_datetime_common).date()
            last_business_time = self.market_time.market_close_common
        elif (self.market_time.current_datetime_common.time() <
              self.market_time.market_open_common) & self.business_day:
            last_business_day = self.market_time.business_calendar.rollback(
                self.market_time.current_datetime_common - dt.timedelta(days=1)).date()
            last_business_time = self.market_time.market_close_common
        elif (self.market_time.current_datetime_common.time() >
              self.market_time.market_close_common) & \
                self.business_day:
            last_business_day = self.market_time.business_calendar.rollback(
                self.market_time.current_datetime_common + dt.timedelta(days=0)).date()
            last_business_time = self.market_time.market_close_common
        else:
            last_business_day = self.market_time.business_calendar.rollback(
                self.market_time.current_datetime_common + dt.timedelta(days=0)).date()
            last_business_time = self.market_time.current_datetime_common.time()
        last_market_datetime = self.market_time.cfg.common_timezone().localize(
            dt.datetime.combine(last_business_day, last_business_time))
        log.info(f"Last market datetime: {last_market_datetime}")
        return last_market_datetime

    def view_last_market_time(self):
        """Return the last market time
        """
        return self.last_market_time

#
#
# cfg_ = config.Config("../resources/config.txt")
# a = MarketTime(cfg)
# b = BusinessHours(market_time=a)
# print(a.__dict__)
# print(b.__dict__)
