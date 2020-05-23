"""This is the market time module

It is used for tracking the current time relative to the market hours

"""
import datetime as dt

import logger
import pandas as pd
import utilities
from pandas.tseries.holiday import USFederalHolidayCalendar

log = logger.get_logger(__name__)


class LastBusinessHours():
    business_calendar = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    business_calendar_month = pd.offsets.CustomBusinessMonthEnd(calendar=USFederalHolidayCalendar())

    def __init__(self, cfg, function, interval=None, testdate=None, testtime=None):
        """Initialize the business hours object
        """
        log.info("Initializing LastBusinessHours object")
        self.cfg = cfg
        current_datetime = self.get_current_datetime(testdate, testtime)
        market_open = self.cfg.market_timezone().localize(
            dt.datetime.combine(current_datetime.date(), self.cfg.market_open()))
        market_close = self.cfg.market_timezone().localize(
            dt.datetime.combine(current_datetime.date(), self.cfg.market_close()))
        self.current_datetime_common = current_datetime.astimezone(self.cfg.common_timezone())
        self.market_open_common = market_open.astimezone(self.cfg.common_timezone()).time()
        self.market_close_common = market_close.astimezone(self.cfg.common_timezone()).time()
        self.is_today_business_day = self.business_calendar.rollback(
            self.current_datetime_common).date() == self.current_datetime_common.date()
        self.market_hours = self.is_market_hours()
        self.last_market_time = self.get_last_market_time()
        self.last_complete_period = self.get_last_complete_period(function, interval)

    def get_last_market_time(self):
        """Get the last time the market was open
        """
        log.info("Get the last time the market was open")
        if not self.is_today_business_day:
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_common).date()
            last_business_time = self.market_close_common
        elif (self.current_datetime_common.time() < self.market_open_common) & \
                self.is_today_business_day:
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_common - dt.timedelta(days=1)).date()
            last_business_time = self.market_close_common
        elif (self.current_datetime_common.time() > self.market_close_common) & \
                self.is_today_business_day:
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_common + dt.timedelta(days=0)).date()
            last_business_time = self.market_close_common
        else:
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_common + dt.timedelta(days=0)).date()
            last_business_time = self.current_datetime_common.time()
        last_market_datetime = self.cfg.common_timezone().localize(
            dt.datetime.combine(last_business_day, last_business_time))
        log.info(f"Last market datetime: {last_market_datetime}")
        return last_market_datetime

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

    def is_market_hours(self):
        """Check the current time relative to market hours
        """
        log.info("Checking if currently market hours")
        if self.current_datetime_common.time() < self.market_open_common:
            log.info("It is before market hours")
            return 'before'
        if self.current_datetime_common.time() > self.market_close_common:
            log.info("It is after market hours")
            return 'after'
        log.info("It is currently market hours")
        return 'open'

    def get_last_complete_period(self, function, interval=None):
        """Get the last complete time period for the current function
        """
        log.info("Getting the last complete time period")
        if 'INTRADAY' in function:
            return self.get_last_complete_interval(interval)
        if 'DAILY' in function:
            return self.get_last_complete_day()
        if 'WEEKLY' in function:
            return self.get_last_complete_week()
        return self.get_last_complete_month()

    def get_last_complete_month(self):
        """Get the last complete month
        """
        log.info("Getting the last complete month")
        offset_days = 0 if self.market_hours == 'after' else 1
        day = self.current_datetime_common - dt.timedelta(days=offset_days)
        day = self.business_calendar_month.rollback(day).date()
        return self.cfg.common_timezone().localize(
            dt.datetime.combine(
                date=day, time=self.market_close_common))

    def get_last_complete_week(self):
        """Get the last complete week
        """
        log.info("Getting the last complete week")
        day_of_week = self.current_datetime_common.isoweekday()
        offset_days = day_of_week + 1 if \
            ((day_of_week in [1, 2, 3, 4]) |
             ((day_of_week == 5) &
              (self.market_hours != 'after'))) else 0
        day = self.business_calendar.rollback(
            self.current_datetime_common - dt.timedelta(days=offset_days)).date()
        return self.cfg.common_timezone().localize(
            dt.datetime.combine(
                date=day, time=self.market_close_common))

    def get_last_complete_day(self):
        """Get the last complete day
        """
        log.info("Getting the last complete day")
        if self.is_today_business_day:
            if self.market_hours == 'after':
                day = self.current_datetime_common.date()
            else:
                day = self.business_calendar.rollback(
                    self.current_datetime_common - dt.timedelta(days=1)).date()
        else:
            day = self.business_calendar.rollback(self.current_datetime_common).date()
        return self.cfg.common_timezone().localize(
            dt.datetime.combine(date=day, time=self.market_close_common))

    def get_last_complete_interval(self, interval):
        """Get the last complete intraday time interval
        """
        log.info("Getting the last complete intraday interval")
        if (not self.is_today_business_day) | (self.market_hours != 'open'):
            return self.last_market_time
        minute = self.current_datetime_common.minute
        hour = self.current_datetime_common.hour
        print('minute', minute)
        print('hour', hour)
        print('interval', interval)
        if interval == '5min':
            minute = utilities.round_down(minute, 5)
        elif interval == '15min':
            minute = utilities.round_down(minute, 15)
        elif interval == '30min':
            minute = utilities.round_down(minute, 30)
        else:
            if minute < 30:
                hour = hour - 1
            minute = 30
        print('minute', minute)
        print('hour', hour)

        new_time = self.cfg.common_timezone().localize(
            dt.datetime.combine(
                self.current_datetime_common.date(),
                dt.time(hour=hour, minute=minute, second=0))).time()
        new_time = new_time if (
                new_time >= self.market_open_common) else self.market_open_common
        return new_time

    def view_last_complete_period(self):
        """Return the last complete time period
        """
        return self.last_complete_period

    def view_last_market_time(self):
        """Return the last market time
        """
        return self.last_market_time

# BUSINESS_DAYS = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
# BUSINESS_MONTHS = pd.offsets.CustomBusinessMonthEnd(calendar=USFederalHolidayCalendar())
# MARKET_OPEN = [9, 30, 0]
# MARKET_CLOSE = [16, 0, 0]
#
#
# class LastBusinessHours:
#     """Determine the last business hours
#     """
#     def __init__(self, function, cfg, testdate=None, testtime=None):
#
#         self.cfg = cfg
#         # self.business_calendar = BUSINESS_MONTHS if "MONTH" in function else BUSINESS_DAYS
#         self.business_calendar = BUSINESS_DAYS
#         if (testdate is None) & (testtime is None):
#             current_datetime = self.cfg.user_timezone().localize(dt.datetime.now())
#         elif (testdate is None) & (testtime is not None):
#             current_date = dt.datetime.now().date()
#             current_time = dt.time(*testtime)
#             current_datetime = dt.datetime.combine(current_date, current_time)
#         elif (testdate is not None) & (testtime is None):
#             current_date = dt.date(*testdate)
#             current_time = dt.datetime.now().time()
#             current_datetime = dt.datetime.combine(current_date, current_time)
#         else:
#             current_time = dt.time(*testtime)
#             current_date = dt.date(*testdate)
#             current_datetime = dt.datetime.combine(current_date, current_time)
#
#         market_open = self.cfg.market_timezone().localize(
#             dt.datetime.combine(current_datetime.date(), dt.time(*MARKET_OPEN)))
#         market_close = self.cfg.market_timezone().localize(
#             dt.datetime.combine(current_datetime.date(), dt.time(*MARKET_CLOSE)))
#         self.current_datetime_common = current_datetime.astimezone(self.cfg.common_timezone())
#         self.market_open_common = market_open.astimezone(self.cfg.common_timezone())
#         self.market_close_common = market_close.astimezone(self.cfg.common_timezone())
#         self.is_today_business_day = self.business_calendar.rollback(
#             self.current_datetime_common).date() == self.current_datetime_common.date()
#         self.is_market_open = self.is_market_hours()
#         self.last_business_datetime = self.determine_last_market_time()
#
#     def is_market_hours(self):
#         """If the market is open, return true. Else, return false
#         """
#         is_markethours = (self.current_datetime_common.time() >=
#                           self.market_open_common.time()) & \
#                          (self.current_datetime_common.time() <=
#                           self.market_close_common.time()) & \
#                          self.is_today_business_day
#         log.info(f"Is it market hours: {is_markethours}")
#         return is_markethours
#
#     def determine_last_market_time(self):
#         """Return the last market date and time
#         """
#         log.info("Determine last market time")
#         if not self.is_today_business_day:
#             log.info("It is not a market day")
#             last_business_day = self.business_calendar.rollback(
#                 self.current_datetime_common).date()
#             last_business_time = self.market_close_common.time()
#         elif (self.current_datetime_common.time() < self.market_open_common.time()) & \
#                 self.is_today_business_day:
#             log.info("It is before the market opens")
#             last_business_day = self.business_calendar.rollback(
#                 self.current_datetime_common - dt.timedelta(days=1)).date()
#             last_business_time = self.market_close_common.time()
#         elif (self.current_datetime_common.time() > self.market_close_common.time()) & \
#                 self.is_today_business_day:
#             log.info("The market has closed")
#             last_business_day = self.business_calendar.rollback(
#                 self.current_datetime_common + dt.timedelta(days=0)).date()
#             last_business_time = self.market_close_common.time()
#         else:
#             log.info("The market is open")
#             last_business_day = self.business_calendar.rollback(
#                 self.current_datetime_common + dt.timedelta(days=0)).date()
#             last_business_time = self.current_datetime_common.time()
#         last_market_time = self.cfg.common_timezone().localize(
#             dt.datetime.combine(last_business_day, last_business_time))
#         log.info(f"Last business date and time: {str(last_market_time)}")
#         return last_market_time
#
#     def view_last_market_time(self):
#         """View last market time as string
#         """
#         return str(self.last_business_datetime)
