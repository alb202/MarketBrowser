import datetime as dt

import pandas as pd
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

MARKET_TZ_CODE = 'US/Eastern'
UTC_TZ_CODE = 'UTC'
BUSINESS_DAYS = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
BUSINESS_MONTHS = pd.offsets.CustomBusinessMonthEnd(calendar=USFederalHolidayCalendar())
MARKET_OPEN = [9, 30, 0]
MARKET_CLOSE = [16, 0, 0]


class LastBusinessHours:

    def __init__(self, function, testdate=None, testtime=None):
        self.market_tz = pytz.timezone(MARKET_TZ_CODE)
        self.utc_tz = pytz.timezone(UTC_TZ_CODE)
        self.business_calendar = BUSINESS_MONTHS if "MONTH" in function else BUSINESS_DAYS
        # Use test date/time or get the current date/time
        if (testdate is None) & (testtime is None):
            self.current_datetime = self.market_tz.localize(
                dt.datetime.now())
            print(self.current_datetime, self.current_datetime.tzinfo)
        elif (testdate is None) & (testtime is not None):
            current_date = dt.datetime.now().date()
            current_time = dt.time(*testtime)
            self.current_datetime = dt.datetime.combine(current_date, current_time)
        elif (testdate is not None) & (testtime is None):
            current_date = dt.date(*testdate)
            current_time = dt.datetime.now().time()
            self.current_datetime = dt.datetime.combine(current_date, current_time)
        else:
            current_time = dt.time(*testtime)
            current_date = dt.date(*testdate)
            self.current_datetime = dt.datetime.combine(current_date, current_time)

        self.market_open = self.market_tz.localize(
            dt.datetime.combine(self.current_datetime.date(), dt.time(*MARKET_OPEN)))
        self.market_close = self.market_tz.localize(
            dt.datetime.combine(self.current_datetime.date(), dt.time(*MARKET_CLOSE)))
        self.current_datetime_utc = self.current_datetime.astimezone(self.utc_tz)
        self.market_open_utc = self.market_open.astimezone(self.utc_tz)
        self.market_close_utc = self.market_close.astimezone(self.utc_tz)
        self.is_today_business_day = self.business_calendar.rollback(
            self.current_datetime_utc).date() == self.current_datetime_utc.date()
        self.is_market_open = self.is_market_hours()
        self.last_business_datetime = self.get_last_market_time()

    def is_market_hours(self):
        # Return True if market is open, else return false
        print(self.current_datetime_utc)
        print(self.market_open_utc)
        print(self.market_close_utc)
        return (self.current_datetime_utc.time() >= self.market_open_utc.time()) & \
               (self.current_datetime_utc.time() <= self.market_close_utc.time()) & \
               self.is_today_business_day

    def get_last_market_time(self):
        if not self.is_today_business_day:
            print("it is not a business day")
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_utc).date()
            last_business_time = self.market_close_utc.time()
        elif (self.current_datetime_utc.time() < self.market_open_utc.time()) & \
                self.is_today_business_day:
            print("It is before the market opens")
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_utc - dt.timedelta(days=1)).date()
            last_business_time = self.market_close_utc.time()
        elif (self.current_datetime_utc.time() > self.market_close_utc.time()) & \
                self.is_today_business_day:
            print("The market has closed")
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_utc + dt.timedelta(days=0)).date()
            last_business_time = self.market_close_utc.time()
        else:
            print("It is during market hours")
            last_business_day = self.business_calendar.rollback(
                self.current_datetime_utc + dt.timedelta(days=0)).date()
            last_business_time = self.current_datetime_utc.time()
        print('last_business_time', last_business_time)
        print('last_business_day', last_business_day)
        last_market_time = self.utc_tz.localize(
            dt.datetime.combine(last_business_day, last_business_time))

        return last_market_time
