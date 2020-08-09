from abc import ABC

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from app_utilities import *
from plotly.subplots import make_subplots


class MovingAverages(ABC):

    @staticmethod
    def simple_moving_average(ts, period):
        return ts.rolling(window=period).mean()

    @staticmethod
    def exponential_moving_average(ts, period):
        return ts.ewm(span=period, adjust=False).mean()


class MACD(MovingAverages):
    def __init__(self, data, function, short_period=12, long_period=26, signal_period=9):
        if len(data.columns) != 2:
            print("Timeseries must have exactly 1 data column and 1 date column! Exiting.")
            exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.xaxis = data['datetime']
        self.periods = {'short_period': short_period,
                        'long_period': long_period,
                        'signal_period': signal_period}
        data_col = [i for i in data.columns if i != 'datetime'][0]
        self.macd = self.exponential_moving_average(ts=data[data_col], period=short_period) - \
                    self.exponential_moving_average(ts=data[data_col], period=long_period)
        self.macd_signal = self.exponential_moving_average(ts=self.macd, period=signal_period)
        self.macd_histogram = self.macd - self.macd_signal

    def macd_crossovers(self):
        cross = np.sign(self.macd - self.macd_signal)
        return np.sign(cross.diff().replace({np.NaN: 0}))

    def plot_macd(self, trace_only=False):
        bar_color = self.macd_histogram.copy(deep=True)
        bar_color.loc[bar_color < 0] = -1
        bar_color.loc[bar_color > 0] = 1
        bar_color.replace({-1: '#FF0000', 0: '#C0C0C0', 1: '#009900'}, inplace=True)
        histogram_trace = dict(
            trace=go.Bar(name='MACD Histogram',
                         x=self.xaxis,
                         marker_color=list(bar_color),
                         y=self.macd_histogram * 2))
        macd_trace = dict(
            trace=go.Scatter(name='MACD', mode='lines',
                             x=self.xaxis,
                             marker_color='#000000',
                             y=self.macd))
        signal_trace = dict(
            trace=go.Scatter(name='MACD Signal', mode='lines',
                             x=self.xaxis,
                             marker_color='#FF0000',
                             y=self.macd_signal))
        if trace_only:
            return dict(histogram=histogram_trace,
                        macd=macd_trace,
                        signal=signal_trace)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **histogram_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **macd_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **signal_trace)
        fig.update_layout(dict(bargap=.0, width=1200, height=500,
                               title=f'MACD ({self.periods["short_period"]},'
                                     f'{self.periods["long_period"]},'
                                     f'{self.periods["signal_period"]})',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text="MACD Signal")),
                               yaxis2=dict(showgrid=False, ticks="", title=dict(text="MACD Histogram"))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig


class RSI(MovingAverages):
    def __init__(self, data, function, period=14):
        if len(data.columns) != 2:
            print("Timeseries must have exactly 1 data column and 1 date column! Exiting.")
            exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.xaxis = data['datetime']#.dt.strftime('%d/%m/%y %-H:%M')
        self.period = period
        data_col = [i for i in data.columns if i != 'datetime'][0]
        self.data = data[data_col]
        self.change_in_price = data[data_col].diff(1).replace({np.NaN: 0})
        self.avg_gain = self.simple_moving_average(self.change_in_price.mask(self.change_in_price < 0, 0.0),
                                                   period=self.period)
        self.avg_loss = self.simple_moving_average(self.change_in_price.mask(self.change_in_price > 0, 0.0),
                                                   period=self.period)
        self.rsi_line = (100 - (100 / (1 + (abs(self.avg_gain / self.avg_loss)))))

    def plot_rsi(self, trace_only=False):

        rsi_trace = dict(
            trace=go.Scatter(
                name='RSI',
                mode='lines',
                x=self.xaxis,
                marker_color='black',
                y=self.rsi_line))
        bottom_line = dict(y0=30, y1=30, x0=self.xaxis[0], x1=self.xaxis[len(self.xaxis) - 1],
                           line=dict(color="green", width=2, dash="dash"), type="line")
        top_line = dict(y0=70, y1=70, x0=self.xaxis[0], x1=self.xaxis[len(self.xaxis) - 1],
                        line=dict(color="red", width=2, dash="dash"), type="line")
        if trace_only:
            return dict(rsi=rsi_trace,
                        top_line=top_line,
                        bottom_line=bottom_line)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **rsi_trace)
        fig.add_shape(row=1, col=1, **top_line)
        fig.add_shape(row=1, col=1, **bottom_line)
        fig.update_layout(dict(bargap=.0, width=1200, height=500,
                               title=f'RSI ({self.period})',
                               xaxis=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis=dict(showgrid=False, range=[0, 100], title=dict(text="RSI"))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig


class HeikinAshi(MovingAverages):
    def __init__(self, data, function):
        if len(data.columns) != 5:
            print("Timeseries must have exactly OHLC columns and 1 date column! Exiting.")
            exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.xaxis = data['datetime']#.dt.strftime('%d/%m/%y %-H:%M')
        self.ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = [self.ha_close[0]]
        for i in range(len(self.ha_close) - 1):
            ha_open.append((ha_open[i] + self.ha_close[i]) / 2)
        self.ha_open = pd.Series(ha_open)
        self.ha_low = pd.concat([data['low'], self.ha_close, self.ha_open], axis=1).min(axis=1)
        self.ha_high = pd.concat([data['high'], self.ha_close, self.ha_open], axis=1).max(axis=1)

    def ha_indicator(self):
        current_change = self.ha_close > self.ha_open
        last_change = self.ha_close.shift(1) > self.ha_open.shift(1)
        buy_indicator = current_change & ~last_change
        sell_indicator = ~current_change & last_change
        return np.where(buy_indicator == False,
                        sell_indicator.replace({True: -1, False: 0}),
                        buy_indicator.replace({True: 1, False: 0}))

    def get_values(self):
        return dict(datetime=self.xaxis, open=self.ha_open,
                    high=self.ha_high, low=self.ha_low, close=self.ha_close)

    def plot_ha(self, trace_only=False, show_indicators=True):
        bar_color = np.sign(self.ha_open - self.ha_close)
        bar_color.replace({-1: 'red', 0: 'gray', 1: 'green'}, inplace=True)
        ha_trace = dict(
            trace=go.Candlestick(
                name='candlestick',
                showlegend=False,
                x=self.xaxis,
                open=self.ha_open,
                high=self.ha_high,
                low=self.ha_low,
                close=self.ha_close))
        if show_indicators | ~trace_only:
            ha_indicator = self.ha_indicator()
            ha_indicator_colors = np.where(ha_indicator == 0, 'white', np.where(ha_indicator == 1, 'green', 'red'))
            ha_indicator_trace = dict(
                trace=go.Scatter(
                    name='HA Indicator', mode='markers',
                    showlegend=False,
                    x=self.xaxis,
                    marker_color=ha_indicator_colors,
                    y=np.where(
                        ha_indicator == 0,
                        None,
                        np.where(
                            ha_indicator == 1,
                            self.ha_high * 1.02,
                            self.ha_low * .98))))
        if trace_only:
            return dict(ha=ha_trace, indicator=ha_indicator_trace)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **ha_trace)
        if show_indicators:
            fig.add_trace(row=1, col=1, **ha_indicator_trace)
        fig.update_layout(dict(width=1200, height=600,
                               title=f'Heikin Ashi',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(showgrid=True, title=dict(text="Heikin Ashi"))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig


class MovingAverageCrossover(MovingAverages):
    def __init__(self, data, function, ma1_type='sma', ma2_type='sma', ma1_period=13, ma2_period=26):
        if len(data.columns) != 2:
            print("Timeseries must have exactly 1 data column and 1 date column! Exiting.")
            exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.params = dict(
            ma1_type=ma1_type, ma1_period=ma1_period,
            ma2_type=ma2_type, ma2_period=ma2_period)
        self.xaxis = data['datetime']
        data_col = [i for i in data.columns if i != 'datetime'][0]
        self.data = data[data_col]
        self.ma1 = self.simple_moving_average(ts=self.data, period=ma1_period) \
            if ma1_type == 'sma' \
            else self.exponential_moving_average(ts=self.data, period=ma1_period)
        self.ma2 = self.simple_moving_average(ts=self.data, period=ma2_period) \
            if ma2_type == 'sma' \
            else self.exponential_moving_average(ts=self.data, period=ma2_period)
        current_change = self.ma1 > self.ma2
        last_change = self.ma1.shift(1) > self.ma2.shift(1)
        cross_up = current_change & ~last_change
        cross_down = ~current_change & last_change
        self.indicator = np.where(cross_up == False,
                                  cross_down.replace({True: -1, False: 0}),
                                  cross_up.replace({True: 1, False: 0}))

    def get_MAC_indicator(self):
        return pd.DataFrame(
            data={
                'datetime': pd.Series(self.xaxis),
                'indicator': pd.Series(self.indicator),
                'ma1': pd.Series(self.ma1),
                'ma2': pd.Series(self.ma2)})

    def plot_MAC(self, trace_only=False):
        indicator_color = self.indicator
        indicator_color = np.where(indicator_color == 0, 'white',
                                   np.where(indicator_color == 1, 'yellow', 'purple'))
        MAC_trace = dict(
            trace=go.Scatter(
                name='Moving Average Crossover', mode='markers',
                showlegend=False,
                x=self.xaxis,
                marker_color=indicator_color,
                y=np.where(
                    self.indicator == 0,
                    None,
                    np.where(
                        self.indicator == 1,
                        self.ma1 * 1.05,
                        self.ma1 * .95))))

        ma1_trace = dict(
            trace=go.Scatter(
                name='Moving Average 1', mode='lines',
                showlegend=False,
                x=self.xaxis,
                marker_color='orange',
                y=self.ma1))
        ma2_trace = dict(
            trace=go.Scatter(
                name='Moving Average 2', mode='lines',
                showlegend=False,
                x=self.xaxis,
                marker_color='blue',
                y=self.ma2))
        if trace_only:
            return dict(crossover=MAC_trace, ma1=ma1_trace, ma2=ma2_trace)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **MAC_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **ma1_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **ma2_trace)
        fig.update_layout(dict(width=1200, height=500,
                               title=f'Moving Average Crossover ({self.params["ma1_type"]}:{self.params["ma1_period"]}, '
                                     f'{self.params["ma2_type"]}:{self.params["ma2_period"]})',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text='Moving Averages'))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig


class MovingAverageZone(MovingAverages):
    def __init__(self, datetime, open, close, function, ma1_type='sma', ma2_type='sma', ma1_period=5, ma2_period=30):
        # if len(data.columns) != 3:
        #     print("Timeseries must have exactly 1 data column and 1 date column! Exiting.")
        #     exit()
        # if 'datetime' not in data.columns:
        #     print("Timeseries must have datetime column! Exiting.")
        #     exit()
        self.function = function
        self.params = dict(
            ma1_type=ma1_type, ma1_period=ma1_period,
            ma2_type=ma2_type, ma2_period=ma2_period)
        self.xaxis = datetime
        self.open = open
        self.close = close
        # data_col = [i for i in data.columns if i != 'datetime'][0]
        # self.data = data[data_col]
        self.ma1 = self.simple_moving_average(ts=self.close, period=ma1_period) \
            if ma1_type == 'sma' \
            else self.exponential_moving_average(ts=self.close, period=ma1_period)
        self.ma2 = self.simple_moving_average(ts=self.close, period=ma2_period) \
            if ma2_type == 'sma' \
            else self.exponential_moving_average(ts=self.close, period=ma2_period)

        self.up_day = self.close > self.open
        trend_up = self.up_day & (self.ma1 > self.ma2) & (self.close > self.ma1)
        trend_down = ~self.up_day & (self.ma1 <= self.ma2) & (self.close < self.ma1)
        self.indicator = np.where(
            trend_up == False,
            trend_down.replace({True: -1, False: 0}),
            trend_up.replace({True: 1, False: 0}))

    def get_MAZ_indicator(self):
        return pd.DataFrame(
            data={
                'datetime': pd.Series(self.xaxis),
                'indicator': pd.Series(self.indicator),
                'ma1': pd.Series(self.ma1),
                'ma2': pd.Series(self.ma2)})

    def plot_MAZ(self, trace_only=False):
        indicator_color = self.indicator
        indicator_color = np.where(indicator_color == 0, 'white',
                                   np.where(indicator_color == 1, 'green', 'red'))
        MAZ_trace = dict(
            trace=go.Scatter(
                name='Moving Average Zones', mode='markers',
                showlegend=False,
                x=self.xaxis,
                marker_color=indicator_color,
                y=np.where(
                    self.indicator == 0,
                    None,
                    np.where(
                        self.indicator == 1,
                        self.ma1 * .9,
                        self.ma1 * .9))))

        if trace_only:
            return dict(indicator=MAZ_trace)  # , ma1=ma1_trace, ma2=ma2_trace)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **MAZ_trace)
        # fig.add_trace(row=1, col=1, secondary_y=False, **ma1_trace)
        # fig.add_trace(row=1, col=1, secondary_y=False, **ma2_trace)
        fig.update_layout(dict(width=1200, height=500,
                               title=f'Moving Average Zones ({self.params["ma1_type"]}:{self.params["ma1_period"]}, '
                                     f'{self.params["ma2_type"]}:{self.params["ma2_period"]})',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text='Moving Averages'))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig

#
#
# #
# #
# # #
# from config import Config
# # from data_status import DataStatus
# from database import Database
# # from market_symbols import MarketSymbols
# from timeseries import TimeSeries
# #
# # #
# cfg = Config("../resources/config.txt")
# db_connection = Database(cfg.view_db_location())
# db_connection.check_database()
# #
# the_function = 'TIME_SERIES_DAILY_ADJUSTED'
# the_interval = None#'30min#
# the_symbol = 'SPY'
#
# query = TimeSeries(con=db_connection,
#                    function=the_function,
#                    symbol=the_symbol,
#                    interval=the_interval)
# query.get_local_data()
# price_data = query.view_data()['prices'].loc[:, ['datetime', 'close', 'open', 'high', 'low']]
# # print(price_data)
# #
# # new_macd = MACD(
# #     data=price_data[['datetime', 'close']],
# #     function=the_function)
# #
# # # print(new_macd.__dict__['rate_of_change'])
# # # new_macd.plot_macd(trace_only=False)
# # #
# # new_rsi = RSI(
# #     data=price_data[['datetime', 'close']],
# #     function=the_function,
# #     period=14)
# # # new_rsi.plot_rsi()
# # # # print(new_rsi.__dict__['rsi_line'])
# # #
# # new_ha = HeikinAshi(function=the_function, data=price_data)
# # # c = new_ha.ha_indicator()
# # # print(len(c))
# # # print(c)
# # # new_ha.plot_ha()
# # # # a = new_ha.get_values()
# # # # print((a['open'] > a['close']) | (a['open'] < a['close']))
# # #
# # new_ma = MovingAverageCrossover(function=the_function,
# #                                 data=price_data[['datetime', 'close']], ma1_period=10, ma2_period=20)
# # # print(new_ma.get_indicator().query("indicator == 1"))
# # # d = new_ma.plot_MAC(trace_only=True)
# # # print(d)
# # new_ma.plot_MAC(trace_only=False)
#
#
# #
# new_maz = MovingAverageZone(
#     function=the_function,
#     datetime=price_data.datetime,
#     open=price_data.open,
#     close=price_data.close,
#     ma1_period=5, ma2_period=30)
# # print(new_ma.get_indicator().query("indicator == 1"))
# # d = new_ma.plot_MAC(trace_only=True)
# # print(d)
# new_maz.plot_MAZ(trace_only=False)
