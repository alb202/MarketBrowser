from abc import ABC

# from plotly.subplots import make_subplots
from src.app_utilities import *
# from src.app_utilities import make_rangebreaks
from src.utilities import *
from src.retracements import *

class MovingAverages(ABC):

    @staticmethod
    def simple_moving_average(ts, period):
        return ts.rolling(window=period).mean().round(4)

    @staticmethod
    def exponential_moving_average(ts, period):
        return ts.ewm(span=period, adjust=False).mean().round(4)


class MACD(MovingAverages):
    def __init__(self, data, function, close_col='close', short_period=12, long_period=26, signal_period=9):
        # if len(data.columns) != 2:
        #     print("Timeseries must have exactly 1 data column and 1 date column! Exiting.")
        #     exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.xaxis = data['datetime']
        self.periods = {'short_period': short_period,
                        'long_period': long_period,
                        'signal_period': signal_period}
        # data_col = [i for i in data.columns if i != 'datetime'][0]
        self.macd = self.exponential_moving_average(ts=data[close_col], period=short_period) - \
                    self.exponential_moving_average(ts=data[close_col], period=long_period)
        self.macd_signal = self.exponential_moving_average(ts=self.macd, period=signal_period)
        self.macd_histogram = eval('self.macd - self.macd_signal')
        self.crossovers = self.macd_crossovers()
        self.macd_trend = np.where(eval('(self.macd_histogram - self.macd_histogram.shift(1)) >= 0'), 1, -1)

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': self.xaxis,
                  'macd': self.macd,
                  'macd_signal': self.macd_signal,
                  'macd_histogram': self.macd_histogram,
                  'macd_crossovers': self.crossovers,
                  'macd_trend': self.macd_trend}).set_index('datetime')

    def macd_crossovers(self):
        # cross =
        return np.sign(self.macd - self.macd_signal).diff().replace({np.NaN: 0})

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
    def __init__(self, data, function, close_col='close', period=14):
        # if len(data.columns) != 2:
        #     print("Timeseries must have exactly 1 data column and 1 date column! Exiting.")
        #     exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.xaxis = data['datetime']  # .dt.strftime('%d/%m/%y %-H:%M')
        self.period = period
        # data_col = [i for i in data.columns if i != 'datetime'][0]
        self.data = data[close_col]
        self.change_in_price = data[close_col].diff(1).replace({np.NaN: 0})
        self.avg_gain = self.simple_moving_average(self.change_in_price.mask(self.change_in_price < 0, 0.0),
                                                   period=self.period)
        self.avg_loss = self.simple_moving_average(self.change_in_price.mask(self.change_in_price > 0, 0.0),
                                                   period=self.period)
        self.rsi_line = (100 - (100 / (1 + (abs(self.avg_gain / self.avg_loss)))))

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': self.xaxis,
                  'rsi': self.rsi_line,
                  'rsi_crossover': np.where(self.rsi_line >= 80, 1,
                                            np.where(self.rsi_line <= 20, -1, 0))}) \
            .set_index('datetime')

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
        # if len(data.columns) != 5:
        #     print("Timeseries must have exactly OHLC columns and 1 date column! Exiting.")
        #     exit()
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.xaxis = data['datetime']  # .dt.strftime('%d/%m/%y %-H:%M')
        self.ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = [self.ha_close[0]]
        for i in range(len(self.ha_close) - 1):
            ha_open.append((ha_open[i] + self.ha_close[i]) / 2)
        self.ha_open = pd.Series(ha_open)
        self.ha_low = pd.concat([data['low'], self.ha_close, self.ha_open], axis=1).min(axis=1)
        self.ha_high = pd.concat([data['high'], self.ha_close, self.ha_open], axis=1).max(axis=1)
        self.indicator = self.ha_indicator()
        self.trend = np.where(self.ha_close >= self.ha_open, 1, 0)
        lower_val = np.where(self.ha_close > self.ha_open, self.ha_open, self.ha_close)
        higher_val = np.where(self.ha_close > self.ha_open, self.ha_close, self.ha_open)
        self.bottom_shadow = np.where(self.ha_low < lower_val, 1, 0)
        self.top_shadow = np.where(self.ha_high > higher_val, 1, 0)

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': self.xaxis,
                  'ha_open': self.ha_open,
                  'ha_high': self.ha_high,
                  'ha_low': self.ha_low,
                  'ha_close': self.ha_close,
                  'ha_trend': self.trend,
                  'ha_bottom': self.bottom_shadow,
                  'ha_top': self.top_shadow,
                  'ha_indicator': self.indicator}).set_index('datetime')

    def ha_indicator(self):
        current_change = self.ha_close > self.ha_open
        last_change = self.ha_close.shift(1) > self.ha_open.shift(1)
        buy_indicator = (current_change & ~last_change).replace({True: 1, False: 0})
        sell_indicator = (~current_change & last_change).replace({True: -1, False: 0})
        return np.where(buy_indicator == 1, buy_indicator, sell_indicator)

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
    def __init__(self, data, function, close_col='close', ma1_type='sma', ma2_type='sma', ma1_period=13, ma2_period=26):
        if 'datetime' not in data.columns:
            print("Timeseries must have datetime column! Exiting.")
            exit()
        self.function = function
        self.params = dict(
            ma1_type=ma1_type, ma1_period=ma1_period,
            ma2_type=ma2_type, ma2_period=ma2_period)
        self.xaxis = data['datetime']
        self.data = data[close_col]
        self.ma1 = self.simple_moving_average(ts=self.data, period=ma1_period) \
            if ma1_type == 'sma' \
            else self.exponential_moving_average(ts=self.data, period=ma1_period)
        self.ma2 = self.simple_moving_average(ts=self.data, period=ma2_period) \
            if ma2_type == 'sma' \
            else self.exponential_moving_average(ts=self.data, period=ma2_period)
        current_change = self.ma1 > self.ma2
        last_change = self.ma1.shift(1) > self.ma2.shift(1)
        # print('cross_up: ', (current_change & ~last_change))
        cross_up = (current_change & ~last_change).replace({True: 1, False: 0})
        cross_down = (~current_change & last_change).replace({True: -1, False: 0})
        self.indicator = np.where(cross_up == 1, cross_down, cross_up)

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': self.xaxis,
                  'mac_indicator': self.indicator,
                  'mac_ma1': self.ma1,
                  'mac_ma2': self.ma2}).set_index('datetime')

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
    def __init__(self, datetime, close, ha_open, ha_close, function,
                 ma1_type='sma', ma2_type='sma', ma1_period=5, ma2_period=30):
        self.function = function
        self.params = dict(
            ma1_type=ma1_type, ma1_period=ma1_period,
            ma2_type=ma2_type, ma2_period=ma2_period)
        self.xaxis = datetime
        self.ma1 = self.simple_moving_average(ts=close, period=ma1_period) \
            if ma1_type == 'sma' \
            else self.exponential_moving_average(ts=close, period=ma1_period)
        self.ma2 = self.simple_moving_average(ts=close, period=ma2_period) \
            if ma2_type == 'sma' \
            else self.exponential_moving_average(ts=close, period=ma2_period)

        self.up_day = ha_close > ha_open
        trend_up = (self.up_day & (self.ma1 > self.ma2) & (close > self.ma1)).replace({True: 1, False: 0})
        trend_down = (~self.up_day & (self.ma1 <= self.ma2) & (close < self.ma1)).replace({True: -1, False: 0})
        self.indicator = np.where(trend_up == 1, trend_down, trend_up)

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': pd.Series(self.xaxis),
                  'maz_indicator': pd.Series(self.indicator),
                  'maz_ma1': pd.Series(self.ma1),
                  'maz_ma2': pd.Series(self.ma2)}).set_index('datetime')

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


class HABuySell(MovingAverages):
    def __init__(self, datetime, close, ha_open, ha_close, function, ma1_type='sma', ma2_type='sma',
                 ma1_period=5, ma2_period=30, only_first=False, only_ma_crossover=True):
        self.function = function
        self.params = dict(
            ma1_type=ma1_type, ma1_period=ma1_period,
            ma2_type=ma2_type, ma2_period=ma2_period)
        self.xaxis = datetime
        self.ma1 = self.simple_moving_average(ts=close, period=ma1_period) \
            if ma1_type == 'sma' \
            else self.exponential_moving_average(ts=close, period=ma1_period)
        self.ma2 = self.simple_moving_average(ts=close, period=ma2_period) \
            if ma2_type == 'sma' \
            else self.exponential_moving_average(ts=close, period=ma2_period)

        up_day = ha_close > ha_open
        last_day = ha_close.shift(1).fillna(1) <= ha_open.shift(1).fillna(0) \
            if only_first \
            else close > 0
        trend_up = (up_day & last_day)
        trend_up = trend_up.replace({True: 1, False: 0})
        self.ma_crossover = (self.ma1 >= self.ma2).replace({True: 1, False: 0})
        trend_up = trend_up & self.ma_crossover \
            if only_ma_crossover \
            else trend_up
        trend_down = (~up_day & ~last_day).replace({True: -1, False: 0})
        self.indicator = np.where(trend_up > 0, trend_up, trend_down)

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': pd.Series(self.xaxis),
                  'habs_indicator': pd.Series(self.indicator),
                  'habs_crossover': pd.Series(self.ma_crossover),
                  'habs_ma1': pd.Series(self.ma1),
                  'habs_ma2': pd.Series(self.ma2)}).set_index('datetime')

    def plot_HABS(self, trace_only=False):
        indicator_color = self.indicator
        indicator_color = np.where(indicator_color == 0, 'white',
                                   np.where(indicator_color == 1, 'green', 'red'))
        HABS_trace = dict(
            trace=go.Scatter(
                name='Hieken Ashi Buy Sell', mode='markers',
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

        ma1_trace = dict(
            trace=go.Scatter(
                name='Short MA', mode='lines',
                showlegend=False,
                x=self.xaxis,
                marker_color='blue',
                y=self.ma1))

        ma2_trace = dict(
            trace=go.Scatter(
                name='Long MA', mode='lines',
                showlegend=False,
                x=self.xaxis,
                marker_color='red',
                y=self.ma2))

        if trace_only:
            return dict(indicator=HABS_trace)  # , ma1=ma1_trace, ma2=ma2_trace)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **HABS_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **ma1_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **ma2_trace)
        fig.update_layout(dict(width=1200, height=500,
                               title=f'Hieken Ashi Buy Sell ({self.params["ma1_type"]}:{self.params["ma1_period"]}, '
                                     f'{self.params["ma2_type"]}:{self.params["ma2_period"]})',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text='Moving Averages'))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig


class MAChange(MovingAverages):
    def __init__(self, datetime, close, function, ma_type='ema', period=8, change=1):
        self.function = function
        self.period = period
        self.ma_type = ma_type
        self.change = change
        self.xaxis = datetime
        self.close = close
        if self.ma_type == 'sma':
            self.ma = self.simple_moving_average(ts=self.close, period=period)
        else:
            self.ma = self.exponential_moving_average(ts=self.close, period=period)
        self.ma_shift = self.ma.shift(change)
        self.trend = self.ma > self.ma_shift
        self.indicator = np.where(self.trend, 1, -1)

    def table(self):
        return pd.DataFrame.from_dict(
            orient='columns',
            data={'datetime': pd.Series(self.xaxis),
                  'ma_indicator': pd.Series(self.indicator),
                  'ma': self.ma,
                  'ma_previous': self.ma_shift}).set_index('datetime')

    def plot_MAChange(self, trace_only=False):
        indicator_color = self.indicator
        indicator_color = np.where(indicator_color == 1, 'green', 'red')
        MA_change = dict(
            trace=go.Scatter(
                name='Moving Average Change', mode='markers',
                showlegend=False,
                x=self.xaxis,
                marker_color=indicator_color,
                y=self.close * .85))

        if trace_only:
            return dict(indicator=MA_change)  # , ma1=ma1_trace, ma2=ma2_trace)
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **MA_change)
        fig.update_layout(dict(width=1200, height=500,
                               title=f'Moving Average Change ({self.ma_type} {self.period} {self.change})',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text='Moving Averages'))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig


def create_indicator_table(data, function, indicators={1, 2, 3, 4, 5, 6, 7, 8}):
    """Calculate the indicators for the table
    """
    if (len(data) == 0) | (len(indicators) == 0):
        return pd.DataFrame.from_dict({i: [] for i in ['datetime']}, orient='columns')
        # return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close'], index=None).set_index('datetime')
        # return None
    results = data.loc[:, ['datetime', 'open', 'high', 'low', 'close']].set_index('datetime')

    if 1 in indicators:
        _macd = MACD(data=data.loc[:, ['datetime', 'close']], function=function)
        _macd_df = _macd.table()
        results = results.join(_macd_df, how='outer')

    if 2 in indicators:
        _rsi = RSI(data=data.loc[:, ['datetime', 'close']], function=function)
        _rsi_df = _rsi.table()
        results = results.join(_rsi_df, how='outer')

    if len({3, 4, 5, 6, 7, 8}.intersection(indicators)) > 0:
        _ha = HeikinAshi(data=data.loc[:, ['datetime', 'open', 'high', 'low', 'close']], function=function)
        _ha_df = _ha.table()

        ha_data_ma = _ha_df[['ha_open', 'ha_close']].reset_index(drop=False)
        ha_data_ma.columns = ['datetime', 'open', 'close']

        if 3 in indicators:
            results = results.join(_ha_df, how='outer')

        if 4 in indicators:
            _mac = MovingAverageCrossover(data=ha_data_ma.loc[:, ['datetime', 'close']],
                                          function=function,
                                          ma1_period=10,
                                          ma2_period=20)
            _mac_df = _mac.table()
            results = results.join(_mac_df, how='outer')

        if 5 in indicators:
            _maz = MovingAverageZone(datetime=ha_data_ma['datetime'],
                                     close=data['close'],
                                     ha_open=ha_data_ma['open'],
                                     ha_close=ha_data_ma['close'],
                                     function=function,
                                     ma1_type='sma',
                                     ma2_type='sma',
                                     ma1_period=5,
                                     ma2_period=30)
            _maz_df = _maz.table()
            results = results.join(_maz_df, how='outer')

        if 6 in indicators:
            _habs = HABuySell(datetime=ha_data_ma['datetime'],
                              close=data['close'],
                              ha_open=ha_data_ma['open'],
                              ha_close=ha_data_ma['close'],
                              function=function,
                              only_first=True,
                              only_ma_crossover=False,
                              ma1_type='sma',
                              ma2_type='sma',
                              ma1_period=10,
                              ma2_period=20)
            _habs_df = _habs.table()
            results = results.join(_habs_df, how='outer')

        if 7 in indicators:
            _ma_change = MAChange(datetime=ha_data_ma['datetime'],
                                  close=ha_data_ma['close'],
                                  function=function,
                                  ma_type='ema',
                                  period=8,
                                  change=1)
            _ma_change_df = _ma_change.table()
            results = results.join(_ma_change_df, how='outer')

        if 8 in indicators:
            _retrace = Retracements(high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    dates=data['datetime'],
                                    function=function)
            _retrace.get_retracements(low=.38, high=.6)
            _retrace_df = _retrace.table()
            results = results.join(_retrace_df, how='outer')

    results = round_float_columns(data=results, digits=2)

    return results.reset_index(drop=False).sort_values('datetime', ascending=False)

#
# #
# #
# # #
# # #
# # # #
# from config import Config
# # from data_status import DataStatus
# from database import Database
# # from market_symbols import MarketSymbols
# from timeseries import TimeSeries
# from app_utilities import make_rangebreaks
# # #
# cfg = Config("../resources/config.txt")
# db_connection = Database(cfg.view_db_location())
# db_connection.check_database()
# #
# the_function = 'TIME_SERIES_DAILY_ADJUSTED'
# the_interval = None
# the_symbol = 'SPY'
#
# query = TimeSeries(con=db_connection,
#                    function=the_function,
#                    symbol=the_symbol,
#                    interval=the_interval)
# query.get_local_data()
# price_data = query.view_data()['prices'].loc[:, ['datetime', 'close', 'open', 'high', 'low']]
# print(price_data)
# # #
# # # new_macd = MACD(
# # #     data=price_data[['datetime', 'close']],
# # #     function=the_function)
# # #
# # # # print(new_macd.__dict__['rate_of_change'])
# # # # new_macd.plot_macd(trace_only=False)
# # # #
# # # new_rsi = RSI(
# # #     data=price_data[['datetime', 'close']],
# # #     function=the_function,
# # #     period=14)
# # # # new_rsi.plot_rsi()
# # # # # print(new_rsi.__dict__['rsi_line'])
# # # #
# new_ha = HeikinAshi(function=the_function, data=price_data)
# # c = new_ha.ha_indicator()
# # # print(len(c))
# # # # print(c)
# # # # new_ha.plot_ha()
# # # # # a = new_ha.get_values()
# # # # # print((a['open'] > a['close']) | (a['open'] < a['close']))
# # # #
# # # new_ma = MovingAverageCrossover(function=the_function,
# # #                                 data=price_data[['datetime', 'close']], ma1_period=10, ma2_period=20)
# # # # print(new_ma.get_indicator().query("indicator == 1"))
# # # # d = new_ma.plot_MAC(trace_only=True)
# # # # print(d)
# # # new_ma.plot_MAC(trace_only=False)
#
# new_machange = MAChange(
#     datetime=price_data.datetime,
#     close=price_data.close,
#     function=the_function,
#     ma_type='ema',
#     period=8,
#     change=1)
# q = new_machange.table()
# print(q)
# d = new_machange.plot_MAChange(trace_only=False)
# print(d)
# #
# # new_maz = MovingAverageZone(
# #     function=the_function,
# #     datetime=price_data.datetime,
# #     open=price_data.open,
# #     close=price_data.close,
# #     ma1_period=5, ma2_period=30)
# # # print(new_ma.get_indicator().query("indicator == 1"))
# # # d = new_ma.plot_MAC(trace_only=True)
# # # print(d)
# # new_maz.plot_MAZ(trace_only=False)
