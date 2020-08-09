import numpy as np
import plotly.graph_objects as go
from app_utilities import *
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


class Retracements:

    def __init__(self, high, low, close, dates, function, peak_prominance=.04, peak_gap=12):
        self.high = high.values
        self.low = low.values
        self.close = close.values
        self.datetime = dates
        self.function = function
        self.maxima, _ = find_peaks(x=self.high,
                                    distance=peak_gap,
                                    prominence=peak_prominance * self.high)
        self.minima, _ = find_peaks(x=(-1) * self.low,
                                    distance=peak_gap,
                                    prominence=peak_prominance * self.low)
        self.peak_directions = np.array([0.] * len(dates))
        for i in self.maxima:
            self.peak_directions[i] = 1
        for i in self.minima:
            self.peak_directions[i] = -1
        self.peak_heights = np.where(
            self.peak_directions == 0,
            None,
            np.where(self.peak_directions == 1,
                     self.high,
                     self.low))

    def plot_peaks(self, trace_only=False):
        peak_colors = np.where(self.peak_directions == 0, 'white',
                               np.where(self.peak_directions == 1, 'green', 'red'))
        peak_trace = dict(trace=go.Scatter(
            name='Peaks',
            mode='markers',
            showlegend=False,
            x=self.datetime,
            marker_color=peak_colors,
            y=self.peak_heights))
        if trace_only:
            return dict(indicator=peak_trace)
        price_trace = dict(
            trace=go.Scatter(
                name='Price',
                mode='lines',
                showlegend=False,
                x=self.datetime,
                marker_color='black',
                y=self.close))
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(row=1, col=1, secondary_y=False, **peak_trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **price_trace)
        fig.update_layout(dict(width=1200, height=500,
                               title=f'Price peaks',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text='Price peaks'))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig

    def plot_retracements(self, low=.4, high=.6, peak_find_max=4, retrace_find_max=4, trace_only=False):
        peaks = [(i, j, k, l) for i, (j, k, l) in enumerate(
            zip(self.peak_directions[::-1],
                self.peak_heights[::-1],
                self.datetime[::-1])) if k is not None]
        x_coords = []
        y_coords = []
        for retrace_index, (price_index, peak_direction, price, price_date) in enumerate(peaks):
            peak_found = False
            bottom_found = False
            if (retrace_index < (len(peaks) - 2)) & (peak_direction == -1):
                to_peak = 1
                while (to_peak < retrace_find_max) & (peak_found is False):
                    if peaks[retrace_index + to_peak][1] == 1:
                        peak_found = True
                    else:
                        to_peak += 1
                if peak_found:
                    to_bottom = to_peak + 1
                    while (to_bottom < peak_find_max) & (bottom_found is False):
                        if peaks[retrace_index + to_bottom][1] == -1:
                            bottom_found = True
                        else:
                            to_bottom += 1
                    if bottom_found:
                        peak_range = peaks[retrace_index + to_peak][2] - peaks[retrace_index + to_bottom][2]
                        retrace_lower = peaks[retrace_index + to_bottom][2] + (low * peak_range)
                        retrace_higher = peaks[retrace_index + to_bottom][2] + (high * peak_range)
                        if (peaks[retrace_index][2] >= retrace_lower) & (peaks[retrace_index][2] <= retrace_higher):
                            x_coords.append([peaks[retrace_index][3],
                                             peaks[retrace_index + to_peak][3],
                                             peaks[retrace_index + to_bottom][3],
                                             peaks[retrace_index][3]])
                            y_coords.append([peaks[retrace_index][2],
                                             peaks[retrace_index + to_peak][2],
                                             peaks[retrace_index + to_bottom][2],
                                             peaks[retrace_index][2]])
        retracement_traces = []
        for x, y in zip(x_coords, y_coords):
            retracement_traces.append(dict(trace=go.Scatter(
                x=x,
                y=y,
                fill="toself",
                showlegend=False,
                line={'color': 'lightblue'},
                fillcolor='blue',
                opacity=.5)))
        if trace_only:
            return retracement_traces
        price_trace = dict(
            trace=go.Scatter(
                name='Price',
                mode='lines',
                showlegend=False,
                x=self.datetime,
                marker_color='black',
                y=self.close))
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        for trace in retracement_traces:
            fig.add_trace(row=1, col=1, secondary_y=False, **trace)
        fig.add_trace(row=1, col=1, secondary_y=False, **price_trace)
        fig.update_layout(dict(width=1200, height=500,
                               title=f'50% Retracements',
                               xaxis1=dict(type="date", rangeslider=dict(visible=False)),
                               yaxis1=dict(title=dict(text='Retracements'))))
        fig.update_xaxes(rangebreaks=make_rangebreaks(self.function))
        fig.show()
        return fig

#
#
# cfg = Config("../resources/config.txt")
# db_connection = Database(cfg.view_db_location())
# db_connection.check_database()
# the_function = 'TIME_SERIES_DAILY_ADJUSTED'
# the_interval = None
# the_symbol = 'AMAT'
# peak_range = 10
# query = TimeSeries(con=db_connection,
#                    function=the_function,
#                    symbol=the_symbol,
#                    interval=the_interval)
# query.get_local_data()
# price_data = query.view_data()['prices'].loc[:, ['datetime', 'close', 'open', 'high', 'low']]
#
# a = Retracements(high=price_data['high'],
#                  low=price_data['low'],
#                  close=price_data['close'],
#                  dates=price_data['datetime'],
#                  function=the_function)
# # print(a.plot_peaks(trace_only=True))
# a.plot_retracements()
