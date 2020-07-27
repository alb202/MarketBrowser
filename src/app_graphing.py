import dash_bootstrap_components as dbc
import dash_core_components as dcc
import main
import pandas as pd
import plotly.graph_objects as go
from app_utilities import *
from dash.dependencies import Output, Input, State
from indicators import *
from plotly.subplots import make_subplots


def register_graphing_callbacks(app):
    @app.callback(Output('input_interval', 'disabled'),
                  [Input('input_function', 'value')])
    def set_dropdown_enabled_state(function):
        """Return true if the function is intraday, else false
        """
        if not function:
            return True
        return 'INTRADAY' not in function

    @app.callback([Output('time_series_plot', 'figure'),
                   Output('plot_status_indicator', 'hidden')],
                  [Input('submit_val', 'n_clicks')],
                  [State('input_symbol', 'value'),
                   State('input_function', 'value'),
                   State('input_interval', 'value'),
                   State('show_dividends', 'value')])
    def begin_plotting(n_clicks, input_symbol, input_function, input_interval, show_dividends):
        """Begin plotting the price data
        """
        if "INTRADAY" not in input_function:
            input_interval = None
        stock_data = get_data(n_clicks, input_symbol, input_function, input_interval)

        params = dict(
            show_dividends=show_dividends,
            nrows=4, ncols=1,
            # nrows=5, ncols=1,
            row_heights=[.50, .1, .2, .2],
            # row_heights=[.22, .12, .22, .22, .22],
            vertical_spacing=.02)
        return [create_main_graph(data=stock_data, symbol=input_symbol,
                                  function=input_function, params=params),
                n_clicks == 0]


def generate_symbol_input():
    """Generate the input for creating symbols
    """
    return dbc.Input(placeholder='Enter a symbol...',
                     type='text',
                     value=None,
                     id='input_symbol')


def generate_function_dropdown():
    """Generate the dropdown for selecting a function
    """
    return dcc.Dropdown(id='input_function',
                        options=[
                            {'label': 'Intraday', 'value': 'TIME_SERIES_INTRADAY'},
                            {'label': 'Daily', 'value': 'TIME_SERIES_DAILY_ADJUSTED'},
                            {'label': 'Weekly', 'value': 'TIME_SERIES_WEEKLY_ADJUSTED'},
                            {'label': 'Monthly', 'value': 'TIME_SERIES_MONTHLY_ADJUSTED'}
                        ], multi=False, value="TIME_SERIES_DAILY_ADJUSTED")


def generate_interval_dropdown():
    """Generate the dropdown for selecting an interval
    """
    return dcc.Dropdown(id='input_interval',
                        options=[
                            {'label': '5 Minute', 'value': '5min'},
                            {'label': '15 Minute', 'value': '15min'},
                            {'label': '30 Minute', 'value': '30min'},
                            {'label': '60 Minute', 'value': '60min'}
                        ], multi=False, value=None, disabled=True, )


def generate_show_dividend_checkbox():
    """Generate the checkbox for showing dividends
    """
    return dbc.Checklist(id='show_dividends',
                         options=[
                             {'label': 'Show dividends', 'value': 'yes'}],
                         value=[],
                         inline=False,
                         switch=True)


def generate_plot():
    """Generate the main plot
    """
    return dcc.Graph(id='time_series_plot')


def get_data(n_clicks, symbol, function, interval):
    """Get the data from main
    """
    if (('INTRADAY' in function) & (interval is None)) | \
            (n_clicks == 0) | \
            (symbol is None) | \
            (function is None):
        return {'prices': pd.DataFrame({'datetime': [],
                                        'open': [],
                                        'high': [],
                                        'low': [],
                                        'close': [],
                                        'volume': []}),
                'dividends': pd.DataFrame({'symbol': [],
                                           'datetime': [],
                                           'dividend_amount': []})}

    return main.main(
        {'function': [function],
         'symbol': [symbol.upper()],
         'interval': [interval],
         'config': None,
         'get_all': False,
         'no_return': False,
         'data_status': False,
         'get_symbols': False})





def create_main_graph(data, symbol, function, params):
    """Plot the data on the main graph
    """
    fig = make_subplots(
        rows=params['nrows'],
        cols=params['ncols'],
        row_heights=params['row_heights'],
        shared_xaxes=True,
        vertical_spacing=params['vertical_spacing'])
    fig.update_xaxes(rangebreaks=make_rangebreaks(function))
    fig.add_trace(row=1, col=1,
                  trace=go.Candlestick(name='candlestick',
                                       showlegend=False,
                                       x=[] if len(data['prices']['datetime']) == 0 else data['prices'][
                                           'datetime'].dt.strftime('%d/%m/%y %-H:%M'),
                                       open=data['prices']['open'],
                                       high=data['prices']['high'],
                                       low=data['prices']['low'],
                                       close=data['prices']['close']))
    volume_color = data['prices']['close'].astype('float') - data['prices']['open'].astype('float')
    volume_color.loc[volume_color < 0] = -1
    volume_color.loc[volume_color > 0] = 1
    volume_color.replace({-1: '#FF0000', 0: '#C0C0C0', 1: '#009900'}, inplace=True)
    fig.add_trace(row=2, col=1,
                  trace=go.Bar(
                      name='volume',
                      showlegend=False,
                      x=data['prices']['datetime'],
                      marker_color=list(volume_color),
                      y=data['prices']['volume']))
    if len(data['prices']['close'] > 0):
        _macd = MACD(data=data['prices'][['datetime', 'close']], function=function)
        _macd_plot = _macd.plot_macd(trace_only=True)
        fig.add_trace(row=3, col=1, **_macd_plot['histogram'])
        fig.add_trace(row=3, col=1, **_macd_plot['signal'])
        fig.add_trace(row=3, col=1, **_macd_plot['macd'])
        _rsi = RSI(data=data['prices'][['datetime', 'close']], function=function)
        _rsi_plot = _rsi.plot_rsi(trace_only=True)
        fig.add_trace(row=4, col=1, **_rsi_plot['rsi'])
        fig.add_shape(row=4, col=1, **_rsi_plot['top_line'])
        fig.add_shape(row=4, col=1, **_rsi_plot['bottom_line'])
        _ha = HeikinAshi(data=data['prices'][['datetime', 'open', 'high', 'low', 'close']],
                         function=function)
        _ha_plot = _ha.plot_ha(trace_only=True, show_indicators=True)
        _ha_data = _ha.get_values()
        # fig.add_trace(row=5, col=1, **_ha_plot['ha'])
        # fig['layout']['xaxis1'].update(rangeslider={'visible': False})
        fig.add_trace(row=1, col=1, **_ha_plot['indicator'])
        _mac = MovingAverageCrossover(data=pd.DataFrame({'datetime': _ha_data['datetime'],
                                                         'close': _ha_data['close']}),
                                      function=function,
                                      ma1_period=10,
                                      ma2_period=20)
        _mac_plot = _mac.plot_MAC(trace_only=True)
        fig.add_trace(row=1, col=1, **_mac_plot['ma1'])
        fig.add_trace(row=1, col=1, **_mac_plot['ma2'])
        fig.add_trace(row=1, col=1, **_mac_plot['crossover'])
    else:
        fig.add_trace(row=3, col=1, trace=go.Scatter(name='MACD', x=[], y=[]))
        fig.add_trace(row=4, col=1, trace=go.Scatter(name='RSI', x=[], y=[]))
        # fig.add_trace(row=5, col=1, trace=go.Scatter(name='HA', x=[], y=[]))
        # fig.add_trace(row=6, col=1, trace=go.Scatter(name='MAC', x=[], y=[]))
    fig.update_layout(get_layout_params(symbol=symbol))
    if 'yes' in params['show_dividends']:
        add_dividends_to_plot(fig=fig, data=data)
    return fig


def add_dividends_to_plot(fig, data):
    """Add dividend data to plot
    """
    dividend_y = data['dividends'].merge(
        data['prices'],
        how='left',
        on='datetime')[['datetime', 'dividend_amount', 'high']] \
        .drop_duplicates().reset_index(drop=True)
    for index, row in dividend_y.iterrows():
        fig.add_annotation(
            x=row['datetime'],
            y=row['high'],
            text=str('$'),
            hovertext=str('$') + str(row['dividend_amount']),
            font=dict(family="Courier New, monospace", size=16, color="#228B22"))
    return fig
