import dash_core_components as dcc
import copy
import datetime
import os

import dash_core_components as dcc
import main
import pandas as pd
from app_utilities import *
from dash.dependencies import Output, Input, State
from indicators import *
from plotly.subplots import make_subplots
from retracements import *


def register_screener_callbacks(app):
    @app.callback(Output('screener_input_symbol', 'value'),
                  [Input('get_tracked_symbols', 'n_clicks')],
                  [State('screener_function_options', 'value')])
    def get_tracked_symbol_by_function(n_clicks, function):
        """Return true if the function is intraday, else false
        """
        if n_clicks == 0:
            return None

        df = main.main(
            {'function': None,
             'symbol': None,
             'interval': None,
             'config': None,
             'get_all': False,
             'no_return': False,
             'data_status': True,
             'get_symbols': False,
             'refresh': False,
             'no_api': True})

        df = df.loc[df['function'] == FUNCTION_LOOKUP[function], :]
        if function >= 4:
            df = df.loc[df['interval'] == INTERVAL_LOOKUP[function], :]
        return ' '.join(sorted(list(df['symbol'].values)))

    @app.callback([Output('screener_plot', 'figure'),
                   Output('screener_status_indicator', 'hidden')],
                  [Input('submit_screener', 'n_clicks')],
                  [State('screener_input_symbol', 'value'),
                   State('screener_function_options', 'value'),
                   State('screener_indicator_options', 'value')])
    def begin_screener(n_clicks, symbols, function, indicators):
        """Begin plotting the price data
        """
        if n_clicks == 0:
            return [screener_plots(dfs=[]), True]

        interval = INTERVAL_LOOKUP[function] if function >= 4 else None
        function = FUNCTION_LOOKUP[function]
        symbols = process_symbol_input(symbols)
        print(symbols)

        indicator_dict = dict()
        for symbol in symbols:
            indicator_data = create_indicators(
                data=get_price_data(n_clicks=n_clicks,
                                    symbol=symbol,
                                    function=function,
                                    interval=interval,
                                    no_api=True),
                function=function,
                indicators=indicators)

            plot_columns = ['datetime'
                            'macd_crossovers',
                            'macd_trend',
                            'rsi_crossover',
                            'mac_indicator',
                            'maz_indicator',
                            'retracements']
            indicator_data = indicator_data.set_index(['datetime'])
            indicator_data = indicator_data.loc[:, [col for col in indicator_data.columns if col in plot_columns]]

            # print(indicator_data.head(5))
            indicator_dict[symbol] = indicator_data

        # Save the indicator data to a local tsv file
        save_indicator_data(dfs=copy.deepcopy(indicator_dict))
        return [screener_plots(dfs=indicator_dict), n_clicks == 0]


def save_indicator_data(dfs):
    check_dir_exists("../downloads")
    all_dfs = []
    now = str(datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    for symbol, df in dfs.items():
        df['symbol'] = symbol
        all_dfs.append(df.head(200))
    pd.concat(all_dfs).fillna(0).reset_index(drop=False) \
        .to_csv(path_or_buf='../downloads/indicators_' + now + '.csv', sep='\t', index=False)


def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def save_indicator_figure(fig):
    check_dir_exists("../downloads")
    now = str(datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    fig.write_image('../downloads/indicator_figure_' + now + '.jpg')


def generate_screener_symbol_input():
    """Generate the input for creating symbols
    """
    return dcc.Textarea(placeholder='Symbols to screen ...',
                        rows=6,
                        cols=60,
                        value=None,
                        id='screener_input_symbol')


def screener_plots(dfs=[]):
    """Generate the input for creating symbols
    """
    if not dfs:
        return make_subplots(rows=1, cols=1)
    cs = [[0.0, 'red'],
          [0.5, 'lightgrey'],
          [1.0, 'green']]

    data_list = [(key, value) for key, value in dfs.items()]
    data_list = sorted(data_list, reverse=True, key=lambda x: (x[1].iloc[0, :].sum(),
                                                               x[1].iloc[1, :].sum(),
                                                               x[1].iloc[2, :].sum()))  # ,
    # x[1].iloc[3, :].sum(),
    # x[1].iloc[4, :].sum()))
    subplot_titles = [i[0] for i in data_list]
    data_list = [i[1] for i in data_list]

    fig = make_subplots(rows=len(dfs), cols=1,
                        row_heights=[100] * len(dfs),
                        shared_xaxes=True,
                        subplot_titles=subplot_titles)
    fig.update_layout(
        width=800,
        height=(60 * len(data_list[0].columns)) * len(dfs))
    fig.update(layout_showlegend=False)
    fig.update(layout_coloraxis_showscale=False)

    for i, df in enumerate(data_list):
        df = df.fillna(0).head(200)
        hm = go.Heatmap(
            {'z': df.transpose().values.tolist(),
             'y': df.columns.tolist(),
             'x': df.index.tolist()},
            colorscale=cs, showscale=False, xgap=1, ygap=1, zmin=-1, zmax=1)
        fig.append_trace(hm, row=i + 1, col=1)
    fig.layout.coloraxis.showscale = False
    save_indicator_figure(fig)
    return fig

# def generate_indicator_symbol_input():
#     """Generate the input for creating symbols
#     """
#     return dbc.Input(placeholder='Enter a symbol...',
#                      type='text',
#                      value=None,
#                      id='input_indicator_symbol')
#
#
# def generate_indicator_function_dropdown():
#     """Generate the dropdown for selecting a function
#     """
#     return dcc.Dropdown(id='input_indicator_function',
#                         options=[
#                             {'label': 'Intraday', 'value': 'TIME_SERIES_INTRADAY'},
#                             {'label': 'Daily', 'value': 'TIME_SERIES_DAILY_ADJUSTED'},
#                             {'label': 'Weekly', 'value': 'TIME_SERIES_WEEKLY_ADJUSTED'},
#                             {'label': 'Monthly', 'value': 'TIME_SERIES_MONTHLY_ADJUSTED'}
#                         ], multi=False, value="TIME_SERIES_DAILY_ADJUSTED")
#
#
# def generate_indicator_interval_dropdown():
#     """Generate the dropdown for selecting an interval
#     """
#     return dcc.Dropdown(id='input_indicator_interval',
#                         options=[
#                             {'label': '5 Minute', 'value': '5min'},
#                             {'label': '15 Minute', 'value': '15min'},
#                             {'label': '30 Minute', 'value': '30min'},
#                             {'label': '60 Minute', 'value': '60min'}
#                         ], multi=False, value=None, disabled=True, )
