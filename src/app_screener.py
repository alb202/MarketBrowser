import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table
import main
import pandas as pd
from app_utilities import *
from dash.dependencies import Output, Input, State
from indicators import *
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
             'get_symbols': False})
        df = df.loc[df['function'] == FUNCTION_LOOKUP[function], :]
        if function >= 4:
            df = df.loc[df['interval'] == INTERVAL_LOOKUP[function], :]
        return ' '.join(sorted(list(df['symbol'].values)))

    @app.callback([Output('screener_table', 'data'),
                   Output('screener_status_indicator', 'hidden')],
                  [Input('submit_screener', 'n_clicks')],
                  [State('screener_input_symbol', 'value'),
                   State('screener_function_options', 'value'),
                   State('screener_indicator_options', 'value')])
    def begin_screener(n_clicks, symbols, functions, indicators):
        """Begin plotting the price data
        """
        if n_clicks == 0:
            return [pd.DataFrame({}).to_dict('records'), True]
        symbols = process_symbol_input(symbols)
        print(symbols)

        return [pd.DataFrame({}).to_dict('records'), True]
        # for symbol in
        # if "INTRADAY" not in input_function:
        #     input_interval = None
        # # price_data =
        # indicator_data = create_indicators(
        #     data=get_price_data(n_clicks, input_symbol, input_function, input_interval),
        #     function=input_function)
        #
        # return [indicator_data.to_dict('records'),
        #         [{"name": i, "id": i} for i in indicator_data.columns],
        #         n_clicks == 0]


def generate_screener_symbol_input():
    """Generate the input for creating symbols
    """
    return dcc.Textarea(placeholder='Symbols to screen ...',
                        rows=6,
                        cols=60,
                        value=None,
                        id='screener_input_symbol')


def generate_screener_table():
    """Generate the input for creating symbols
    """
    return dash_table.DataTable(
        id='screener_table')


def generate_indicator_table():
    """Generate the main plot
    """
    price_increase = {'if': {'filter_query': '{close} > {open}', 'column_id': 'close'},
                      'backgroundColor': '#00FF00', 'color': 'black'}
    price_decrease = {'if': {'filter_query': '{close} < {open}', 'column_id': 'close'},
                      'backgroundColor': '#FF0000', 'color': 'black'}
    macd_histogram__on = {'if': {'filter_query': '{macd_histogram} > 0', 'column_id': 'macd_histogram'},
                          'backgroundColor': '#00FF00', 'color': 'black'}
    macd_histogram__off = {'if': {'filter_query': '{macd_histogram} <= 0', 'column_id': 'macd_histogram'},
                           'backgroundColor': '#FF0000', 'color': 'black'}
    macd_trend__on = {'if': {'filter_query': '{macd_trend} > 0', 'column_id': 'macd_trend'},
                      'backgroundColor': '#00FF00', 'color': 'black'}
    macd_trend__off = {'if': {'filter_query': '{macd_trend} < 0', 'column_id': 'macd_trend'},
                       'backgroundColor': '#FF0000', 'color': 'black'}
    macd_crossovers__on = {'if': {'filter_query': '{macd_crossovers} > 0', 'column_id': 'macd_crossovers'},
                           'backgroundColor': '#00FF00', 'color': 'black'}
    macd_crossovers__off = {'if': {'filter_query': '{macd_crossovers} < 0', 'column_id': 'macd_crossovers'},
                            'backgroundColor': '#FF0000', 'color': 'black'}
    rsi_crossover__on = {'if': {'filter_query': '{rsi_crossover} > 0', 'column_id': 'rsi_crossover'},
                         'backgroundColor': '#00FF00', 'color': 'black'}
    rsi_crossover__off = {'if': {'filter_query': '{rsi_crossover} < 0', 'column_id': 'rsi_crossover'},
                          'backgroundColor': '#FF0000', 'color': 'black'}
    ha_indicator__on = {'if': {'filter_query': '{ha_indicator} > 0', 'column_id': 'ha_indicator'},
                        'backgroundColor': '#00FF00', 'color': 'black'}
    ha_indicator__off = {'if': {'filter_query': '{ha_indicator} < 0', 'column_id': 'ha_indicator'},
                         'backgroundColor': '#FF0000', 'color': 'black'}
    mac_indicator__on = {'if': {'filter_query': '{mac_indicator} > 0', 'column_id': 'mac_indicator'},
                         'backgroundColor': '#00FF00', 'color': 'black'}
    mac_indicator__off = {'if': {'filter_query': '{mac_indicator} < 0', 'column_id': 'mac_indicator'},
                          'backgroundColor': '#FF0000', 'color': 'black'}
    mac_positive__on = {'if': {'filter_query': '{mac_ma1} > {mac_ma2}', 'column_id': 'mac_indicator'},
                        'backgroundColor': '#b3ffb8', 'color': 'black'}
    mac_positive__off = {'if': {'filter_query': '{mac_ma1} < {mac_ma2}', 'column_id': 'mac_indicator'},
                         'backgroundColor': '#e09292', 'color': 'black'}
    maz_indicator__on = {'if': {'filter_query': '{maz_indicator} > 0', 'column_id': 'maz_indicator'},
                         'backgroundColor': '#00FF00', 'color': 'black'}
    maz_indicator__off = {'if': {'filter_query': '{maz_indicator} < 0', 'column_id': 'maz_indicator'},
                          'backgroundColor': '#FF0000', 'color': 'black'}
    retracements__on = {'if': {'filter_query': '{retracements} > 0', 'column_id': 'retracements'},
                        'backgroundColor': '#00FF00', 'color': 'black'}
    return dash_table.DataTable(id='indicator_table',
                                style_data_conditional=[
                                    price_increase, price_decrease,
                                    macd_histogram__on, macd_histogram__off,
                                    macd_crossovers__on, macd_crossovers__off,
                                    macd_trend__on, macd_trend__off,
                                    rsi_crossover__on, rsi_crossover__off,
                                    ha_indicator__on, ha_indicator__off,
                                    mac_positive__on, mac_positive__off,
                                    mac_indicator__on, mac_indicator__off,
                                    maz_indicator__on, maz_indicator__off,
                                    retracements__on])


def generate_indicator_symbol_input():
    """Generate the input for creating symbols
    """
    return dbc.Input(placeholder='Enter a symbol...',
                     type='text',
                     value=None,
                     id='input_indicator_symbol')


def generate_indicator_function_dropdown():
    """Generate the dropdown for selecting a function
    """
    return dcc.Dropdown(id='input_indicator_function',
                        options=[
                            {'label': 'Intraday', 'value': 'TIME_SERIES_INTRADAY'},
                            {'label': 'Daily', 'value': 'TIME_SERIES_DAILY_ADJUSTED'},
                            {'label': 'Weekly', 'value': 'TIME_SERIES_WEEKLY_ADJUSTED'},
                            {'label': 'Monthly', 'value': 'TIME_SERIES_MONTHLY_ADJUSTED'}
                        ], multi=False, value="TIME_SERIES_DAILY_ADJUSTED")


def generate_indicator_interval_dropdown():
    """Generate the dropdown for selecting an interval
    """
    return dcc.Dropdown(id='input_indicator_interval',
                        options=[
                            {'label': '5 Minute', 'value': '5min'},
                            {'label': '15 Minute', 'value': '15min'},
                            {'label': '30 Minute', 'value': '30min'},
                            {'label': '60 Minute', 'value': '60min'}
                        ], multi=False, value=None, disabled=True, )
