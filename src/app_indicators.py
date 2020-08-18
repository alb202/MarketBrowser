import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table
import main
import pandas as pd
from app_utilities import *
from dash.dependencies import Output, Input, State
from indicators import *
from retracements import *


def register_indicator_callbacks(app):
    @app.callback(Output('input_indicator_interval', 'disabled'),
                  [Input('input_indicator_function', 'value')])
    def set_indicator_dropdown_enabled_state(function):
        """Return true if the function is intraday, else false
        """
        if not function:
            return True
        return 'INTRADAY' not in function

    @app.callback([Output('indicator_table', 'data'),
                   Output('indicator_table', 'columns'),
                   Output('indicator_status_indicator', 'hidden')],
                  [Input('submit_indicator_val', 'n_clicks')],
                  [State('input_indicator_symbol', 'value'),
                   State('input_indicator_function', 'value'),
                   State('input_indicator_interval', 'value')])
    def begin_plotting(n_clicks, input_symbol, input_function, input_interval):
        """Begin plotting the price data
        """
        if "INTRADAY" not in input_function:
            input_interval = None
        # price_data =
        indicator_data = create_indicators(
            data=get_price_data(n_clicks, input_symbol, input_function, input_interval),
            function=input_function)

        return [indicator_data.to_dict('records'),
                [{"name": i, "id": i} for i in indicator_data.columns],
                n_clicks == 0]


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


def get_price_data(n_clicks, symbol, function, interval):
    """Get the data from main
    """
    if (('INTRADAY' in function) & (interval is None)) | \
            (n_clicks == 0) | \
            (symbol is None) | \
            (function is None):
        return pd.DataFrame({'datetime': [],
                             'open': [],
                             'high': [],
                             'low': [],
                             'close': [],
                             'volume': []})
        # return None
    return main.main(
        {'function': [function],
         'symbol': [symbol.upper()],
         'interval': [interval],
         'config': None,
         'get_all': False,
         'no_return': False,
         'data_status': False,
         'get_symbols': False})['prices']


def create_indicators(data, function):
    """Calculate the indicators for the table
    """
    if len(data) == 0:
        # table_columns = ['datetime', 'open', 'high', 'low', 'close', 'macd', 'macd_signal',
        #                  'macd_histogram', 'macd_crossovers', 'macd_trend', 'rsi',
        #                  'rsi_crossover', 'ha_open', 'ha_high', 'ha_low', 'ha_close',
        #                  'ha_indicator', 'mac_indicator', 'mac_ma1', 'mac_ma2', 'maz_indicator',
        #                  'maz_ma1', 'maz_ma2', 'peaks', 'retracements']

        table_columns = ['datetime']
        return pd.DataFrame.from_dict({i: [] for i in table_columns}, orient='columns')

    _macd = MACD(data=data.loc[:, ['datetime', 'close']], function=function)
    _macd_df = _macd.table()
    # print(_macd_df)

    _rsi = RSI(data=data.loc[:, ['datetime', 'close']], function=function)
    _rsi_df = _rsi.table()
    # print(_rsi_df)

    _ha = HeikinAshi(data=data.loc[:, ['datetime', 'open', 'high', 'low', 'close']], function=function)
    _ha_df = _ha.table()
    # print(_ha_df)
    ha_data_mac = _ha_df[['ha_close']].reset_index(drop=False)
    ha_data_maz = _ha_df[['ha_open', 'ha_close']].reset_index(drop=False)
    ha_data_mac.columns = ['datetime', 'close']
    ha_data_maz.columns = ['datetime', 'open', 'close']

    _mac = MovingAverageCrossover(data=ha_data_mac,
                                  function=function,
                                  ma1_period=10,
                                  ma2_period=20)
    _mac_df = _mac.table()
    # print(_mac_df)

    _maz = MovingAverageZone(datetime=ha_data_maz['datetime'],
                             open=ha_data_maz['open'],
                             close=ha_data_maz['close'],
                             function=function)
    _maz_df = _maz.table()
    # print(_maz_df)

    _retrace = Retracements(high=data['high'],
                            low=data['low'],
                            close=data['close'],
                            dates=data['datetime'],
                            function=function)
    _retrace.get_retracements(low=.38, high=.6)
    _retrace_df = _retrace.table()
    # print(_retrace_df)
    results = data.loc[:, ['datetime', 'open', 'high', 'low', 'close']].set_index('datetime').join(_macd_df,
                                                                                                   how='outer')
    results = results.join(_rsi_df, how='outer')
    # results = results.join(_ha_df, how='outer')
    results = results.join(_mac_df, how='outer')
    results = results.join(_maz_df, how='outer')
    results = results.join(_retrace_df, how='outer')

    results = round_float_columns(data=results, digits=2)

    return results.reset_index(drop=False).sort_values('datetime', ascending=False)


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
