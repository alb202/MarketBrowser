import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_table

from src.app_utilities import *
from src.indicators import create_indicator_table


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
        print("Begin plotting ..........")
        indicator_data = create_indicator_table(
            data=get_price_data(n_clicks,
                                input_symbol,
                                input_function,
                                input_interval),
            function=input_function)

        return [indicator_data.to_dict('records'),
                [{"name": i, "id": i} for i in indicator_data.columns],
                n_clicks == 0]


def generate_indicator_table():
    """Generate the main plot
    """
    on_color = '#00FF00'
    off_color = '#FF0000'
    price_increase = {'if': {'filter_query': '{close} > {open}', 'column_id': 'close'},
                      'backgroundColor': on_color, 'color': 'black'}
    price_decrease = {'if': {'filter_query': '{close} < {open}', 'column_id': 'close'},
                      'backgroundColor': off_color, 'color': 'black'}
    macd_histogram__on = {'if': {'filter_query': '{macd_histogram} > 0', 'column_id': 'macd_histogram'},
                          'backgroundColor': on_color, 'color': 'black'}
    macd_histogram__off = {'if': {'filter_query': '{macd_histogram} <= 0', 'column_id': 'macd_histogram'},
                           'backgroundColor': off_color, 'color': 'black'}
    macd_trend__on = {'if': {'filter_query': '{macd_trend} > 0', 'column_id': 'macd_trend'},
                      'backgroundColor': on_color, 'color': 'black'}
    macd_trend__off = {'if': {'filter_query': '{macd_trend} < 0', 'column_id': 'macd_trend'},
                       'backgroundColor': off_color, 'color': 'black'}
    macd_crossovers__on = {'if': {'filter_query': '{macd_crossovers} > 0', 'column_id': 'macd_crossovers'},
                           'backgroundColor': on_color, 'color': 'black'}
    macd_crossovers__off = {'if': {'filter_query': '{macd_crossovers} < 0', 'column_id': 'macd_crossovers'},
                            'backgroundColor': off_color, 'color': 'black'}
    rsi_crossover__on = {'if': {'filter_query': '{rsi_crossover} > 0', 'column_id': 'rsi_crossover'},
                         'backgroundColor': on_color, 'color': 'black'}
    rsi_crossover__off = {'if': {'filter_query': '{rsi_crossover} < 0', 'column_id': 'rsi_crossover'},
                          'backgroundColor': off_color, 'color': 'black'}
    ha_indicator__on = {'if': {'filter_query': '{ha_indicator} > 0', 'column_id': 'ha_indicator'},
                        'backgroundColor': on_color, 'color': 'black'}
    ha_indicator__off = {'if': {'filter_query': '{ha_indicator} < 0', 'column_id': 'ha_indicator'},
                         'backgroundColor': off_color, 'color': 'black'}
    mac_indicator__on = {'if': {'filter_query': '{mac_indicator} > 0', 'column_id': 'mac_indicator'},
                         'backgroundColor': on_color, 'color': 'black'}
    mac_indicator__off = {'if': {'filter_query': '{mac_indicator} < 0', 'column_id': 'mac_indicator'},
                          'backgroundColor': off_color, 'color': 'black'}
    mac_positive__on = {'if': {'filter_query': '{mac_ma1} > {mac_ma2}', 'column_id': 'mac_indicator'},
                        'backgroundColor': on_color, 'color': 'black'}
    mac_positive__off = {'if': {'filter_query': '{mac_ma1} < {mac_ma2}', 'column_id': 'mac_indicator'},
                         'backgroundColor': off_color, 'color': 'black'}
    maz_indicator__on = {'if': {'filter_query': '{maz_indicator} > 0', 'column_id': 'maz_indicator'},
                         'backgroundColor': on_color, 'color': 'black'}
    maz_indicator__off = {'if': {'filter_query': '{maz_indicator} < 0', 'column_id': 'maz_indicator'},
                          'backgroundColor': off_color, 'color': 'black'}
    machange_indicator__on = {'if': {'filter_query': '{ma_indicator} > 0', 'column_id': 'ma_indicator'},
                              'backgroundColor': on_color, 'color': 'black'}
    machange_indicator__off = {'if': {'filter_query': '{ma_indicator} < 0', 'column_id': 'ma_indicator'},
                               'backgroundColor': off_color, 'color': 'black'}
    retracements__on = {'if': {'filter_query': '{retracements} > 0', 'column_id': 'retracements'},
                        'backgroundColor': on_color, 'color': 'black'}

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
                                    machange_indicator__on, machange_indicator__on,
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
