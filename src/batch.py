import dash_core_components as dcc
import dash_table
import main
import pandas as pd
from dash.dependencies import Output, Input, State


def register_batch_callbacks(app):
    @app.callback([Output('status_indicator', 'hidden')],
                  [Input('submit_batch', 'n_clicks')],
                  [State('batch_input_symbol', 'value')])
    def process_batch_request(n_clicks, batch_input_symbol):
        """Begin plotting the price data
        """
        if batch_input_symbol is None:
            print("Symbol list is empty")
            return [True]
        if batch_input_symbol == '':
            print("Symbol list is empty")
            return [True]

        print("Symbol list is NOT empty: ", batch_input_symbol)
        symbol_list = [i.strip(' ').upper() for i in
                       batch_input_symbol.replace('\n',
                                                  ' ').replace('\n', ' ').replace(',', ' ').replace(';', ' ').split(
                           ' ')]
        symbol_list = [i if i != '' else None for i in symbol_list]

        print('symbol_list: ', symbol_list)
        print('n clicks:', n_clicks)
        result = get_batch_data(n_clicks=n_clicks, symbols=symbol_list)
        return [not result]

    @app.callback([Output('data_status_table', 'data'),
                   Output('get_previous', 'disabled'),
                   Output('data_status_loading_indicator', 'hidden')],
                  [Input('view_data_status', 'n_clicks')])
    def create_data_status_table(n_clicks):
        """Begin plotting the price data
        """
        status_table_data = get_data_status(n_clicks=n_clicks)
        status_table_data['datetime'] = pd.DatetimeIndex(status_table_data['datetime']) \
            .strftime('%Y-%m-%d %H:%M:%S')
        # status_table_data['datetime'] = status_table_data['datetime'].astype('datetime64')
        return [status_table_data.to_dict(orient='records'), n_clicks == 0, n_clicks == 0]

    @app.callback(Output('batch_input_symbol', 'value'),
                  [Input('get_previous', 'n_clicks')])
    def get_previous_symbols(n_clicks):
        """Begin plotting the price data
        """
        symbols = ' '.join(list(sorted(set(get_data_status(n_clicks=n_clicks)['symbol'].to_list()))))
        return symbols


def generate_batch_symbol_input():
    """Generate the input for creating symbols
    """
    return dcc.Textarea(placeholder='Enter a symbol on each line ...',
                        rows=6,
                        cols=60,
                        value=None,
                        id='batch_input_symbol')


def generate_data_status_table():
    """Generate the input for creating symbols
    """
    return dash_table.DataTable(
        id='data_status_table',
        data=pd.DataFrame(
            columns=['symbol', 'function', 'interval', 'datetime']).to_dict(orient='records'),
        columns=[
            {'id': 'symbol', 'name': 'symbol'},
            {'id': 'function', 'name': 'function'},
            {'id': 'interval', 'name': 'interval'},
            {'id': 'datetime', 'name': 'datetime'}],
        editable=False, sort_action='native',
        fixed_columns={'headers': True},
        style_table={'height': '600px', 'width': '700px',
                     'overflowY': True, 'overflowX': False},
        style_cell={'textAlign': 'right'},
        style_data_conditional=[{'if': {'row_index': 'odd'},
                                 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)',
                      'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': 'symbol'}, 'width': '20%'},
            {'if': {'column_id': 'function'}, 'width': '30%'},
            {'if': {'column_id': 'interval'}, 'width': '20%'},
            {'if': {'column_id': 'datetime'}, 'width': '35%'}])


def get_data_status(n_clicks):
    """Get the data from main
    """
    if (n_clicks == 0):
        return pd.DataFrame(
            columns=['symbol', 'function', 'interval', 'datetime'])

    results = main.main(
        {'function': None,
         'symbol': None,
         'interval': None,
         'config': None,
         'data_status': True}).sort_values('symbol')
    return (results)


def get_batch_data(n_clicks, symbols):
    """Get the data from main
    """
    if (n_clicks == 0) | (not symbols):
        print('not running')
        return True

    main.main(
        {'function': None,
         'symbol': symbols,
         'interval': None,
         'config': None,
         'get_all': True,
         'no_return': True,
         'data_status': False})
    return False
