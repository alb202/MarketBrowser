import dash
import dash_core_components as dcc
import main
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

    @app.callback(Output('batch_input_symbol', 'value'),
                  [Input('get_selected_symbols', 'n_clicks'),
                   Input('get_previous_symbols', 'n_clicks')],
                  [State('market_symbol_table', 'selected_rows'),
                   State('market_symbol_table', 'data'),
                   State('data_status_table', 'data')])
    def get_batch_symbols(allsymbols_click, previoussymbols_click, selected_rows, all_symbols, previous_symbols):
        """Begin plotting the price data
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = None
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id:
            if button_id == "get_previous_symbols":
                symbols = sorted(list(set([i['symbol'] for i in previous_symbols])))
            if button_id == "get_selected_symbols":
                if selected_rows:
                    symbols = sorted(list(set([all_symbols[i]['symbol'] for i in selected_rows])))
                else:
                    symbols = ['']
        else:
            symbols = ['']
        return ' '.join(symbols)


def generate_batch_symbol_input():
    """Generate the input for creating symbols
    """
    return dcc.Textarea(placeholder='Enter a symbol on each line ...',
                        rows=6,
                        cols=60,
                        value=None,
                        id='batch_input_symbol')


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
         'get_symbols': None,
         'data_status': False})
    return False
