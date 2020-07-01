import dash_core_components as dcc
import main

from dash.dependencies import Output, Input, State


def register_batch_callbacks(app):
    @app.callback(Output('status_indicator', 'hidden'),
                  [Input('submit_batch', 'n_clicks')],
                  [State('batch_input_symbol', 'value')])
    def process_batch_request(n_clicks, batch_input_symbol):
        """Begin plotting the price data
        """
        if batch_input_symbol is not None:
            symbol_list = [i.strip(' ').upper() for i in
                           batch_input_symbol.replace('\n',
                                                      ' ').replace('\n', ' ').replace(',', ' ').replace(';', ' ').split(
                               ' ')]
            symbol_list = [i if i != '' else None for i in symbol_list]
        else:
            symbol_list = []
        print('symbol_list: ', symbol_list)
        result = get_batch_data(n_clicks=n_clicks, symbols=symbol_list)
        return result


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
         'no_return': True})
    return False
