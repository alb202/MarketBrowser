import dash_table
import main
import pandas as pd
from dash.dependencies import Output, Input


def register_data_status_callbacks(app):
    @app.callback([Output('data_status_table', 'data'),
                   Output('get_previous_symbols', 'disabled'),
                   Output('data_status_loading_indicator', 'hidden')],
                  [Input('view_data_status', 'n_clicks')])
    def create_data_status_table(n_clicks):
        """Begin plotting the price data
        """
        status_table_data = get_data_status(n_clicks=n_clicks)
        status_table_data['datetime'] = pd.DatetimeIndex(status_table_data['datetime']) \
            .strftime('%Y-%m-%d %H:%M:%S')
        return [status_table_data.to_dict(orient='records'), n_clicks == 0, n_clicks == 0]



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

    return main.main(
        {'function': None,
         'symbol': None,
         'interval': None,
         'config': None,
         'get_symbols': None,
         'data_status': True}).sort_values('symbol')
