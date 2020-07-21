import dash_table
import main
import pandas as pd
from dash.dependencies import Output, Input


def register_market_symbol_callbacks(app):
    @app.callback([Output('market_symbol_table', 'data'),
                   Output('get_selected_symbols', 'disabled'),
                   Output('market_symbol_loading_indicator', 'hidden')],
                  [Input('view_market_symbols', 'n_clicks')])
    def create_market_symbol_table(n_clicks):
        """Begin plotting the price data
        """
        symbol_table_data = get_market_symbols(n_clicks=n_clicks)
        return [symbol_table_data.to_dict(orient='records'), n_clicks == 0, n_clicks == 0]


def get_selected_rows():
    # indicies = market_symbol
    pass


def generate_market_symbol_table():
    """Generate the input for creating symbols
    """
    return dash_table.DataTable(
        id='market_symbol_table',
        data=pd.DataFrame(
            columns=['symbol', 'name', 'type']).to_dict(orient='records'),
        columns=[
            {'id': 'symbol', 'name': 'symbol'},
            {'id': 'name', 'name': 'name'},
            {'id': 'type', 'name': 'type'}],
        editable=False,
        row_selectable="multi",
        sort_action='native',
        page_action="native",
        page_current=0,
        page_size=14,
        # filter_action='native',
        # fixed_columns={'headers': True},
        style_table={'height': '450px',
                     'width': '700px',
                     'overflowY': True,
                     'overflowX': False},
        style_cell={'textAlign': 'right'},
        style_data_conditional=[{'if': {'row_index': 'odd'},
                                 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)',
                      'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell_conditional=[
            {'if': {'column_id': 'symbol'}, 'width': '12%'},
            {'if': {'column_id': 'name'}, 'width': '70%'},
            {'if': {'column_id': 'type'}, 'width': '18%'}])


def get_market_symbols(n_clicks):
    """Get the data from main
    """
    if (n_clicks == 0):
        return pd.DataFrame(
            columns=['symbol', 'name', 'type'])

    return main.main(
        {'function': None,
         'symbol': None,
         'interval': None,
         'config': None,
         'get_all': None,
         'get_symbols': True,
         'no_return': None,
         'data_status': None})
