import dash_table
import main
import numpy as np
import pandas as pd
from app_data_status import get_data_status
from dash.dependencies import Output, Input, State
from dash_table.Format import Format

# from dash_table.Format import Format

COLS = ['symbol', 'name', 'type', 'sector', 'industry', 'marketCap', 'sharesOutstanding', 'float']
COLS_NAMES = [
    {'id': 'symbol', 'name': 'symbol'},
    {'id': 'name', 'name': 'name'},
    {'id': 'type', 'name': 'type'},
    {'id': 'sector', 'name': 'sector'},
    {'id': 'industry', 'name': 'industry'},
    {'id': 'marketCap', 'name': 'marketCap',
     'type': 'numeric', 'format': Format(group=',')},
    {'id': 'sharesOutstanding', 'name': 'sharesOutstanding',
     'type': 'numeric', 'format': Format(group=',')},
    {'id': 'float', 'name': 'float',
     'type': 'numeric', 'format': Format(group=',')}]


def register_market_symbol_callbacks(app):
    @app.callback([Output('market_symbol_table', 'data'),
                   Output('market_symbol_table', 'selected_rows'),
                   Output('get_selected_symbols', 'disabled'),
                   Output('market_symbol_loading_indicator', 'hidden')],
                  [Input('view_market_symbols', 'n_clicks')],
                  [State('view_tracked_symbols', 'value'),
                   State('refresh_market_symbols', 'value')])
    def create_market_symbol_table(n_clicks, view_tracked_only, refresh_market_symbols):
        """Begin plotting the price data
        """
        refresh_market_symbols = True if 'yes' in refresh_market_symbols else False
        symbol_table_data = get_market_symbols(n_clicks=n_clicks, refresh=refresh_market_symbols)
        if view_tracked_only:
            data_status = get_data_status(n_clicks)[['symbol']].drop_duplicates().reset_index(drop=True)
            symbol_table_data = symbol_table_data.merge(data_status, how='inner', on='symbol')
        return [symbol_table_data.to_dict(orient='records'),
                np.arange(len(symbol_table_data)),
                n_clicks == 0, n_clicks == 0]

def generate_market_symbol_table():
    """Generate the input for creating symbols
    """
    return dash_table.DataTable(
        id='market_symbol_table',
        data=pd.DataFrame(columns=COLS).to_dict(orient='records'),
        columns=COLS_NAMES,
        editable=False,
        row_selectable='multi',
        column_selectable=None,
        sort_action='native',
        page_action="native",
        selected_rows=None,
        page_current=0,
        page_size=10,
        filter_action='native',
        style_table={'height': '450px',
                     'width': '800px',
                     'overflowY': True,
                     'overflowX': False},
        style_cell={'textAlign': 'right'},
        style_data_conditional=[{'if': {'row_index': 'odd'},
                                 'backgroundColor': 'rgb(248, 248, 248)'}],
        style_header={'backgroundColor': 'rgb(230, 230, 230)',
                      'fontWeight': 'bold', 'textAlign': 'center'})


def get_market_symbols(n_clicks, refresh):
    """Get the data from main
    """
    if (n_clicks == 0):
        return pd.DataFrame(columns=COLS)

    return main.main(
        {'function': None,
         'symbol': None,
         'interval': None,
         'config': None,
         'get_all': None,
         'get_symbols': True,
         'refresh': refresh,
         'no_return': None,
         'data_status': None})
