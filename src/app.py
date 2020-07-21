"""The Dash app

Module running the Dash app

"""

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from app_batch import *
from app_data_status import *
from app_graphing import *
from app_market_symbols import *

app = dash.Dash(
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUMEN])
register_graphing_callbacks(app)
register_batch_callbacks(app)
register_market_symbol_callbacks(app)
register_data_status_callbacks(app)

# Create tab for plotting functions
MarketBrowserTab = dcc.Tab(label='MarketBrowser', children=[
    dbc.Row(no_gutters=False, justify="start", children=[
        dbc.Col(align='baseline', children=[
            dbc.Card(style={"width": "auto"}, children=[
                dbc.CardBody(children=[
                    dbc.Row(align='baseline', no_gutters=False, justify='start', children=[
                        dbc.Col(children=[
                            dbc.Button(children=['View plot ...'], id='submit_val', n_clicks=0)]),
                        dbc.Col(generate_symbol_input()),
                        dbc.Col(generate_function_dropdown()),
                        dbc.Col(generate_interval_dropdown()),
                        dbc.Col(children=[generate_show_dividend_checkbox()]),
                        dbc.Col(children=[
                            dcc.Loading(id="plot_loading", type="default", children=[
                                html.P(hidden=True, id='plot_status_indicator')])])])])])])]),
    dbc.Card(style={"width": 'auto'}, children=[
        dbc.CardBody(children=[generate_plot()])])])

# Tab for data status
DataStatusTab = dcc.Tab(label="Data Status", children=[
    dbc.Card(children=[
        dbc.CardBody(children=[
            dbc.Row(no_gutters=False, justify='center', align='baseline', children=[
                dbc.Col(width=True, children=[
                    dbc.Button('View data status ...', id='view_data_status', n_clicks=0)]),
                dbc.Col(width=True, children=[
                    dcc.Loading(id="status_loading_indicator", type="default", children=[
                        html.P(hidden=True, id='data_status_loading_indicator')])])])])]),
    dbc.Card(children=[
        dbc.CardBody(children=[
            dbc.Row(no_gutters=False, justify='center',
                    align='baseline', children=[
                    generate_data_status_table()])])])])

# Tab for market symbols
MarketSymbolsTab = dcc.Tab(label="Market Symbols", children=[
    dbc.Card(children=[
        dbc.CardBody(children=[
            dbc.Row(no_gutters=False, justify='center', align='baseline', children=[
                dbc.Col(width=True, children=[
                    dbc.Button('View all symbols ...', id='view_market_symbols', n_clicks=0)]),
                dbc.Col(width=True, children=[
                    dcc.Loading(id="market_symbol_indicator", type="default", children=[
                        html.P(hidden=True, id='market_symbol_loading_indicator')])])])])]),
    dbc.Card(children=[
        dbc.CardBody(children=[
            dbc.Row(no_gutters=False, justify='center',
                    align='baseline', children=[
                    generate_market_symbol_table()])])])])

# Create tab for batch downloading
MarketDownloaderTab = dcc.Tab(label='MarketDownloader', children=[
    dbc.Row(children=[
        dbc.Col(children=[
            dbc.CardGroup([
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        dbc.Row(children=[
                            dbc.Card(children=[
                                dbc.CardBody(children=[
                                    dbc.Row(no_gutters=False, justify='center', align='baseline', children=[
                                        dbc.Col(width=True, children=[generate_batch_symbol_input()])])])]),
                            dbc.Card(children=[
                                dbc.CardBody(children=[
                                    dbc.Row(children=[
                                        dbc.Col(children=[
                                            dbc.Button(children=['Update symbols'], id='submit_batch',
                                                       n_clicks=0)]),
                                        dbc.Col(align='center', children=[
                                            dcc.Loading(id="loading_indicator", type="default", children=[
                                                html.P(hidden=True, id='status_indicator')])])]),
                                    dbc.Row(children=html.Hr()),
                                    dbc.Row(children=html.Br()),
                                    dbc.Row(align="middle", children=[
                                        dbc.Button(children=['Get tracked symbols ...'], id='get_previous_symbols',
                                                   n_clicks=0, disabled=True)]),
                                    dbc.Row(children=html.Br()),
                                    dbc.Row(align="middle", children=[
                                        dbc.Button(children=['Get selected symbols ...'], id='get_selected_symbols',
                                                   n_clicks=0, disabled=True)])])])])])])])])])])

app.layout = html.Div(id='main', children=[
    dcc.Tabs(children=[MarketBrowserTab, MarketDownloaderTab, DataStatusTab, MarketSymbolsTab])])

if __name__ == '__main__':
    app.run_server(debug=True)
