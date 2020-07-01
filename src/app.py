"""The Dash app

Module running the Dash app

"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from batch import *
from graphing import *

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(suppress_callback_exceptions=True)
register_graphing_callbacks(app)
register_batch_callbacks(app)

app.layout = html.Div(
    dcc.Tabs([
        dcc.Tab(label='Graphing',
                children=[
                    html.H4(
                        id='graphing_header',
                        children='MarketBrowser v0.1'),
                    html.Div(id='div_select1',
                             children=[
                                 generate_symbol_input(),
                                 generate_function_dropdown(),
                                 generate_interval_dropdown(),
                                 generate_show_dividend_checkbox(),
                                 html.Button('View plot ...', id='submit_val', n_clicks=0)],
                             style={'width': '150pt',
                                    'display': 'inline-block',
                                    'vertical-align': 'middle'}),
                    html.Div(children=generate_plot())]),
        dcc.Tab(label='Database', children=[
            html.H4(
                id='batch_header',
                children='MarketDownloader v0.1'),
            html.Div(id='batch_download',
                     children=[
                         generate_batch_symbol_input(),
                         html.Button('Get all symbols ...', id='submit_batch', n_clicks=0)],
                     style={'width': '150pt',
                            'display': 'inline-block',
                            'vertical-align': 'middle'}),
            html.Div(children=html.P('Complete!', hidden=True, id='status_indicator'))])]))

if __name__ == '__main__':
    app.run_server(debug=True)
