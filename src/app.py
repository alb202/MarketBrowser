import dash
import dash_core_components as dcc
import dash_html_components as html
import main
import pandas as pd
import plotly.graph_objects as go
# import plotly.express as px
from dash.dependencies import Output, Input, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(suppress_callback_exceptions=True)


def generate_symbol_input():
    return dcc.Input(placeholder='Enter a symbol...',
                     type='text',
                     value=None,
                     id='input_symbol')


def generate_function_dropdown():
    return dcc.Dropdown(id='input_function',
                        options=[
                            {'label': 'Intraday', 'value': 'TIME_SERIES_INTRADAY'},
                            {'label': 'Daily', 'value': 'TIME_SERIES_DAILY_ADJUSTED'},
                            {'label': 'Weekly', 'value': 'TIME_SERIES_WEEKLY_ADJUSTED'},
                            {'label': 'Monthly', 'value': 'TIME_SERIES_MONTHLY_ADJUSTED'}
                        ], multi=False, value="TIME_SERIES_INTRADAY")


def generate_interval_dropdown():
    return dcc.Dropdown(id='input_interval',
                        options=[
                            {'label': '5 Minute', 'value': '5min'},
                            {'label': '15 Minute', 'value': '15min'},
                            {'label': '30 Minute', 'value': '30min'},
                            {'label': '60 Minute', 'value': '60min'}
                        ], multi=False, value=None, disabled=True, )


@app.callback(Output('input_interval', 'disabled'),
              [Input('input_function', 'value')])
def set_dropdown_enabled_state(function):
    return 'INTRADAY' not in function


def make_figure(dataframe, layout):
    fig = go.Figure(layout=layout,
                    data=[go.Candlestick(x=dataframe['datetime'],
                                         open=dataframe['open'],
                                         high=dataframe['high'],
                                         low=dataframe['low'],
                                         close=dataframe['close'])])
    return fig


def generate_plot():
    return dcc.Graph(id='time_series_plot_0')


def make_df_for_plotting(n_clicks, symbol, function, interval):
    if (('INTRADAY' in function) & (interval is None)) | \
            (n_clicks == 0) | \
            (symbol is None) | \
            (function is None):
        results = pd.DataFrame({'datetime': [], 'open': [],
                                'high': [], 'low': [], 'close': []})
    else:
        results = main.main(
            {'function': function,
             'symbol': symbol.upper(),
             'interval': interval})
    return results


def make_rangebreaks(function):
    if 'INTRADAY' in function:
        return [
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=["2015-12-25", "2016-01-01"]),
            dict(pattern='hour', bounds=[16, 9.5])  # hide Christmas and New Year's
        ]
    elif 'DAILY' in function:
        return [
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=["2015-12-25", "2016-01-01"])
        ]
    else:
        return


@app.callback(
    Output('time_series_plot_0', 'figure'),
    [Input('submit_val', 'n_clicks')],
    [State('input_symbol', 'value'),
     State('input_function', 'value'),
     State('input_interval', 'value')])
def update_output(n_clicks, input_symbol, input_function, input_interval):
    fig = make_figure(
        layout=dict(
            title=input_symbol,
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(text="Time")),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="Price $ - US Dollars")),
            width=1200,
            height=800
        ),
        dataframe=make_df_for_plotting(
            n_clicks, input_symbol, input_function, input_interval))
    fig.update_xaxes(rangebreaks=make_rangebreaks(input_function))
    fig.update_yaxes(rangemode="tozero")

    return fig


app.layout = html.Div(
    children=[
        html.H4(id='div_header', children='MarketBrowser v0.1'),
        html.Div(id='div_select1',
                 children=[
                     generate_symbol_input(),
                     generate_function_dropdown(),
                     generate_interval_dropdown(),
                     html.Button('View plot ...', id='submit_val', n_clicks=0)],
                 style={'width': '150pt',
                        'display': 'inline-block',
                        'vertical-align': 'middle'}),
        html.Div(children=generate_plot())])

if __name__ == '__main__':
    app.run_server(debug=True)
