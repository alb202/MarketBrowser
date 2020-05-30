"""The Dash app

Module running the Dash app

"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import main
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Output, Input, State
from plotly.subplots import make_subplots

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(suppress_callback_exceptions=True)


def generate_symbol_input():
    """Generate the input for creating symbols
    """
    return dcc.Input(placeholder='Enter a symbol...',
                     type='text',
                     value=None,
                     id='input_symbol')


def generate_function_dropdown():
    """Generate the dropdown for selecting a function
    """
    return dcc.Dropdown(id='input_function',
                        options=[
                            {'label': 'Intraday', 'value': 'TIME_SERIES_INTRADAY'},
                            {'label': 'Daily', 'value': 'TIME_SERIES_DAILY_ADJUSTED'},
                            {'label': 'Weekly', 'value': 'TIME_SERIES_WEEKLY_ADJUSTED'},
                            {'label': 'Monthly', 'value': 'TIME_SERIES_MONTHLY_ADJUSTED'}
                        ], multi=False, value="TIME_SERIES_INTRADAY")


def generate_interval_dropdown():
    """Generate the dropdown for selecting an interval
    """
    return dcc.Dropdown(id='input_interval',
                        options=[
                            {'label': '5 Minute', 'value': '5min'},
                            {'label': '15 Minute', 'value': '15min'},
                            {'label': '30 Minute', 'value': '30min'},
                            {'label': '60 Minute', 'value': '60min'}
                        ], multi=False, value=None, disabled=True, )


def generate_show_dividend_checkbox():
    """Generate the checkbox for showing dividends
    """
    return dcc.Checklist(id='show_dividends',
                         options=[{'label': 'Show dividends', 'value': 'yes'}],
                         value=[])


@app.callback(Output('input_interval', 'disabled'),
              [Input('input_function', 'value')])
def set_dropdown_enabled_state(function):
    """Return true if the function is intraday, else false
    """
    return 'INTRADAY' not in function


def generate_plot():
    """Generate the main plot
    """
    return dcc.Graph(id='time_series_plot')


def get_data(n_clicks, symbol, function, interval):
    """Get the data from main
    """
    if (('INTRADAY' in function) & (interval is None)) | \
            (n_clicks == 0) | \
            (symbol is None) | \
            (function is None):
        return {'prices': pd.DataFrame({'datetime': [],
                                        'open': [],
                                        'high': [],
                                        'low': [],
                                        'close': [],
                                        'volume': []}),
                'dividends': pd.DataFrame({'symbol': [],
                                           'datetime': [],
                                           'dividend_amount': []})}

    return main.main(
        {'function': function,
         'symbol': symbol.upper(),
         'interval': interval})


def make_rangebreaks(function):
    """Set the range breaks for x axis
    """
    if 'INTRADAY' in function:
        return [
            dict(bounds=["sat", "mon"]),  # hide weekends
            # dict(values=["2015-12-25", "2016-01-01"]),
            dict(pattern='hour', bounds=[16, 9.5])  # hide Christmas and New Year's
        ]

    if 'DAILY' in function:
        return [
            dict(bounds=["sat", "mon"])  # ,  # hide weekends
            # dict(values=["2015-12-25", "2016-01-01"])
        ]
    return None


@app.callback(
    Output('time_series_plot', 'figure'),
    [Input('submit_val', 'n_clicks')],
    [State('input_symbol', 'value'),
     State('input_function', 'value'),
     State('input_interval', 'value'),
     State('show_dividends', 'value')])
def begin_plotting(n_clicks, input_symbol, input_function, input_interval, show_dividends):
    """Begin plotting the price data
    """
    if "INTRADAY" not in input_function:
        input_interval = None
    stock_data = get_data(n_clicks, input_symbol, input_function, input_interval)
    params = dict(
        show_dividends=show_dividends,
        nrows=2, ncols=1,
        row_heights=[.85, .15],
        vertical_spacing=.02)
    return create_main_graph(data=stock_data, symbol=input_symbol,
                             function=input_function, params=params)


def get_layout_params(symbol):
    """Create the layout parameters
    """
    symbol = symbol if symbol is not None else ' '
    layout = dict(
        width=1200,
        height=600,
        title=symbol,
        xaxis1=dict(rangeselector=dict(buttons=list([
            dict(count=5, label="5d", step="day", stepmode="backward"),
            dict(count=15, label="15d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")])), type="date", rangeslider=dict(visible=False)),
        yaxis1=dict(title=dict(text="Price $ - US Dollars")),
        yaxis2=dict(title=dict(text="Volume")))
    return layout


def create_main_graph(data, symbol, function, params):
    """Plot the data on the main graph
    """
    layout_params = get_layout_params(symbol=symbol)
    fig = make_subplots(
        rows=params['nrows'],
        cols=params['ncols'],
        row_heights=params['row_heights'],
        shared_xaxes=True,
        vertical_spacing=params['vertical_spacing'])
    fig.update_xaxes(rangebreaks=make_rangebreaks(function))
    fig.update_layout(layout_params)
    fig.append_trace(row=1, col=1,
                     trace=go.Candlestick(name='candlestick',
                                          showlegend=False,
                                          x=data['prices']['datetime'],
                                          open=data['prices']['open'],
                                          high=data['prices']['high'],
                                          low=data['prices']['low'],
                                          close=data['prices']['close']))
    volume_color = data['prices']['close'].astype('float') - data['prices']['open'].astype('float')
    volume_color.loc[volume_color < 0] = -1
    volume_color.loc[volume_color > 0] = 1
    volume_color.replace({-1: '#FF0000', 0: '#C0C0C0', 1: '#009900'}, inplace=True)
    fig.append_trace(row=2, col=1,
                     trace=go.Bar(
                         name='volume',
                         showlegend=False,
                         x=data['prices']['datetime'],
                         marker_color=list(volume_color),
                         y=data['prices']['volume']))

    if 'yes' in params['show_dividends']:
        add_dividends_to_plot(fig=fig, data=data)
    return fig


def add_dividends_to_plot(fig, data):
    """Add dividend data to plot
    """
    dividend_y = data['dividends'].merge(
        data['prices'],
        how='left',
        on='datetime')[['datetime', 'dividend_amount', 'high']] \
        .drop_duplicates().reset_index(drop=True)
    for index, row in dividend_y.iterrows():
        fig.add_annotation(
            x=row['datetime'],
            y=row['high'],
            text=str('$'),
            hovertext=str('$') + str(row['dividend_amount']),
            font=dict(family="Courier New, monospace", size=16, color="#228B22"))
    return fig


app.layout = html.Div(
    children=[
        html.H4(id='div_header', children='MarketBrowser v0.1'),
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
        html.Div(children=generate_plot())])

if __name__ == '__main__':
    app.run_server(debug=True)
