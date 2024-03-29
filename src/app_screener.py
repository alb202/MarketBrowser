import copy
import os
from functools import reduce

import dash_core_components as dcc
from dash.dependencies import Output, Input, State

from src.main import *
from src.indicators import *
from src.retracements import *
from src.app_utilities import *

def register_screener_callbacks(app):
    @app.callback(Output('screener_input_symbol', 'value'),
                  [Input('get_tracked_symbols', 'n_clicks')],
                  [State('screener_function_options', 'value')])
    def get_tracked_symbol_by_function(n_clicks, function):
        """Return true if the function is intraday, else false
        """
        if n_clicks == 0:
            return None

        df = main(
            {'function': None,
             'symbol': None,
             'interval': None,
             'config': None,
             'get_all': False,
             'no_return': False,
             'force_update': False,
             'data_status': True,
             'get_symbols': False,
             'refresh': False,
             'no_api': True})

        df = df.loc[df['function'] == FUNCTION_LOOKUP[function], :]
        if function >= 4:
            df = df.loc[df['interval'] == INTERVAL_LOOKUP[function], :]
        return ' '.join(sorted(list(df['symbol'].values)))

    @app.callback([Output('screener_plot', 'figure'),
                   Output('screener_status_indicator', 'hidden')],
                  [Input('submit_screener', 'n_clicks')],
                  [State('screener_input_symbol', 'value'),
                   State('screener_function_options', 'value'),
                   State('screener_indicator_options', 'value'),
                   State('make_screener_fig', 'value')])
    def begin_screener(n_clicks, symbols, function, indicators, make_screener_fig):
        """Begin plotting the price data
        """
        if n_clicks == 0:
            return [screener_plots(dfs={}), True]

        interval = INTERVAL_LOOKUP[function] if function >= 4 else None
        function = FUNCTION_LOOKUP[function]
        symbols = process_symbol_input(symbols)
        print(symbols)

        indicator_dict = dict()
        for symbol in symbols:
            indicator_data = create_indicator_table(
                data=get_price_data(n_clicks=n_clicks,
                                    symbol=symbol,
                                    function=function,
                                    interval=interval,
                                    no_api=True),
                function=function,
                indicators=indicators)
            if len(indicator_data) == 0:
                print(f'Indicator data NOT retrieved for {symbol}')
                continue
            print(f'Indicator data retrieved for {symbol}')
            plot_columns = ['datetime'
                            'macd_crossovers',
                            'macd_trend',
                            'rsi_crossover',
                            'ha_indicator',
                            'mac_indicator',
                            'maz_indicator',
                            'habs_indicator',
                            'ma_indicator',
                            'retracements']
            indicator_data = indicator_data.set_index(['datetime'])
            print('Indicator index set as datetime')
            indicator_data = indicator_data.loc[:, [col for col in indicator_data.columns if col in plot_columns]]
            print('Indicator columns set')
            # print(indicator_data.head(5))
            indicator_dict[symbol] = indicator_data
            print(indicator_data)

        # Save the indicator data to a local tsv file
        save_indicator_data(dfs=copy.deepcopy(indicator_dict))
        if (len(make_screener_fig) > 0) & (len(indicator_dict) > 0):
            return [screener_plots(
                dfs=indicator_dict,
                save_fig=(1 in make_screener_fig),
                show_fig=(2 in make_screener_fig)), n_clicks == 0]
        else:
            return [screener_plots(dfs=dict(), save_fig=False, show_fig=False), n_clicks == 0]


def save_indicator_data(dfs):

    check_dir_exists("../downloads")
    all_dfs = []
    levels = ['symbol', 'indicator']
    now = str(datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    for symbol, df in dfs.items():
        df.columns = [col + '__' + symbol for col in df.columns]
        all_dfs.append(df.head(200))
    final_df = reduce(lambda x, y: pd.merge(x, y, on='datetime', how='outer'), all_dfs).fillna(0)
    final_df = final_df.transpose()
    symbols = [i.split('__')[1] for i in final_df.index]
    final_df.index = [i.split('__')[0] for i in final_df.index]
    final_df['symbol'] = symbols
    final_df = final_df.reset_index(drop=False) \
        .rename(columns={'index': 'indicator'}) \
        .set_index(['symbol', 'indicator'])
    final_df.to_csv(path_or_buf='../downloads/indicators_' + now + '.csv', sep=',', index=True)
    # pd.concat(all_dfs).fillna(0).reset_index(drop=False) \
    #     .to_csv(path_or_buf='../downloads/indicators_' + now + '.csv', sep=',', index=False)


def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def save_indicator_figure(fig):
    check_dir_exists("../downloads")
    now = str(datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    fig.write_image('../downloads/indicator_figure_' + now + '.png')


def generate_screener_symbol_input():
    """Generate the input for creating symbols
    """
    return dcc.Textarea(placeholder='Symbols to screen ...',
                        rows=6,
                        cols=60,
                        value=None,
                        id='screener_input_symbol')


def screener_plots(dfs={}, save_fig=True, show_fig=True):
    """Generate the input for creating symbols
    """
    print(f'Generating screen plots - save_fig: {save_fig}   show_fig: {show_fig}')
    if (len(dfs) > 0) & (save_fig | show_fig):
        cs = [[0.0, 'red'],
              [0.5, 'lightgrey'],
              [1.0, 'green']]

        data_list = [(key, value) for key, value in dfs.items()]
        data_list = sorted(data_list, reverse=True, key=lambda x: (x[1].iloc[0, :].sum(),
                                                                   x[1].iloc[1, :].sum(),
                                                                   x[1].iloc[3, :].sum(),
                                                                   x[1].iloc[4, :].sum(),
                                                                   x[1].iloc[5, :].sum()))
        # if len(data_list) % 2 == 1:
        #     data_list.append(('', pd.DataFrame({})))
        number_of_plots = len(data_list)
        subplot_cols = int(np.ceil(number_of_plots / 50))
        subplot_rows = int(np.ceil(number_of_plots / subplot_cols))

        # subplot_cols = int(np.around(len(data_list) / 50)) if len(data_list) > 50 else int(1)
        print('subplot_cols', subplot_cols)
        # subplot_rows = int(len(data_list) / subplot_cols)
        print('subplot_rows', subplot_rows)
        subplot_titles = [i[0] for i in data_list]
        # print('subplot_titles',  subplot_titles)
        data_list = [i[1] for i in data_list]
        rows_heights = [100] * subplot_rows
        print('rows_heights', rows_heights)
        fig = make_subplots(rows=subplot_rows, cols=subplot_cols,
                            row_heights=rows_heights,
                            shared_xaxes=True,
                            subplot_titles=subplot_titles)
        fig_width = 600 * subplot_cols
        fig_height = 40 * len(data_list[0].columns) * subplot_rows
        print('fig_width: ', fig_width)
        print('fig_height: ', fig_height)
        fig.update_layout(
            width=fig_width,
            height=fig_height)
        fig.update(layout_showlegend=False)
        fig.update(layout_coloraxis_showscale=False)

        for row in range(1, subplot_rows + 1):
            for col in range(1, subplot_cols + 1):
                df = data_list.pop(0)
                df = df.fillna(0).head(200)
                hm = go.Heatmap(
                    {'z': df.transpose().values.tolist(),
                     'y': df.columns.tolist(),
                     'x': df.index.tolist()},
                    # title=dict(text=),
                    colorscale=cs, showscale=False,
                    xgap=1, ygap=1, zmin=-1, zmax=1)
                fig.append_trace(hm, row=row, col=col)
                # fig.update_
                # fig.update_layout(
                #     annotations=[
                #         dict(
                #             x=2, y=2,  # annotation point
                #             xref='x1',
                #             yref='y1',
                #             text='dict Text',
                #             showarrow=True,
                #             arrowhead=7,
                #             ax=10,
                #             ay=70
                #         )
                if not data_list:
                    break
            if not data_list:
                break
            fig.layout.coloraxis.showscale = False
        if save_fig:
            save_indicator_figure(fig)
        if show_fig:
            return fig
    return make_subplots(rows=1, cols=1)


# def generate_indicator_symbol_input():
#     """Generate the input for creating symbols
#     """
#     return dbc.Input(placeholder='Enter a symbol...',
#                      type='text',
#                      value=None,
#                      id='input_indicator_symbol')
#
#
# def generate_indicator_function_dropdown():
#     """Generate the dropdown for selecting a function
#     """
#     return dcc.Dropdown(id='input_indicator_function',
#                         options=[
#                             {'label': 'Intraday', 'value': 'TIME_SERIES_INTRADAY'},
#                             {'label': 'Daily', 'value': 'TIME_SERIES_DAILY_ADJUSTED'},
#                             {'label': 'Weekly', 'value': 'TIME_SERIES_WEEKLY_ADJUSTED'},
#                             {'label': 'Monthly', 'value': 'TIME_SERIES_MONTHLY_ADJUSTED'}
#                         ], multi=False, value="TIME_SERIES_DAILY_ADJUSTED")
#
#
# def generate_indicator_interval_dropdown():
#     """Generate the dropdown for selecting an interval
#     """
#     return dcc.Dropdown(id='input_indicator_interval',
#                         options=[
#                             {'label': '5 Minute', 'value': '5min'},
#                             {'label': '15 Minute', 'value': '15min'},
#                             {'label': '30 Minute', 'value': '30min'},
#                             {'label': '60 Minute', 'value': '60min'}
#                         ], multi=False, value=None, disabled=True, )
