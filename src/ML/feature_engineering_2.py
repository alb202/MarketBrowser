import datetime

import numpy as np
import pandas as pd
# from numba import jit

from .. import main
from indicators import create_indicator_table

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
round_digits = 3


@jit
def get_streaks(df, streak_col, start_val=[1], end_val=[0], price_col='close', peak_col='high',
                peak_func=lambda x, y: x >= y):
    streaks = []
    streak_index = [None] * len(df)
    streak_len = [None] * len(df)
    streak_change = [None] * len(df)
    streak_peak = [None] * len(df)
    streak_start = None
    streak_peak_ = None
    price_start = None
    i = 0
    while i < len(df):
        if i % 100 == 0:
            print(i)
        if df.iloc[i][streak_col] in end_val:
            if not streak_start:
                pass
            else:
                streaks.append((streak_start, i - 1))
                streak_len[streak_start] = streak_index[i - 1]
                streak_change[streak_start] = round(df.iloc[i][price_col] / price_start, round_digits)
                streak_peak_ = streak_peak_ if peak_func(streak_peak_, df.iloc[i][price_col]) else df.iloc[i][price_col]
                streak_peak[streak_start] = round(streak_peak_ / price_start, round_digits)
                streak_start = None
                streak_peak_ = None
            streak_len[i] = 0
            streak_index[i] = 0
        elif df.iloc[i][streak_col] in start_val:
            if not streak_start:
                streak_start = i
                price_start = df.iloc[i][price_col]
                streak_peak_ = df.iloc[i][price_col]
                streak_index[i] = 1
            else:
                streak_index[i] = streak_index[i - 1] + 1
                streak_peak_ = streak_peak_ if peak_func(streak_peak_, df.iloc[i][peak_col]) else df.iloc[i][peak_col]
            streak_len[i] = 1
        else:
            if not streak_start:
                pass
            else:
                streak_index[i] = streak_index[i - 1] + 1
                streak_peak_ = streak_peak_ if peak_func(streak_peak_, df.iloc[i][peak_col]) else df.iloc[i][peak_col]
        i += 1
    if streak_start:
        # for z in range(streak_start, len(streak_index)):
        #     streak_index[z] = None
        streak_len[streak_start] = None
        # streaks.append((streak_start, i-1))
        # streak_len[streak_start] = streak_index[-1]
        # streak_change[streak_start] = round(df.iloc[i-1][price_col] / price_start, round_digits)
        # streak_peak[streak_start] = round(streak_peak_ / price_start, round_digits)
    return (streaks, streak_index, streak_len, streak_change, streak_peak)


@jit
def max_of_future_period(s, days=5):
    result = [np.NaN] * len(s)
    for i in range(len(s) - days):
        result[i] = np.max(s[i + 1:i + days + 1])
    return result


@jit
def mean_of_future_period(s, days=5):
    result = [np.NaN] * len(s)
    for i in range(len(s) - days):
        result[i] = np.mean(s[i + 1:i + days + 1])
    return result


@jit
def sum_of_future_period(s, days=1):
    result = [np.NaN] * len(s)
    for i in range(len(s) - days):
        result[i] = np.sum(s[i + 1:i + days + 1])
    return result


@jit
def adjust_ohlc(o_col, h_col, l_col, c_col, ac_col):
    coof = ac_col / c_col
    return pd.DataFrame({
        'open': o_col * coof,
        'high': h_col * coof,
        'low': l_col * coof})


@jit
def generate_week_ids(start_date='1990-01-01'):
    start_day = pd.to_datetime(start_date).weekday()
    df = pd.DataFrame({'datetime': pd.date_range(start=start_date, end=datetime.datetime.now().date())})
    df = df.iloc[7 - start_day:]
    # df['day'] = df['datetime'].dt.day
    # df['week'] = df['datetime'].dt.week
    # df['day_of_week'] = df['datetime'].dt.weekday
    # df['month'] = df['datetime'].dt.month
    # df['year'] = df['datetime'].dt.year
    df['week_id'] = [j for k in [[i] * 7 for i in range(0, int(np.ceil(len(df) / 7)))] for j in k][:len(df)]
    return df.loc[:, ['datetime', 'week_id']]


@jit
def calculate_price_changes(df, prefix='', suffix=''):
    df[prefix + 'change_open' + suffix] = (
        df[prefix + 'open' + suffix].div(df[prefix + 'open' + suffix].shift(1))).round(round_digits)
    df[prefix + 'change_high' + suffix] = (
        df[prefix + 'high' + suffix].div(df[prefix + 'high' + suffix].shift(1))).round(round_digits)
    df[prefix + 'change_low' + suffix] = (df[prefix + 'low' + suffix].div(df[prefix + 'low' + suffix].shift(1))).round(
        round_digits)
    df[prefix + 'change_close' + suffix] = (
        df[prefix + 'close' + suffix].div(df[prefix + 'close' + suffix].shift(1))).round(round_digits)
    df[prefix + 'range' + suffix] = (df[prefix + 'high' + suffix] - df[prefix + 'low' + suffix]).round(round_digits)
    df[prefix + 'rel_range' + suffix] = (df[prefix + 'range' + suffix].div(df[prefix + 'open' + suffix])).round(
        round_digits)
    df[prefix + 'close_range' + suffix] = ((df[prefix + 'close' + suffix] - df[prefix + 'low' + suffix]) / (
                df[prefix + 'high' + suffix] - df[prefix + 'low' + suffix])).round(round_digits)
    df[prefix + 'true_range' + suffix] = ((df[prefix + 'open' + suffix] - df[prefix + 'close' + suffix]).abs() / (
                df[prefix + 'high' + suffix] - df[prefix + 'low' + suffix])).round(round_digits)
    df[prefix + 'true_range_change' + suffix] = (
        df[prefix + 'true_range' + suffix].div(df[prefix + 'true_range' + suffix].shift(1))).round(round_digits)
    df[prefix + 'ohlc4' + suffix] = ((df[prefix + 'open' + suffix] + df[prefix + 'high' + suffix] + df[
        prefix + 'low' + suffix] + df[prefix + 'close' + suffix]) / 4).round(round_digits)
    df[prefix + 'change_intra' + suffix] = (df[prefix + 'close' + suffix].div(df[prefix + 'open' + suffix])).round(
        round_digits)
    df = df.drop(prefix + 'range' + suffix, axis=1)
    return df


interval = None

raw_data = main.main({
    'function': [None],
    'symbol': [None],
    'interval': [None],
    'config': None,
    'get_all': False,
    'no_return': False,
    'get_symbols': True,
    'data_status': False,
    'only_tracked': True,
    'refresh': False,
    'no_api': True}).loc[:, ['symbol', 'type']].drop_duplicates()
# all_symbols = ['PLAY', 'REGN', 'ALNY', 'ACAD', 'AMAT', 'SIMO', 'AAL', 'INO', 'HEXO', 'IGC', 'ACB', 'JMIA', 'PFE',
#                'SNDL', 'MRNA', 'CIDM', 'MTSI', 'SIMO', 'SPY', 'VIX']
# all_symbols = ['ABBV', 'ABT', 'ACAD', 'ADAP', 'ADVM', 'AGIO', 'AIMT', 'ALEC', 'ALKS', 'ALLO', 'ALNY', 'ALXN',
#                'AMGN', 'AMTI', 'ARNA', 'ARVN', 'ARWR', 'ASND', 'ATRA', 'AUPH', 'AZN', 'BBIO', 'BDTX', 'BDX',
#                'BEAM', 'BGNE', 'BIIB', 'BLUE', 'BMRN', 'BMY', 'BNTX', 'BPMC', 'BSX', 'BTAI', 'CBPO', 'CCXI',
#                'CDMOP', 'CGEN', 'CHRS', 'CRSP', 'CRTX', 'CYTK', 'DCPH', 'DHR', 'DNLI', 'DRNA', 'EBS', 'EDIT',
#                'EIDX', 'EPZM', 'ESPR', 'EW', 'EXAS', 'EXEL', 'FATE', 'FBIOP', 'FGEN', 'FOLD', 'GBT', 'GILD',
#                'GLPG', 'GSK', 'HALO', 'HRTX', 'IGC', 'ILMN', 'IMMU', 'IMVT', 'INCY', 'INO', 'INSM', 'IONS',
#                'IOVA', 'ISRG', 'ITCI', 'ITOS', 'JNJ', 'KOD', 'KPTI', 'KURA', 'KYMR', 'LGND', 'LLY', 'LMNX',
#                'MCRB', 'MDT', 'MESO', 'MGNX', 'MNTA', 'MOR', 'MRK', 'MRNA', 'MRSN', 'MRTX', 'MYGN', 'MYOV',
#                'NBIX', 'NGM', 'NKTX', 'NSTG', 'NTLA', 'NVAX', 'NVO', 'NVS', 'PACB', 'PCVX', 'PFE', 'PTCT',
#                'QURE', 'RARE', 'RCKT', 'RCUS', 'REGN', 'REPL', 'RGNX', 'RLAY', 'RNA', 'RYTM', 'SGEN', 'SGMO',
#                'SLS', 'SNY', 'SRNE', 'SRPT', 'STOK', 'SYK', 'TAK', 'TBIO', 'TECH', 'TGTX', 'TMO', 'TRIL', 'TWST',
#                'TXG', 'VCYT', 'VIE', 'VIR', 'VRTX', 'XLRN', 'XNCR', 'ZLAB', 'ZNTL', 'ZTS', 'SPY']
# all_symbols = ['NTLA', 'NVAX', 'NVO', 'NVS', 'PACB', 'PCVX', 'PFE', 'PTCT',
#                'QURE', 'RARE', 'RCKT', 'RCUS', 'REGN', 'REPL', 'RGNX', 'RLAY', 'RNA', 'RYTM', 'SGEN', 'SGMO',
#                'SLS', 'SNY', 'SRNE', 'SRPT', 'STOK', 'SYK', 'TAK', 'TBIO', 'TECH', 'TGTX', 'TMO', 'TRIL', 'TWST',
#                'TXG', 'VCYT', 'VIE', 'VIR', 'VRTX', 'XLRN', 'XNCR', 'ZLAB', 'ZNTL', 'ZTS', 'SPY']
# all_symbols = ['NTLA']
all_symbols = ['AAL']
raw_data = raw_data[raw_data['symbol'].isin(all_symbols)]
raw_data = raw_data[(raw_data['type'] == 'stock')]
# raw_data = raw_data[(raw_data['type'] == 'stock') | (raw_data['type'] == 'etf')]
# print(raw_data)
print(len(raw_data))
day_of_week_index = generate_week_ids()
ts_start_date = '2017-01-01'
number_of_symbols = 100000
minimum_initial_rows = 300
start_position_in_df = 150
number_rows_for_indicators = 45
for symbol, sym_type in zip(raw_data['symbol'][:number_of_symbols], raw_data['type'][:number_of_symbols]):

    ### Get the timeseries data
    ts_1 = {'function': None, 'symbol': [symbol], 'interval': [None],
            'config': None, 'get_all': False, 'no_return': False, 'get_symbols': None,
            'data_status': False, 'no_api': True}
    ts_d = ts_1.copy()
    ts_d['function'] = ['TIME_SERIES_DAILY_ADJUSTED']
    ts_d = main.main(ts_d)['prices'].drop('split_coefficient', axis=1)
    if len(ts_d) < minimum_initial_rows:
        continue
    ts_d = ts_d[ts_d['datetime'] >= ts_start_date]
    adj_d = adjust_ohlc(o_col=ts_d['open'],
                        h_col=ts_d['high'],
                        l_col=ts_d['low'],
                        c_col=ts_d['close'],
                        ac_col=ts_d['adjusted_close'])
    ts_d['open'] = adj_d['open'].round(round_digits)
    ts_d['high'] = adj_d['high'].round(round_digits)
    ts_d['low'] = adj_d['low'].round(round_digits)
    ts_d['close'] = ts_d['adjusted_close'].round(round_digits)
    ts_d = ts_d.drop('adjusted_close', axis=1)

    ts_d['day'] = ts_d['datetime'].dt.day
    ts_d['month'] = ts_d['datetime'].dt.month
    ts_d['year'] = ts_d['datetime'].dt.year
    ts_d = ts_d.merge(day_of_week_index, how='left', on='datetime')

    ###### Generate the weekly and monthly data
    ### Weekly data
    tmp_open = ts_d.sort_values(['datetime'], ascending=True).groupby(['week_id']).head(1).loc[:,
               ['week_id', 'open']].rename(columns={'open': 'open_w'})
    tmp_low = pd.DataFrame(
        ts_d.sort_values(['datetime'], ascending=True).groupby(['week_id'])['low'].expanding().min().reset_index(
            drop=False))['low']
    tmp_high = pd.DataFrame(
        ts_d.sort_values(['datetime'], ascending=True).groupby(['week_id'])['high'].expanding().max().reset_index(
            drop=False))['high']
    tmp_vol = pd.DataFrame(
        ts_d.sort_values(['datetime'], ascending=True).groupby(['week_id'])['volume'].expanding().sum().reset_index(
            drop=False))['volume']
    ts_d = ts_d.merge(right=tmp_open, on=['week_id'], how='left')
    ts_d['close_w'] = ts_d['close']
    ts_d['low_w'] = tmp_low  # ['low_w']
    ts_d['high_w'] = tmp_high  # ['high_w']
    ts_d['volume_w'] = tmp_vol  # ['volume_w']

    ### Monthly data
    tmp_open = ts_d.sort_values(['datetime'], ascending=True).groupby(['year', 'month']).head(1).loc[:,
               ['month', 'year', 'open']].rename(columns={'open': 'open_m'})
    tmp_low = pd.DataFrame(
        ts_d.sort_values(['datetime'], ascending=True).groupby(['year', 'month'])['low'].expanding().min().reset_index(
            drop=False))['low']
    tmp_high = pd.DataFrame(
        ts_d.sort_values(['datetime'], ascending=True).groupby(['year', 'month'])['high'].expanding().max().reset_index(
            drop=False))['high']
    tmp_vol = pd.DataFrame(ts_d.sort_values(['datetime'], ascending=True).groupby(['year', 'month'])[
                               'volume'].expanding().sum().reset_index(drop=False))['volume']
    ts_d = ts_d.merge(right=tmp_open, on=['month', 'year'], how='left')
    ts_d['close_m'] = ts_d['close']
    ts_d['low_m'] = tmp_low  # ['low_m']
    ts_d['high_m'] = tmp_high  # ['high_m']
    ts_d['volume_m'] = tmp_vol  # ['volume_m']
    #
    print(ts_d)
    print(list(ts_d.columns))
    # break
    indicator_cols = ['open', 'high', 'low', 'close', 'macd', 'macd_signal', 'macd_histogram',
                      'macd_crossovers', 'macd_trend', 'rsi', 'rsi_crossover', 'ha_open', 'ha_high', 'ha_low',
                      'ha_close', 'ha_trend', 'ha_bottom', 'ha_top', 'ha_indicator', 'mac_indicator', 'maz_indicator',
                      'habs_indicator', 'habs_crossover', 'ma_indicator']
    ### Create daily indicators
    ts_d_indicators = create_indicator_table(
        indicators=[1, 2, 3, 4, 5, 6, 7],
        data=ts_d,
        function='TIME_SERIES_DAILY_ADJUSTED')
    # print(len(ts_d_indicators))
    # print(ts_d_indicators.head(5))
    # print(list(ts_d_indicators.columns))
    # break
    ts_d_indicators = ts_d_indicators.loc[:,
                      ['datetime'] + indicator_cols].rename(columns={i: i + '_d' for i in indicator_cols}).sort_values(
        ['datetime'], ascending=True)
    ts_d_indicators = calculate_price_changes(ts_d_indicators, prefix='', suffix='_d')
    ts_d_indicators = calculate_price_changes(ts_d_indicators, prefix='ha_', suffix='_d')

    # print(len(ts_d_indicators))
    # print(ts_d_indicators.head(5))
    # print(list(ts_d_indicators.columns))
    # break

    ### Create lists to hold weekly and monthly rows
    ts_w_ = []
    ts_m_ = []

    ### Get weekly and monthly indicators
    for i in np.arange(100, len(ts_d) + 1):
        df = ts_d.head(i).tail(300)
        df_w = df.sort_values(['datetime'], ascending=True).groupby(['week_id']).tail(1).loc[:,
               ['datetime', 'open_w', 'high_w', 'low_w', 'close_w']].reset_index(drop=True).tail(
            number_rows_for_indicators).reset_index(
            drop=True).rename(columns={'open_w': 'open', 'high_w': 'high', 'low_w': 'low', 'close_w': 'close'})

        df_m = df.sort_values(['datetime'], ascending=True).groupby(['year', 'month']).tail(1).loc[:,
               ['datetime', 'open_m', 'high_m', 'low_m', 'close_m']].reset_index(drop=True).tail(
            number_rows_for_indicators).reset_index(
            drop=True).rename(columns={'open_m': 'open', 'high_m': 'high', 'low_m': 'low', 'close_m': 'close'})
        # print('f_m', len(df_m))
        # break
        ind_tbl_w = create_indicator_table(
            indicators=[1, 2, 3, 4, 5, 6, 7],
            data=df_w,
            function='TIME_SERIES_WEEKLY_ADJUSTED').sort_values(['datetime'], ascending=True)

        ind_tbl_m = create_indicator_table(
            indicators=[1, 2, 3, 4, 5, 6, 7],
            data=df_m,
            function='TIME_SERIES_MONTHLY_ADJUSTED').sort_values(['datetime'], ascending=True)
        print(list(ind_tbl_w.columns))
        ind_tbl_w = ind_tbl_w.loc[:, ['datetime'] + indicator_cols].rename(
            columns={i: i + '_w' for i in indicator_cols})
        ind_tbl_m = ind_tbl_m.loc[:, ['datetime'] + indicator_cols].rename(
            columns={i: i + '_m' for i in indicator_cols})

        ind_tbl_w = calculate_price_changes(ind_tbl_w, prefix='', suffix='_w')
        ind_tbl_w = calculate_price_changes(ind_tbl_w, prefix='ha_', suffix='_w')
        ind_tbl_m = calculate_price_changes(ind_tbl_m, prefix='', suffix='_m')
        ind_tbl_m = calculate_price_changes(ind_tbl_m, prefix='ha_', suffix='_m')

        ts_w_.append(ind_tbl_w.tail(1))
        ts_m_.append(ind_tbl_m.tail(1))
    # break
    ts_w_indicators = pd.concat(ts_w_, axis=0).drop_duplicates().reset_index(drop=True)
    # ts_w_indicators = ts_w_indicators.loc[:,
    # ['datetime']+indicator_cols].rename(columns={i: i+'_w' for i in indicator_cols})

    ts_m_indicators = pd.concat(ts_m_, axis=0).drop_duplicates().reset_index(drop=True)
    # ts_m_indicators = ts_m_indicators.loc[:,
    #                   ['datetime']+indicator_cols].rename(columns={i: i+'_m' for i in indicator_cols})

    ts_d_indicators = ts_d_indicators.set_index('datetime').sort_index()
    ts_w_indicators = ts_w_indicators.set_index('datetime').sort_index()
    ts_m_indicators = ts_m_indicators.set_index('datetime').sort_index()

    print(len(ts_d_indicators))
    print(ts_d_indicators.head(5))
    print(list(ts_d_indicators.columns))

    print(len(ts_w_indicators))
    print(ts_w_indicators.head(5))
    print(list(ts_w_indicators.columns))

    print(len(ts_m_indicators))
    print(ts_m_indicators.head(5))
    print(list(ts_m_indicators.columns))
    # break
    # ts_d_indicators = calculate_price_changes(ts_d_indicators, prefix='', suffix='_d')
    # # ts_w_indicators = calculate_price_changes(ts_w_indicators, prefix='', suffix='_w')
    # # ts_m_indicators = calculate_price_changes(ts_m_indicators, prefix='', suffix='_m')
    # ts_d_indicators = calculate_price_changes(ts_d_indicators, prefix='ha_', suffix='_d')
    # # ts_w_indicators = calculate_price_changes(ts_w_indicators, prefix='ha_', suffix='_w')
    # ts_m_indicators = calculate_price_changes(ts_m_indicators, prefix='ha_', suffix='_m')

    ts_d = ts_d.set_index('datetime').sort_index().rename(
        columns={'open': 'open_d', 'close': 'close_d', 'high': 'high_d', 'low': 'low_d', 'volume': 'volume_d'})

    # ts_d.to_csv('../downloads/data_engineering/' + symbol + '_ml_ts_d.tsv', sep='\t')
    # ts_d_indicators.to_csv('../downloads/data_engineering/' + symbol + '_ml_indicator_d.tsv', sep='\t')
    # ts_w_indicators.to_csv('../downloads/data_engineering/' + symbol + '_ml_indicator_w.tsv', sep='\t')
    # ts_m_indicators.to_csv('../downloads/data_engineering/' + symbol + '_ml_indicator_m.tsv', sep='\t')

    # break
    # ### Begin collecting columns to drop
    drop_cols = []

    print('ts_d index: ', ts_d.index)
    print('ts_d_indicators index: ', ts_d_indicators.index)
    print('ts_w_indicators index: ', ts_w_indicators.index)
    print('ts_m_indicators index: ', ts_m_indicators.index)
    # ts_d['datetime'] = ts_d['datetime'].astype(str)
    # break
    # ts_d_indicators['datetime'] = ts_d_indicators['datetime'].astype(str)
    # ts_w_indicators['datetime'] = ts_w_indicators['datetime'].astype(str)
    # ts_m_indicators['datetime'] = ts_m_indicators['datetime'].astype(str)

    # print('ts_w_indicators: ', ts_w_indicators.columns)
    # print('ts_w_indicators: ', ts_w_indicators)
    # print('ts_m_indicators: ', ts_m_indicators.columns)
    # print('ts_m_indicators: ', ts_m_indicators)
    # break

    ### Merge the daily, weekly, monthly data
    ts_merged = ts_d.merge(ts_d_indicators.drop(['open_d', 'high_d', 'low_d', 'close_d'], axis=1),
                           left_index=True, right_index=True)
    ts_merged = ts_merged.merge(ts_w_indicators.drop(['open_w', 'high_w', 'low_w', 'close_w'], axis=1),
                                left_index=True, right_index=True)
    ts_merged = ts_merged.merge(ts_m_indicators.drop(['open_m', 'high_m', 'low_m', 'close_m'], axis=1),
                                left_index=True, right_index=True)
    # ts_merged = ts_merged.set_index('datetime')
    # print('ts_merged length: ', len(ts_merged))
    # print('ts_merged head: ', ts_merged.head(5))
    # print('ts_merged columns: ', list(ts_merged.columns))

    ### Calculate the relative ohlc4 for day/week day/month, and week/month
    ts_merged['rel_ohlc4_d_w'] = (ts_merged['ohlc4_d'] / ts_merged['ohlc4_w']).round(round_digits)
    ts_merged['rel_ohlc4_d_m'] = (ts_merged['ohlc4_d'] / ts_merged['ohlc4_m']).round(round_digits)
    ts_merged['rel_ohlc4_w_m'] = (ts_merged['ohlc4_w'] / ts_merged['ohlc4_m']).round(round_digits)

    ### Calculate trends in daily open
    for i in [2, 5, 10, 15, 20]:
        ts_merged['open_rel_d_' + str(i)] = (ts_merged['open_d'] / ts_merged['open_d'].rolling(window=i).mean()).round(
            round_digits)
        # final_f_cols.append('open_rel_d_' + str(i))

    ### Calculate trends in daily high
    for i in [2, 5, 10, 15, 20]:
        ts_merged['high_rel_d_' + str(i)] = (ts_merged['high_d'] / ts_merged['high_d'].rolling(window=i).mean()).round(
            round_digits)
        # final_f_cols.append('high_rel_d_' + str(i))

    ### Calculate trends in daily low
    for i in [2, 5, 10, 15, 20]:
        ts_merged['low_rel_d_' + str(i)] = (ts_merged['low_d'] / ts_merged['low_d'].rolling(window=i).mean()).round(
            round_digits)
        # final_f_cols.append('low_rel_d_' + str(i))

    ### Calculate trends in daily close
    for i in [2, 5, 10, 15, 20]:
        ts_merged['close_rel_d_' + str(i)] = (
                    ts_merged['close_d'] / ts_merged['close_d'].rolling(window=i).mean()).round(round_digits)
        # final_f_cols.append('close_rel_d_' + str(i))

    ### Calculate trends in rsi indicators
    for i in [2, 5, 10, 15, 20]:
        ts_merged['rsi_d_' + str(i)] = ts_merged['rsi_d'].rolling(window=i).mean().round(round_digits)
        # final_f_cols.append('rsi_d_' + str(i))

    ### Calculate trends in macd indicators
    for i in [2, 5, 10, 15, 20]:
        ts_merged['macd_d_' + str(i)] = ts_merged['macd_d'].rolling(window=i).mean().round(round_digits)
        # final_f_cols.append('macd_d_' + str(i))

    ### Calculate trends in moving average crossover indicator
    for i in [2, 5, 10, 15, 20]:
        ts_merged['mac_indicator_d_' + str(i)] = ts_merged['mac_indicator_d'].rolling(window=i).mean().round(
            round_digits)
        # final_f_cols.append('mac_d_' + str(i))

    # ### Calculate trends in moving average change indicator
    # for i in [2, 5, 10, 15, 20]:
    #     ts_merged['mach_d_' + str(i)] = ts_merged['mach_d'].rolling(window=i).mean().round(round_digits)
    #     # final_f_cols.append('mach_d_' + str(i))

    ### Calculate trends in heiken ashi buy indicators
    for i in [2, 5, 8, 10, 15, 20, 50, 75, 100]:
        ts_merged['habs_indicator_d' + '_' + str(i)] = ts_merged['habs_indicator_d'].rolling(window=i).mean().round(
            round_digits)
        ts_merged['habs_indicator_w' + '_' + str(i)] = ts_merged['habs_indicator_w'].rolling(window=i).mean().round(
            round_digits)
        ts_merged['habs_indicator_m' + '_' + str(i)] = ts_merged['habs_indicator_m'].rolling(window=i).mean().round(
            round_digits)
        # final_f_cols.append(prefix+'_'+str(i))

    ### Create the simple moving average relationship column
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged['close_sma_' + str(i)] = ts_merged['close_d'].rolling(window=i).mean()
        ts_merged['close_sma_' + str(i) + '_change_d'] = (
                    ts_merged['close_d'] / ts_merged['close_sma_' + str(i)]).round(round_digits)
        ts_merged['close_sma_' + str(i)] = ts_merged['close_d'].rolling(window=i).mean()
        ts_merged['close_sma_' + str(i) + '_change_w'] = (
                    ts_merged['close_w'] / ts_merged['close_sma_' + str(i)]).round(round_digits)
        ts_merged['close_sma_' + str(i)] = ts_merged['close_d'].rolling(window=i).mean()
        ts_merged['close_sma_' + str(i) + '_change_m'] = (
                    ts_merged['close_m'] / ts_merged['close_sma_' + str(i)]).round(round_digits)
        drop_cols.append('close_sma_' + str(i))

    ### Calculate daily, weekly, monthly volume relative to n period simple moving average
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged['vol_sma_' + str(i)] = ts_merged['volume_d'].rolling(window=i).mean()
        ts_merged['vol_sma_' + str(i) + '_change_d'] = (ts_merged['volume_d'] / ts_merged['vol_sma_' + str(i)]).round(
            round_digits)
        ts_merged['vol_sma_' + str(i)] = ts_merged['volume_d'].rolling(window=i).mean()
        ts_merged['vol_sma_' + str(i) + '_change_w'] = (ts_merged['volume_w'] / ts_merged['vol_sma_' + str(i)]).round(
            round_digits)
        ts_merged['vol_sma_' + str(i)] = ts_merged['volume_d'].rolling(window=i).mean()
        ts_merged['vol_sma_' + str(i) + '_change_m'] = (ts_merged['volume_m'] / ts_merged['vol_sma_' + str(i)]).round(
            round_digits)
        drop_cols.append('vol_sma_' + str(i))

    # ### Calculate weekly volume relative to n period simple moving average
    # prefix = 'vol_sma_'
    # for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100, 200]:
    #     ts_merged[prefix+str(i)] = ts_merged['volume_w'].rolling(window=i).mean()
    #     ts_merged[prefix+str(i)+'_change_w'] = (ts_merged['volume_w'] / ts_merged[prefix+str(i)]).round(round_digits)
    #     drop_cols.append(prefix+str(i))
    #     # final_f_cols.append(prefix + str(i) + '_change_w')
    #
    # ### Calculate monthly volume relative to n period simple moving average
    # prefix = 'vol_sma_'
    # for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100, 200]:
    #     ts_merged[prefix+str(i)] = ts_merged['volume_m'].rolling(window=i).mean()
    #     ts_merged[prefix+str(i)+'_change_m'] = (ts_merged['volume_m'] / ts_merged[prefix+str(i)]).round(round_digits)
    #     drop_cols.append(prefix+str(i))
    #     # final_f_cols.append(prefix + str(i) + '_change_m')

    ### Calculate volume change relative to n period simple moving average
    prefix = 'rel_change_vol_sma_'
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged[prefix + str(i)] = (ts_merged['vol_sma_' + str(i) + '_change_d'] * ts_merged['change_close_d']).round(
            round_digits)
        # final_f_cols.append(prefix+str(i))

    ### Calculate volume weighted by the daily change
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged['change_weighted_vol_' + str(i) + '_d'] = ((ts_merged['close_d'] - ts_merged['close_d'].shift(1)) *
                                                             ts_merged['volume_d']).rolling(window=i).sum() / (
                                                                        ts_merged['volume_d'] * ts_merged[
                                                                    'close_d']).rolling(window=i).sum()
        ts_merged['change_weighted_vol_' + str(i) + '_w'] = ((ts_merged['close_w'] - ts_merged['close_w'].shift(1)) *
                                                             ts_merged['volume_w']).rolling(window=i).sum() / (
                                                                        ts_merged['volume_w'] * ts_merged[
                                                                    'close_w']).rolling(window=i).sum()
        ts_merged['change_weighted_vol_' + str(i) + '_m'] = ((ts_merged['close_m'] - ts_merged['close_m'].shift(1)) *
                                                             ts_merged['volume_m']).rolling(window=i).sum() / (
                                                                        ts_merged['volume_m'] * ts_merged[
                                                                    'close_m']).rolling(window=i).sum()

    ### Calculate adjusted close relative to n period minimum
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged['min_close_' + str(i)] = ts_merged['close_d'].rolling(window=i).min()
        ts_merged['min_close_' + str(i) + '_change_d'] = (
                    ts_merged['close_d'] / ts_merged['min_close_' + str(i)]).round(round_digits)
        ts_merged['min_close_' + str(i)] = ts_merged['close_w'].rolling(window=i).min()
        ts_merged['min_close_' + str(i) + '_change_w'] = (
                    ts_merged['close_w'] / ts_merged['min_close_' + str(i)]).round(round_digits)
        ts_merged['min_close_' + str(i)] = ts_merged['close_m'].rolling(window=i).min()
        ts_merged['min_close_' + str(i) + '_change_m'] = (
                    ts_merged['close_m'] / ts_merged['min_close_' + str(i)]).round(round_digits)
        drop_cols.append('min_close_' + str(i))

    ### Calculate adjusted_close relative to n period maximum
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged['max_close_' + str(i)] = ts_merged['close_d'].rolling(window=i).max()
        ts_merged['max_close_' + str(i) + '_change_d'] = (
                    ts_merged['close_d'] / ts_merged['max_close_' + str(i)]).round(round_digits)
        ts_merged['max_close_' + str(i)] = ts_merged['close_w'].rolling(window=i).max()
        ts_merged['max_close_' + str(i) + '_change_w'] = (
                    ts_merged['close_w'] / ts_merged['max_close_' + str(i)]).round(round_digits)
        ts_merged['max_close_' + str(i)] = ts_merged['close_m'].rolling(window=i).max()
        ts_merged['max_close_' + str(i) + '_change_m'] = (
                    ts_merged['close_m'] / ts_merged['max_close_' + str(i)]).round(round_digits)
        drop_cols.append('max_close_' + str(i))

    ### Calculate volume weighted change in price
    prefix = 'volume_close_interaction_'
    for i in [2, 4, 5, 8, 10, 15, 26, 30, 50, 100]:
        ts_merged[prefix + str(i)] = ts_merged['close_d'] * ts_merged['volume_d']
        ts_merged[prefix + str(i) + '_rolling'] = ts_merged[prefix + str(i)].rolling(window=i).max()
        ts_merged[prefix + str(i) + '_change'] = (
                    ts_merged[prefix + str(i)] / ts_merged[prefix + str(i) + '_rolling']).round(round_digits)
        drop_cols.append(prefix + str(i))
        drop_cols.append(prefix + str(i) + '_rolling')
        # final_f_cols.append(prefix + str(i) + '_change_d')

    ### Add the moving average crossover columns
    ts_merged['ma_crossover_5_30'] = np.where(
        ts_merged['close_d'].rolling(5).mean() >= ts_merged['close_d'].rolling(30).mean(), 1, 0)
    ts_merged['ma_crossover_10_20'] = np.where(
        ts_merged['close_d'].rolling(10).mean() >= ts_merged['close_d'].rolling(20).mean(), 1, 0)
    ts_merged['ma_crossover_20_40'] = np.where(
        ts_merged['close_d'].rolling(20).mean() >= ts_merged['close_d'].rolling(40).mean(), 1, 0)
    ts_merged['ma_crossover_30_60'] = np.where(
        ts_merged['close_d'].rolling(30).mean() >= ts_merged['close_d'].rolling(60).mean(), 1, 0)
    ts_merged['ma_crossover_50_100'] = np.where(
        ts_merged['close_d'].rolling(50).mean() >= ts_merged['close_d'].rolling(100).mean(), 1, 0)

    print('before dropna: ', ts_merged.tail(5).index)
    print('before dropna: ', ts_merged.tail(5))
    ts_merged.to_csv('../downloads/data_engineering/' + symbol + '_ml_dataengineering.tsv', sep='\t')
    ### Remove any previous rows with N/A values
    ts_merged = ts_merged.dropna(axis=0)
    print('after dropna: ', ts_merged.tail(5).index)
    # break
    print('ts_merged length a: ', ts_merged.tail(5).index)
    ###### Create the test columns
    # Compare the current close to future high
    for i in [1, 3, 5, 10, 20, 30]:
        ts_merged['future_' + str(i) + '_day_maxhigh'] = max_of_future_period(ts_merged['high_d'].values, days=i)
        ts_merged['future_' + str(i) + '_day_maxhigh_rel'] = (
                    ts_merged['future_' + str(i) + '_day_maxhigh'] / ts_merged['close_d']).round(round_digits)
        drop_cols.append('future_' + str(i) + '_day_maxhigh')
        # final_test_cols.append('future_'+str(i)+'_day_maxhigh_rel')

    ### Compare the current close to future closes
    for i in [1, 3, 5, 10, 20, 30]:
        ts_merged['future_' + str(i) + '_day_maxclose'] = max_of_future_period(ts_merged['close_d'].values, days=i)
        ts_merged['future_' + str(i) + '_day_maxclose_rel'] = (
                    ts_merged['future_' + str(i) + '_day_maxclose'] / ts_merged['close_d']).round(round_digits)
        drop_cols.append('future_' + str(i) + '_day_maxclose')
        # final_test_cols.append('future_' + str(i) + '_day_maxclose_rel')

    ### Fill any future N/A with 0
    ts_merged = ts_merged.fillna(0)
    print('ts_merged length b: ', len(ts_merged))
    ### Generate any streaks for prediction
    streaks = get_streaks(df=ts_merged, streak_col='habs_indicator_d', price_col='close_d',
                          peak_col='high_d', peak_func=lambda x, y: x >= y, start_val=[1], end_val=[-1])
    ts_merged['habs_d_buy_streak_pos'] = streaks[1]
    ts_merged['habs_d_buy_streak_len'] = streaks[2]
    ts_merged['habs_d_buy_streak_change'] = streaks[3]
    ts_merged['habs_d_buy_streak_peak'] = streaks[4]

    streaks = get_streaks(df=ts_merged, streak_col='habs_indicator_w', price_col='close_w',
                          peak_col='high_w', peak_func=lambda x, y: x >= y, start_val=[1], end_val=[-1])
    ts_merged['habs_w_buy_streak_pos'] = streaks[1]
    ts_merged['habs_w_buy_streak_len'] = streaks[2]
    ts_merged['habs_w_buy_streak_change'] = streaks[3]
    ts_merged['habs_w_buy_streak_peak'] = streaks[4]

    streaks = get_streaks(df=ts_merged, streak_col='habs_indicator_m', price_col='close_m',
                          peak_col='high_m', peak_func=lambda x, y: x >= y, start_val=[1], end_val=[-1])
    ts_merged['habs_m_buy_streak_pos'] = streaks[1]
    ts_merged['habs_m_buy_streak_len'] = streaks[2]
    ts_merged['habs_m_buy_streak_change'] = streaks[3]
    ts_merged['habs_m_buy_streak_peak'] = streaks[4]

    # streaks = get_streaks(df=ts_merged, streak_col='habs_indicator_d', price_col='close_d',
    #                       peak_col='low_d', peak_func=lambda x, y: x <= y, start_val=[-1], end_val=[1])
    # ts_merged['habs_d_sell_streak_pos'] = streaks[1]
    # ts_merged['habs_d_sell_streak_len'] = streaks[2]
    # ts_merged['habs_d_sell_streak_change'] = streaks[3]
    # ts_merged['habs_d_sell_streak_peak'] = streaks[4]

    print('ts_merged length c: ', len(ts_merged))
    # Drop the unncessary columns
    ts_merged = ts_merged.drop(drop_cols, axis=1).fillna(0)
    ts_merged.replace(np.inf, 1, inplace=True)
    print('ts_merged length d: ', len(ts_merged))
    # Add the symbol column
    ts_merged['symbol'] = symbol
    ts_merged['type'] = sym_type

    ts_merged = ts_merged.reset_index(drop=False).rename(columns={'index': 'datetime'})
    first_columns = ['datetime', 'symbol', 'type']
    ts_merged = ts_merged.loc[:, first_columns + [i for i in ts_merged.columns if i not in first_columns]]
    print('ts_merged length: ', len(ts_merged))
    print('ts_merged head: ', ts_merged.head(5))
    print('ts_merged columns: ', list(ts_merged.columns))

    ### Saving the data
    ts_merged.to_parquet('../downloads/ml_dataengineering.parquet', partition_cols='symbol')
    ts_merged.to_csv('../downloads/data_engineering/' + symbol + '_ml_dataengineering.tsv', sep='\t')
    # print(list(ts_merged['datetime']))
