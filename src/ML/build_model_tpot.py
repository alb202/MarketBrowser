import datetime
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.cluster import k_means, spectral_clustering
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, RobustScaler, \
    KBinsDiscretizer
from tpot import TPOTClassifier

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
import pickle
from skrebate import ReliefF

sns.set_theme(style="white")
sns.set_style(style='darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

date = '20201220'


def get_feature_correlation(df, X, y):
    results = dict()
    if isinstance(X, str):
        X = [X]
    for i in X:
        results[i] = [round(min(np.corrcoef(df[i], df[y])[0]), 4)]
    return results


def is_list_of_int(l):
    return len(l) == sum([float(int(i)) == float(i) for i in l])


def get_hist_type(df):
    results = dict()
    cols = list(df.columns)
    for i in cols:
        uniq_vals = df[i].drop_duplicates().tolist()
        num_values = len(uniq_vals)
        if (num_values == 2) | (num_values == 3):  # & is_list_of_int(uniq_vals)):
            results[i] = 'cat'
        else:
            results[i] = 'other'
        # elif (scipy.stats.kurtosistest(df[i])[0] > .5) & (abs(scipy.stats.skewtest(df[i])[0]) > 3):
        #     results[i] = 'skewed'
        # elif (scipy.stats.kurtosistest(df[i])[0] > 1) & (abs(scipy.stats.skewtest(df[i])[0]) <= 3):
        #     results[i] = 'normal'

    return results


def plot_histograms(df, out_file, ncols=5):
    nrows = int(np.ceil(len(df.columns) / ncols))
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 12, nrows * 4))
    col_num = 0
    hist_type = get_hist_type(df)
    hist_type_inverted = dict()
    for key, value in hist_type.items():
        hist_type_inverted.setdefault(value, list()).append(key)

    for row in range(0, nrows):
        for col in range(0, ncols):
            if col_num >= len(df.columns):
                continue
            col_name = list(df.columns)[col_num]
            axis[row, col].hist(data=df, x=col_name, bins=50, density=True)
            axis[row, col].set_title(col_name)
            skew_test = scipy.stats.skewtest(df[col_name])
            kurtosis_test = scipy.stats.kurtosistest(df[col_name])
            # print('kurtosis', kurtosis_test)
            # print('norm_test: ', round(norm_test[0], 4))
            # print('norm_test p: ', round(norm_test[1], 8))
            axis[row, col].text(
                x=np.median(axis[row, col].get_xticks()),
                y=np.median(axis[row, col].get_yticks()) * 1.2,
                s='skew_test:' + str(round(skew_test[0], 4)))
            axis[row, col].text(
                x=np.median(axis[row, col].get_xticks()),
                y=np.median(axis[row, col].get_yticks()),
                s='kurtosis_test:' + str(round(kurtosis_test[0], 4)))
            axis[row, col].text(
                x=np.median(axis[row, col].get_xticks()),
                y=np.median(axis[row, col].get_yticks()) * .8,
                s='Unique_vals:' + str(len(df[col_name].drop_duplicates())))
            axis[row, col].text(
                x=np.median(axis[row, col].get_xticks()),
                y=np.median(axis[row, col].get_yticks()) * .6,
                s='Dist_type:' + hist_type[col_name])
            col_num += 1
    fig.subplots_adjust(wspace=.2, hspace=.2)
    plt.autoscale()
    plt.savefig('../downloads/model_data/' + out_file)
    plt.clf()
    plt.cla()
    plt.close('all')


def feature_correlations(df, y):
    results = dict()
    features = [i for i in df.columns if i != y]
    results = {i: [scipy.stats.pearsonr(df[i], df[y])[0]] for i in features}
    return pd.DataFrame.from_dict(results).transpose().rename(columns={0: 'corr'}).sort_values('corr').reset_index(
        drop=False)


def feature_correlation_matrix(df, file=None):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(corr, mask=mask, cmap=cmap,
                     vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.figure.tight_layout()
    plt.autoscale()
    if file:
        plt.savefig('../downloads/model_data/' + file + '.jpg')
    plt.clf()
    plt.cla()
    plt.close('all')


def clear_plot():
    plt.clf()
    plt.cla()
    plt.close('all')


### Get all symbols from parquet directory
symbols = [i.split('=')[1] for i in os.listdir('../downloads/ml_dataengineering.parquet/')]

### Create Scalers
minmax_scaler = MinMaxScaler(feature_range=[-1, 1])
quantile_scaler_bins = 5
qtransform_scaler = QuantileTransformer(n_quantiles=quantile_scaler_bins)
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()
kbins_discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
power_transformer = PowerTransformer()

### Start date for model building and identity of test column
start_date = '2018-01-01'
y_col = 'test_col'
target_filter_col = 'habs_d_buy_streak_pos'
target_filter_val = 1
target_col = 'future_5_day_maxclose_rel'  # 'habs_d_buy_streak_len'#'future_10_day_maxclose_rel'#'habs_d_buy_streak_len'#'future_10_day_maxclose_rel'#'habs_d_buy_streak_change' #'habs_d_buy_streak_len' 'habs_d_buy_streak_high'
min_days_for_model = 200
test_value = 1.1
feature_correlation_cutoff = .1

### Dict to hold data points for each symbol
all_streaks = {'symbol': []}  # dict()
# all_dfs = []

### Number of symbols to process
number_of_symbols = 1000

# for symbol in symbols[:number_of_symbols]:
# for symbol in ['AMAT', 'ACAD', 'IGC', 'HEXO', 'PLAY']:
# for symbol in ['ALNY']:

all_symbols = ['PLAY', 'REGN', 'ALNY', 'ACAD', 'AMAT', 'SIMO', 'AAL', 'INO',
               'HEXO', 'IGC', 'ACB', 'JMIA', 'PFE', 'SNDL', 'MRNA', 'CIDM', 'MTSI']
all_symbols = ['ABBV', 'ABT', 'ACAD', 'ADAP', 'ADVM', 'AGIO', 'AIMT', 'ALEC', 'ALKS', 'ALLO', 'ALNY', 'ALXN',
               'AMGN', 'AMTI', 'ARNA', 'ARVN', 'ARWR', 'ASND', 'ATRA', 'AUPH', 'AZN', 'BBIO', 'BDTX', 'BDX',
               'BEAM', 'BGNE', 'BIIB', 'BLUE', 'BMRN', 'BMY', 'BNTX', 'BPMC', 'BSX', 'BTAI', 'CBPO', 'CCXI',
               'CDMOP', 'CGEN', 'CHRS', 'CRSP', 'CRTX', 'CYTK', 'DCPH', 'DHR', 'DNLI', 'DRNA', 'EBS', 'EDIT',
               'EIDX', 'EPZM', 'ESPR', 'EW', 'EXAS', 'EXEL', 'FATE', 'FBIOP', 'FGEN', 'FOLD', 'GBT', 'GILD',
               'GLPG', 'GSK', 'HALO', 'HRTX', 'IGC', 'ILMN', 'IMMU', 'IMVT', 'INCY', 'INO', 'INSM', 'IONS',
               'IOVA', 'ISRG', 'ITCI', 'ITOS', 'JNJ', 'KOD', 'KPTI', 'KURA', 'KYMR', 'LGND', 'LLY', 'LMNX',
               'MCRB', 'MDT', 'MESO', 'MGNX', 'MNTA', 'MOR', 'MRK', 'MRNA', 'MRSN', 'MRTX', 'MYGN', 'MYOV',
               'NBIX', 'NGM', 'NKTX', 'NSTG', 'NTLA', 'NVAX', 'NVO', 'NVS', 'PACB', 'PCVX', 'PFE', 'PTCT',
               'QURE', 'RARE', 'RCKT', 'RCUS', 'REGN', 'REPL', 'RGNX', 'RLAY', 'RNA', 'RYTM', 'SGEN', 'SGMO',
               'SLS', 'SNY', 'SRNE', 'SRPT', 'STOK', 'SYK', 'TAK', 'TBIO', 'TECH', 'TGTX', 'TMO', 'TRIL', 'TWST',
               'TXG', 'VCYT', 'VIE', 'VIR', 'VRTX', 'XLRN', 'XNCR', 'ZLAB', 'ZNTL', 'ZTS']
for symbol in all_symbols:
    print('Processing symbol:', datetime.datetime.now(), symbol)
    ### Read in the engineered data for the symbol
    try:
        ts_main = pd.read_parquet(f'../downloads/ml_dataengineering.parquet/symbol={symbol}/')
    except (FileNotFoundError) as e:
        print(e)
        continue
    ###### GET TEST ROWS AND CREATE TEST COLUMN
    ### If there is less than n rows for a symbol, do not build a model
    if len(ts_main) < min_days_for_model:
        continue

    ### Filter data points from start date with datetime and
    ts_main = ts_main[ts_main['datetime'] >= start_date]
    # ts_main = ts_main[(ts_main[target_filter_col] == target_filter_val) & (ts_main[target_col] > 0)]
    ts_main = ts_main[(ts_main[target_filter_col] == target_filter_val) & (ts_main[target_col] >= test_value)]

    ### Set the datetime as the index column
    ts_main = ts_main.reset_index(drop=True)
    ts_main['datetime'] = [i.strftime('%Y-%m-%d') + '_' + symbol for i in ts_main['datetime']]
    ts_main = ts_main.sort_values('datetime', ascending=True).set_index('datetime')

    ### Create the y series
    # y_col_series = ts_main[target_col].apply(lambda x: 1 if x >= test_value else 0)
    # y_col_series = ts_main[target_col].apply(lambda x: round(number=x, ndigits=3))
    y_col_series = ts_main[target_col].apply(lambda x: round(number=x, ndigits=0))

    ###### DATA FORMATTING
    ### Delete unneeded columns
    drop_cols = ['open_d', 'high_d', 'low_d', 'close_d', 'volume_d',
                 'open_w', 'high_w', 'low_w', 'close_w', 'volume_w',
                 'open_m', 'high_m', 'low_m', 'close_m', 'volume_m',
                 'type', 'day', 'week_id', 'month', 'year',
                 'ohlc4_d', 'ha_open_d', 'ha_high_d', 'ha_low_d', 'ha_close_d',
                 'ohlc4_w', 'ha_open_w', 'ha_high_w', 'ha_low_w', 'ha_close_w',
                 'ohlc4_m', 'ha_open_m', 'ha_high_m', 'ha_low_m', 'ha_close_m',
                 'future_1_day_maxhigh_rel', 'future_3_day_maxhigh_rel', 'future_5_day_maxhigh_rel',
                 'future_10_day_maxhigh_rel', 'future_20_day_maxhigh_rel', 'future_30_day_maxhigh_rel',
                 'future_1_day_maxclose_rel', 'future_3_day_maxclose_rel', 'future_5_day_maxclose_rel',
                 'future_10_day_maxclose_rel', 'future_20_day_maxclose_rel', 'future_30_day_maxclose_rel',
                 'habs_d_buy_streak_pos', 'habs_d_buy_streak_change', 'habs_d_buy_streak_peak', 'habs_d_buy_streak_len',
                 'habs_w_buy_streak_pos', 'habs_w_buy_streak_change', 'habs_w_buy_streak_peak', 'habs_w_buy_streak_len',
                 'habs_m_buy_streak_pos', 'habs_m_buy_streak_change', 'habs_m_buy_streak_peak', 'habs_m_buy_streak_len']
    # print(drop_cols)
    ts_main = ts_main.drop(drop_cols, axis=1)

    ### Sort the columns by name
    ts_main = ts_main.loc[:, sorted(list(ts_main.columns))]

    ### Perform scaling of all columns, if needed. Otherwise, just rename main data
    # df_scaled = ts_main.copy(deep=True)
    # df_scaled[ts_main.columns] = scaler.fit_transform(ts_main)

    ### Add the unscaled y data back to the main dataframe
    ts_main['TEST'] = y_col_series

    # [3][5]
    # ### Show correlations between features and y; Remove features with low correlation
    # feat_corr = feature_correlations(df=df_scaled, y='TEST')
    # feat_corr = feat_corr[abs(feat_corr['corr']) >= feature_correlation_cutoff]
    # feat_corr.to_csv('../downloads/model_data/'+symbol+'_feature_correlations.tsv', sep='\t', index=False)
    # df_scaled = df_scaled.loc[:, list(feat_corr['index'])+['TEST']]
    # clear_plot()
    # ax = sns.barplot(data=feat_corr, x='corr', y='index')
    # ax.figure.tight_layout()
    # plt.savefig('../downloads/model_data/'+symbol+'_feature_correlations.jpg')
    # clear_plot()
    #
    # ### Create a correlation matrix to view relationships between features
    # feature_correlation_matrix(df=df_scaled, file=symbol+'_feature_matrix')
    # clear_plot()
    # df_scaled.corr(method='pearson').reset_index(drop=False).melt(
    #     id_vars='index', value_name='corr').sort_values('corr').to_csv(path_or_buf='../downloads/model_data/'+symbol+'_feature_corr.tsv', sep='\t')

    ### Add the row data to the data dictionary
    # all_streaks[symbol] = df_scaled
    print(f'Symbol: {symbol}', len(ts_main), len(ts_main.columns))
    all_streaks['symbol'].append(ts_main)
    # pd.DataFrame({'symbol': [symbol], 'columns': ['|'.join(list(df_scaled.columns)[:-1])]}).to_csv(
    #     path_or_buf='../downloads/model_data/'+symbol+'_model_cols.tsv', sep='\t', index=False, header=False)

###### SPLIT THE DATA INTO TRAIN AND TEST
### Do a time series split
test_train_split = .80
cv_splits_inner = 4
# n_random_params = 250
skip_last_for_model = 15  # None # -1
scoring_type = 'f1'
n_best_features = 50
# scoring_type = 'balanced_accuracy'
# scoring_type = 'average_precision'

### Calculate number of possible parameter sets
# print('Number of unique param combinations: ', reduce(lambda x, y: x*y, [len(values) for key, values in params.items()]))

# ### Create containers for output
# accuracy_scores = dict()
# best_parameters = []
# all_parameters = []
# all_feature_importance = []
# score_matrices = pd.DataFrame(
#     data=itertools.product(
#         sorted(list(set([0, 1]))),
#         sorted(list(set([0, 1])))),
#     columns=['y_test', 'y_pred']).set_index(['y_test', 'y_pred'])

###### RUN THE MODEL CREATION
for symbol, df in all_streaks.items():
    ### Loading the data
    print('Loading the data ...')
    df = pd.concat(df).sort_values('datetime', ascending=True)  # .iloc[:, list(range(10))]
    print(f'Loaded {len(df)} rows of data')
    df.to_csv(path_or_buf='../downloads/model_data/all_model_data__raw.tsv', sep='\t', index=True, header=True)
    # break
    ### Split the dataset into X and y data
    X = df.loc[:, df.columns[:-1]]  # .values
    y = df[df.columns[-1:][0]]  # .values

    ###### FEATURE SCALING
    ### Plot histograms of unscaled features
    plot_histograms(X, out_file='all_data_pre_normalize.jpg', ncols=5)

    ### --- Scaling of data
    ### Get the distribution types for each feature
    hist_type = get_hist_type(X)
    hist_type_inverted = dict()
    for key, value in hist_type.items():
        hist_type_inverted.setdefault(value, list()).append(key)
    print('hist_type_inverted', hist_type_inverted)
    with open('../downloads/model_data/feature_types.json', 'w') as file:
        file.write(json.dumps(hist_type_inverted))

    # ### Scale the features
    # columns_sets = []
    # if 'cat' in hist_type_inverted.keys():
    #     X_cat = X.loc[:, hist_type_inverted['cat']].copy(deep=True)
    #     X_cat[hist_type_inverted['cat']] = minmax_scaler.fit_transform(X_cat)
    #     columns_sets.append(X_cat)
    #     pickle.dump(minmax_scaler, open('../downloads/models/cat_minmax_scaler.pkl', 'wb'))
    #
    # if 'other' in hist_type_inverted.keys():
    #     X_other = X.loc[:, hist_type_inverted['other']].copy(deep=True)
    #     X_other[hist_type_inverted['other']] = power_transformer.fit_transform(X_other)
    #     columns_sets.append(X_other)
    #     pickle.dump(power_transformer, open('../downloads/models/cont_power_transformer.pkl', 'wb'))
    # X = pd.concat(columns_sets, axis=1)

    ### Plot the scaled histograms
    plot_histograms(X, out_file='all_data_post_normalize.jpg', ncols=5)

    ### Export the scaled data
    pd.concat([X, y], axis=1).to_csv(path_or_buf='../downloads/model_data/all_model_data__scaled.tsv', sep='\t',
                                     index=True, header=True)

    ### Select the best features
    # clf = ReliefF(n_features_to_select=n_best_features, n_neighbors=100)
    print('Running feature selection ...')
    feature_select = ReliefF(n_features_to_select=n_best_features, n_neighbors=100, discrete_threshold=3, verbose=True,
                             n_jobs=1)
    # feature_select = MultiSURF(n_features_to_select=n_best_features, discrete_threshold=3, verbose=True, n_jobs=1)
    feature_select.fit(X.values, y.values)

    print(type(feature_select.transform(X.values)))
    print(len(feature_select.transform(X.values)))
    print(len(feature_select.transform(X.values)[0]))
    print(feature_select.transform(X.values))

    ### Export the model for feature selection
    pickle.dump(feature_select, open('../downloads/models/feature_select_model.pkl', 'wb'))

    ### Get the best features
    feature_importance_df = pd.DataFrame({'cols': X.columns,
                                          'val': feature_select.feature_importances_}) \
        .sort_values('val', ascending=False).reset_index(drop=True)
    X = pd.DataFrame(
        data=feature_select.transform(X.values),
        columns=feature_importance_df.head(n_best_features)['cols'].tolist())
    feature_importance_df.to_csv('../downloads/model_data/all_feature_importance.tsv', sep='\t')
    X.to_csv('../downloads/model_data/only_selected_features.tsv', sep='\t')

    print('best_features.columns', feature_importance_df)
    print('RELIEFF X column count: ', len(X.columns))
    print('Selected columns: ', X.columns)
    print('Final X: ', X)

    # [2][5]
    ### Create test train indicies
    test_train_split_index = int(test_train_split * len(X))
    train_index = list(range(len(X)))[:test_train_split_index]
    test_index = list(range(len(X)))[test_train_split_index:]
    print('Number of samples: ', len(df))
    print('Number of training samples: ', len(train_index))
    print('Number of test samples: ', len(test_index))

    ### Use the indices to create X and y train and test sets using 1 of the splits
    X_train = np.array([X.iloc[i, :] for i in train_index]).astype(float)
    X_test = np.array([X.iloc[i, :] for i in test_index]).astype(float)
    y_train = np.array([y.iloc[i] for i in train_index]).astype(float)
    y_test = np.array([y.iloc[i] for i in test_index]).astype(float)
    y_test_labels = list(df.iloc[range(test_train_split_index, len(df)), :].index)
    print('y_test_labels', y_test_labels)
    print('y_test', y_test)

    tpot = TPOTClassifier(
        generations=1,
        population_size=1,
        verbosity=3,
        max_time_mins=30,
        max_eval_time_mins=1,
        cv=cv_splits_inner,
        random_state=99,
        log_file='../downloads/model_data/tpot_output.log',
        early_stop=None,
        # scoring='f1',
        periodic_checkpoint_folder='../downloads/model_data/')

    # tpot = TPOTRegressor(generations=30,
    #                      population_size=30,
    #                      # offspring_size=None,
    #                      # mutation_rate=0.9,
    #                      # crossover_rate=0.1,
    #                      memory='auto',
    #                      # scoring='neg_median_absolute_error',
    #                      scoring= 'neg_mean_squared_error', #'r2', #'neg_median_absolute_error' 'neg_mean_squared_error'
    #                      config_dict=None, #'TPOT MDR', #tpot_mdr_regressor_config_dict
    #                      max_time_mins=560,
    #                      max_eval_time_mins=10,
    #                      # cv=TimeSeriesSplit(n_splits=cv_splits_inner),
    #                      cv=cv_splits_inner,
    #                      warm_start=True,
    #                      random_state=99,
    #                      periodic_checkpoint_folder='../downloads/model_data/',
    #                      early_stop=None,
    #                      verbosity=3,
    #                      log_file='../downloads/model_data/tpot_output.log')

    ### Fit the TPOT model
    tpot.fit(X_train, y_train)

    ### Print the training score
    print('TPOT Training Score: ', tpot.score(X_train, y_train))

    ### Save the prediction results to a df
    y_pred = [np.round(a=i, decimals=2) for i in list(tpot.predict(X_test))]
    print('y_pred:', y_pred)
    print('y_test:', y_test)

    results = pd.DataFrame(data={'date': y_test_labels,
                                 'y_test': y_test,
                                 'y_pred': y_pred})
    results['d'] = results['y_pred'] - results['y_test']
    r_squared = str(round(r2_score(results['y_test'], results['y_pred']), 3))
    lin_regression = scipy.stats.linregress(
        x=results['y_test'],
        y=results['y_pred'])
    print(f'R-squared of predictions: {r_squared}')
    print(f'Linear regression: slope={lin_regression[0]} '
          f'intercept={lin_regression[1]} '
          f'rvalue={lin_regression[2]} '
          f'pvalue={lin_regression[3]} '
          f'stderr={lin_regression[4]}')

    print(results)
    results.to_csv('../downloads/model_data/results.tsv', sep='\t', index=True, header=True)

    ### Export the model
    tpot.export('../downloads/model_data/best_tpot_pipeline.py')
    # print('tpot.evaluated_individuals_', tpot.evaluated_individuals_)
    with open('../downloads/model_data/all_tpot_pipelines.json', 'w') as file:
        file.write(json.dumps(tpot.evaluated_individuals_))

    ### Train the best pipeline
    best_pipeline = tpot.fitted_pipeline_
    best_pipeline.fit(X.values, y.values)
    pickle.dump(best_pipeline, open('../downloads/models/general_model.pkl', 'wb'))

    #
    # ### Create final model from full dataset
    # final_model = final_model.set_params(**best_param_dict)
    # final_model.fit(X.values[:skip_last_for_model], y.values[:skip_last_for_model])
    # pickle.dump(final_model, open('../downloads/models/general_model.pkl', 'wb'))

    ### Plot the actual predictions vs actual results
    ax = sns.regplot(data=results, y=y_pred, x=y_test, scatter=True)
    plt.tight_layout()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    ax.set_xlim(left=min([0, max(y_test)]), right=max(y_test) * 1.2)
    ax.set_ylim(bottom=min([0, max(y_pred)]), top=max(y_pred) * 1.2)
    plt.text(s=f'R-squared: {r_squared}', y=max(y_pred) * .7, x=max(y_test) * .7)
    plt.text(s=f'Slope={round(lin_regression[0], 3)}', y=max(y_pred) * .65, x=max(y_test) * .7)
    plt.text(s=f'Intercept={round(lin_regression[1], 3)}', y=max(y_pred) * .60, x=max(y_test) * .7)
    plt.text(s=f'R-Value={round(lin_regression[2], 3)}', y=max(y_pred) * .55, x=max(y_test) * .7)
    plt.savefig('../downloads/model_data/predicted_v_actual.jpg')
    # plt.show()
    clear_plot()

    ### Plot the predictions deltas
    ax = sns.barplot(data=results, y='d', x='date')
    plt.tight_layout()
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylabel('Predicted - Actual')
    ax.set_ylim(bottom=min(results['d']) * 1.2, top=max(results['d']) * 1.2)
    plt.savefig('../downloads/model_data/predicted_minus_actual.jpg')
    # plt.show()
    clear_plot()

#     [3][5]
#     ### Select the model to use
#     # model = Perceptron()
#     model = RandomForestClassifier()
#     # model = HistGradientBoostingClassifier()
#     # model = GradientBoostingClassifier()
#     # model = Perceptron()
#     # model = DecisionTreeClassifier()
#     # model = AdaBoostClassifier()
#     # print(model.get_params())
#     best_model = RandomForestClassifier()
#     final_model = RandomForestClassifier()
#
#     ### Create the randomized search to find the best parameters
#     # clf = RandomizedSearchCV(estimator=model,
#     #                          param_distributions=params,
#     #                          scoring=scoring_type,
#     #                          n_iter=n_random_params,
#     #                          n_jobs=1,
#     #                          cv=TimeSeriesSplit(n_splits=cv_splits_inner),
#     #                          verbose=0,
#     #                          return_train_score=True).fit(X_train, y_train)
#
#
#
#     ### Create a dataframe from all the parameter sets used
#     all_param_df = pd.DataFrame.from_dict(clf.cv_results_['params'])
#     all_param_df['mean_train_score'] = clf.cv_results_['mean_train_score']
#     all_param_df['mean_test_score'] = clf.cv_results_['mean_test_score']
#     all_param_df['rank_test_score'] = clf.cv_results_['rank_test_score']
#     all_param_df['symbol'] = symbol
#     all_parameters.append(all_param_df)
#
#     ### Create a dictionary from the best parameters
#     best_param_dict = clf.best_estimator_.get_params()
#
#     ### Create the model with the best parameters using the full training set
#     best_model = best_model.set_params(**best_param_dict)
#     best_model.fit(X_train, y_train)
#
#     ### Apply the model to the full test set
#     y_pred = best_model.predict(X_test)
#
#     ### Get the model scores and probabilities
#     probs = best_model.predict_proba(X_test)
#     # probs = list(best_model.decision_function(X_test))
#     # print('probs: ', probs)
#     accuracy_score = best_model.score(X_test, y_test)
#     result_df = pd.DataFrame({'y_pred': y_pred,
#                               'y_test': y_test,
#                               symbol: [1]*len(y_pred)})
#     result_df = result_df.groupby(['y_test', 'y_pred']).count()
#
#     ### Create dataframe of feature importance from model
#     feature_importance = pd.DataFrame(
#         # list(zip(list(X.columns), [round(i, 4) for i in best_model.coef_[0]])),
#         list(zip(list(X.columns), [round(i, 4) for i in best_model.feature_importances_])),
#         columns=['name', symbol]).sort_values(symbol).set_index('name')
#     all_feature_importance.append(feature_importance)
#
#     ### Add scoring to score matrix
#     score_matrices = score_matrices.join(result_df, how='outer')
#
#     ### Add accuracy score to score list
#     accuracy_scores[symbol] = [accuracy_score]
#
#     ### Create final dataframe of best parameters and testing results
#     best_param_df = pd.DataFrame.from_dict({key: [value] for key, value in best_param_dict.items()})
#     best_param_df['test_score'] = accuracy_score
#     best_param_df = pd.concat([best_param_df]*len(probs)).reset_index(drop=True)
#     best_param_df['y_prob'] = list(
#         zip(y.iloc[test_index].index, y_pred, y_test, [abs(round(x[1]-.5, 2)) for x in probs]))
#     # best_param_df['y_prob'] = list(
#     #     zip(y.iloc[test_index].index, y_pred, y_test, [abs(round(x, 2)) for x in probs]))
#     best_param_df['y_label'] = best_param_df['y_prob'].apply(lambda x: x[0])
#     best_param_df['y_pred'] = best_param_df['y_prob'].apply(lambda x: x[1])
#     best_param_df['y_test'] = best_param_df['y_prob'].apply(lambda x: x[2])
#     best_param_df['y_prob'] = best_param_df['y_prob'].apply(lambda x: x[3])
#     best_param_df['result'] = best_param_df.apply(lambda row: str(row['y_pred']) + '_' + str(row['y_test']), axis=1)
#     best_param_df['symbol'] = symbol
#     ### Plot best parameters
#     clear_plot()
#     ax = sns.swarmplot(data=best_param_df, x='result', y='y_prob')
#     ax.figure.tight_layout()
#     plt.savefig('../downloads/model_data/'+symbol+'_best_parameters_.jpg')
#     clear_plot()
#
#
#
#     best_parameters.append(best_param_df)
#
#     ### Create final model from full dataset
#     final_model = final_model.set_params(**best_param_dict)
#     final_model.fit(X.values[:skip_last_for_model], y.values[:skip_last_for_model])
#     pickle.dump(final_model, open('../downloads/models/general_model.pkl', 'wb'))
#     # clear_plot()
#     # plot_precision_recall_curve(estimator=final_model,
#     #                             X=X.values[:skip_last_for_model],
#     #                             y=y.values[:skip_last_for_model],
#     #                             response_method='predict_proba',
#     #                             name=symbol,
#     #                             sample_weight=None)
#     # plt.savefig('../downloads/model_data/'+symbol+'_precision_recall_curve.jpg')
#     # clear_plot()
#     # plot_roc_curve(estimator=final_model,
#     #                X=X.values[:skip_last_for_model],
#     #                y=y.values[:skip_last_for_model],
#     #                drop_intermediate=False,
#     #                response_method='predict_proba',
#     #                name=symbol,
#     #                sample_weight=None)
#     # plt.savefig('../downloads/model_data/'+symbol+'_roc_curve.jpg')
#     clear_plot()
#
#
# # print('best parameters: ', pd.concat(best_parameters))
# # print('all parameters: ', pd.concat(all_parameters))
# print('score matrices: ', score_matrices.fillna(0))
# print('accuracy scores: ', accuracy_scores)
#
# ### Save all score data
# score_matrices.reset_index(drop=False).fillna(0).to_csv('../downloads/model_data/all_score_matrix_'+date+'.tsv', sep='\t')
# pd.DataFrame(accuracy_scores, index=[scoring_type]).to_csv('../downloads/model_data/all_accuracy_scores_'+date+'.tsv', sep='\t')
#
# ### Save all feature importance data
# feature_importance_df = reduce(lambda x, y: pd.merge(x, y, on='name', how='outer'), all_feature_importance).fillna(0)
#
#
# ### Plot feature importance
# clear_plot()
# cmap = sns.color_palette("rocket", as_cmap=True)
# ax = sns.heatmap(feature_importance_df, vmin=0, vmax=1, cmap=cmap)
# ax.figure.tight_layout()
#
# plt.savefig('../downloads/model_data/all_feature_importance_'+date+'.jpg')
# clear_plot()
#
# ### Save all best parameters
# pd.concat(all_parameters).sort_values(['rank_test_score', 'mean_test_score']).to_csv('../downloads/model_data/all_symbol_results_'+date+'.tsv', sep='\t')
# best_parameters = pd.concat(best_parameters).sort_values(['test_score']).fillna(0)
# best_parameters.to_csv('../downloads/model_data/best_param_results_'+date+'.tsv', sep='\t')
#
# ### Plot best parameters
# clear_plot()
# ax = sns.stripplot(data=best_parameters, x='result', y='y_prob')
# ax.figure.tight_layout()
# plt.savefig('../downloads/model_data/best_parameters_'+date+'.jpg')
#
