import pandas as pd
import json

# from sklearn.decomposition import PCA
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.cluster import k_means, spectral_clustering
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit, RandomizedSearchCV
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_theme(style="white")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# date = '20201220'

def clear_plot():
    plt.clf()
    plt.cla()
    plt.close('all')


### Get all symbols from parquet directory
# symbols = [i.split('_')[0] for i in os.listdir('../downloads/models/')]
symbols = ['PLAY', 'REGN', 'ALNY', 'ACAD', 'AMAT', 'SIMO', 'AAL', 'INO',
           'HEXO', 'IGC', 'ACB', 'JMIA', 'PFE', 'SNDL', 'MRNA', 'CIDM', 'MTSI']

drop_cols = ['open', 'high', 'low', 'close', 'volume_d', 'volume_w', 'volume_m', 'week', 'month', 'year', 'type',
             'ohlc4_d', 'ha_open_d', 'ha_high_d', 'ha_low_d', 'ha_close_d',
             'ohlc4_w', 'ha_open_w', 'ha_high_w', 'ha_low_w', 'ha_close_w',
             'ohlc4_m', 'ha_open_m', 'ha_high_m', 'ha_low_m', 'ha_close_m',
             'future_1_day_maxhigh_rel', 'future_3_day_maxhigh_rel', 'future_5_day_maxhigh_rel',
             'future_10_day_maxhigh_rel', 'future_20_day_maxhigh_rel', 'future_30_day_maxhigh_rel',
             'future_1_day_maxclose_rel', 'future_3_day_maxclose_rel', 'future_5_day_maxclose_rel',
             'future_10_day_maxclose_rel', 'future_20_day_maxclose_rel', 'future_30_day_maxclose_rel',
             'habs_d_buy_streak_pos', 'habs_d_buy_streak_change', 'habs_d_buy_streak_peak',
             'habs_d_buy_streak_len', 'habs_w_buy_streak_pos', 'habs_w_buy_streak_change',
             'habs_w_buy_streak_peak', 'habs_w_buy_streak_len', 'habs_d_sell_streak_pos',
             'habs_d_sell_streak_change', 'habs_d_sell_streak_peak', 'habs_d_sell_streak_len']

# ### Create Scalers
# minmax_scaler = MinMaxScaler(feature_range=[-1, 1])
# quantile_scaler_bins = 5
# qtransform_scaler = QuantileTransformer(n_quantiles=quantile_scaler_bins)
# standard_scaler = StandardScaler()
# robust_scaler = RobustScaler()
# kbins_discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
# power_transformer = PowerTransformer()

# ### Start date for model building and identity of test column
# start_date = '2016-01-01'
# y_col = 'test_col'
# target_filter_col = 'habs_d_buy_streak_pos'
# target_filter_val = 1
# target_col = 'future_10_day_maxclose_rel'#'habs_d_buy_streak_change' #'habs_d_buy_streak_len' 'habs_d_buy_streak_high'
# min_days_for_model = 300
# test_value = 1.1
# feature_correlation_cutoff = .05

### Dict to hold predictions
all_rows = []

### Number of symbols to process
number_of_symbols = 1
days_to_last_signal = 10
target_filter_col = 'habs_d_buy_streak_pos'
target_filter_val = 1
target_col = 'habs_d_buy_streak_len'
label_col = 'datetime'

### Load column types
with open('../downloads/model_data/feature_types.json', 'rb') as file:
    hist_type_inverted = json.load(file)
print('hist_type_inverted', hist_type_inverted)

### Load models
minmax_scaler = pickle.load(open('../downloads/models/cat_minmax_scaler.pkl', 'rb'))
power_transformer = pickle.load(open('../downloads/models/cont_power_transformer.pkl', 'rb'))
feature_select = pickle.load(open('../downloads/models/feature_select_model.pkl', 'rb'))
best_pipeline = pickle.load(open('../downloads/models/general_model.pkl', 'rb'))
for symbol in symbols[:number_of_symbols]:

    print('Processing symbol:', symbol)
    ### Read in the engineered data for the symbol
    ts_main = pd.read_parquet(f'../downloads/ml_dataengineering.parquet/symbol={symbol}/')

    X = None
    for i in range(days_to_last_signal):
        row_index = len(ts_main) - 1 - i
        print(ts_main.iloc[row_index][target_filter_col])
        if ts_main.iloc[row_index][target_filter_col] == -target_filter_val:
            break
        if ts_main.iloc[row_index][target_filter_col] == target_filter_val:
            X = ts_main.loc[row_index:row_index, :].reset_index(drop=True)
            break
    if X is not None:
        # label = str(X[label_col][0].strftime('%Y-%m-%d'))
        # X['label'] = label
        X['symbol'] = symbol
        all_rows.append(X)

X = pd.concat(all_rows, axis=1)
# print('all_rows', all_rows)
# print('all_rows columns', all_rows.columns)
# print('all_rows columns length', len(all_rows.columns))
# print('drop cols', drop_cols)
# print('drop cols length', len(drop_cols))
results = X.copy(deep=True).loc[:, ['datetime', 'symbol']]
X = X.drop(drop_cols, axis=1)
X = X.drop('symbol', axis=1)
columns_sets = []
if 'cat' in hist_type_inverted.keys():
    X_cat = X.loc[:, hist_type_inverted['cat']].copy(deep=True)
    X_cat[hist_type_inverted['cat']] = minmax_scaler.transform(X_cat)
    columns_sets.append(X_cat)

if 'other' in hist_type_inverted.keys():
    X_other = X.loc[:, hist_type_inverted['other']].copy(deep=True)
    X_other[hist_type_inverted['other']] = power_transformer.transform(X_other)
    columns_sets.append(X_other)

X = pd.concat(columns_sets, axis=1)
# X_cols = X.columns
X = feature_select.transform(X.values)
y_pred = best_pipeline.predict(X)

# print('X cols: ', X_cols)
# print('X cols len: ', len(X_cols))
print('all_rows', X)
print('all_rows num rows', X.shape[0])
print('all_rows num columns', X.shape[1])
print('all_rows', X)
results['y_pred'] = y_pred
print('results', results)

[4][6]
# print(f'Prediction for {symbol} on {label}:', model.predict(X.values))
# else:
# print(f'{symbol} does not have a buy signal in the past {days_to_last_signal} days')

[4][6]

#     ###### GET TEST ROWS AND CREATE TEST COLUMN
#     ### If there is less than n rows for a symbol, do not build a model
#     if len(ts_main) < min_days_for_model:
#         continue
#
#     ### Filter data points from start date with datetime and
#     ts_main = ts_main[ts_main['datetime'] >= start_date]
#     ts_main = ts_main[ts_main[target_filter_col] == target_filter_val]
#
#     ### Set the datetime as the index column
#     ts_main = ts_main.reset_index(drop=True)
#     ts_main = ts_main.sort_values('datetime', ascending=True).set_index('datetime')
#
#     ### Create the y series
#     y_col_series = ts_main[target_col].apply(lambda x: 1 if x >= test_value else 0)
#
#     ###### DATA FORMATTING
#     ### Delete unneeded columns
#
#     # print(drop_cols)
#     ts_main = ts_main.drop(drop_cols, axis=1)
#
#     ### Sort the columns by name
#     ts_main = ts_main.loc[:, sorted(list(ts_main.columns))]
#
#     ###### FEATURE SCALING
#     ### Plot histograms of unscaled features
#     plot_histograms(ts_main, out_file=symbol+'_pre_normalize.jpg', ncols=5)
#
#     ### Get the distribution types for each feature
#     hist_type = get_hist_type(ts_main)
#     hist_type_inverted = dict()
#     for key, value in hist_type.items():
#         hist_type_inverted.setdefault(value, list()).append(key)
#
#     ### Scale the features
#     ts_main_cat = ts_main.loc[:, hist_type_inverted['2']+hist_type_inverted['3']].copy(deep=True)
#     ts_main_cat[hist_type_inverted['2']+hist_type_inverted['3']] = minmax_scaler.fit_transform(ts_main_cat)
#     ts_main_nonnormal = ts_main.loc[:, hist_type_inverted['skewed']+hist_type_inverted['other']+hist_type_inverted['normal']].copy(deep=True)
#     ts_main_nonnormal[hist_type_inverted['skewed']+hist_type_inverted['other']+hist_type_inverted['normal']] = robust_scaler.fit_transform(ts_main_nonnormal)
#     ts_main = pd.concat([ts_main_cat, ts_main_nonnormal], axis=1)
#
#     ### Plot the scaled histograms
#     plot_histograms(ts_main, out_file=symbol+'_post_normalize.jpg', ncols=5)
#
#     ### Perform scaling of all columns, if needed. Otherwise, just rename main data
#     df_scaled = ts_main.copy(deep=True)
#     # df_scaled[ts_main.columns] = scaler.fit_transform(ts_main)
#
#     ### Run the PCA
#     # n_components = 15
#     # pca = PCA(n_components=n_components)
#     # pca.fit(df_scaled)
#     # print('pca explained_variance_: ', pca.explained_variance_)
#     # print('pca explained_variance_ratio_: ', pca.explained_variance_ratio_)
#     # X_ = pca.transform(df_scaled)
#     # df_scaled = pd.DataFrame(X_, columns=list(range(1, n_components + 1)))
#
#     ### Add the unscaled y data back to the main dataframe
#     df_scaled['TEST'] = y_col_series
#
#     ### Show correlations between features and y; Remove features with low correlation
#     feat_corr = feature_correlations(df=df_scaled, y='TEST')
#     feat_corr = feat_corr[abs(feat_corr['corr']) >= feature_correlation_cutoff]
#     feat_corr.to_csv('../downloads/model_data/'+symbol+'_feature_correlations.tsv', sep='\t', index=False)
#     df_scaled = df_scaled.loc[:, list(feat_corr['index'])+['TEST']]
#     clear_plot()
#     sns.barplot(data=feat_corr, x='corr', y='index')
#     plt.autoscale()
#     plt.savefig('../downloads/model_data/'+symbol+'_feature_correlations.jpg')
#     clear_plot()
#
#     ### Create a correlation matrix to view relationships between features
#     feature_correlation_matrix(df=df_scaled, file=symbol+'_feature_matrix')
#     clear_plot()
#     df_scaled.corr(method='pearson').reset_index(drop=False).melt(
#         id_vars='index', value_name='corr').sort_values('corr').to_csv(path_or_buf='../downloads/model_data/'+symbol+'_feature_corr.tsv', sep='\t')
#
#     ### Add the row data to the data dictionary
#     all_streaks[symbol] = df_scaled
#
#
# # ###### Choose the parameters
# # ### Define the random forest classifier
# # params = {'n_estimators': [10, 20, 30, 100, 200],
# #           'criterion': ['gini', 'entropy'],
# #           'max_depth': [None, 5, 10, 30],
# #           'min_samples_split': [2, 5, 10, 15, 30],
# #           'min_samples_leaf': [1, 5, 10, 20],
# #           'min_weight_fraction_leaf': [0, .1, .25, .50],
# #           'max_features': [None, 'sqrt', 'log2', .1, .25, .5, .75],
# #           'max_leaf_nodes': [None, 25, 50],
# #           'min_impurity_decrease': [0, .1, .2],
# #           'bootstrap': [False],
# #           # 'oob_score': [False, True],
# #           'random_state': [99],
# #           'class_weight': ['balanced', 'balanced_subsample', None],
# #           'ccp_alpha': [0, .1, .25],
# #           'max_samples': [None]}
#
# # ### Random Forest Classifier
# # params = {'n_estimators': [10, 25, 75, 150],
# #           'criterion': ['gini', 'entropy'],
# #           'max_depth': [None, 5, 15],
# #           'min_samples_split': [2, 5, 10],
# #           'min_samples_leaf': [1, 5, 10],
# #           'min_weight_fraction_leaf': [0, .05, .1, .15, .25],
# #           'max_features': [None, 'sqrt', 'log2', .1, .5],
# #           'max_leaf_nodes': [None, 25, 50],
# #           'min_impurity_decrease': [.01, .05, .1, .2],
# #           'bootstrap': [False],
# #           'oob_score': [False],
# #           'class_weight': ['balanced'],
# #           'ccp_alpha': [0, .01, .1, .2, .3],
# #           'max_samples': [None]}
#
# ### Random Forest Classifier
# params = {'n_estimators': [25, 50, 100, 150],
#           'criterion': ['gini', 'entropy'],
#           'max_depth': [None, 20],
#           'min_samples_split': [2, 8, 16],
#           'min_samples_leaf': [1, 6, 12],
#           'min_weight_fraction_leaf': [0, .01],
#           'max_features': [None, .5],
#           'max_leaf_nodes': [None, 50],
#           'min_impurity_decrease': [0, .01, .05],
#           'bootstrap': [False],
#           'oob_score': [False],
#           'class_weight': ['balanced'],
#           'ccp_alpha': [0],
#           'max_samples': [None]}
#
# # ### Gradient Boosting Classifier
# # params = {'loss': ['deviance', 'exponential'],
# #           'learning_rate': [.1, .05, .01, .005, .001],
# #           'n_estimators': [10, 50, 100, 150, 200],
# #           'subsample': [.8, .9, 1],
# #           'criterion': ['friedman_mse', 'mse', 'mae'],
# #           'max_depth': [3, 5, 10],
# #           'min_samples_split': [2],
# #           'min_samples_leaf': [1],
# #           'min_weight_fraction_leaf': [0],
# #           'min_impurity_decrease': [0],
# #           'init': ['zero', None],
# #           'random_state': [99],
# #           'max_features': [None, 'sqrt'],
# #           'max_leaf_nodes': [5, 20, None],
# #           'n_iter_no_change': [None],
# #           'tol': [.0001],
# #           'ccp_alpha': [0, .01, .1]}
#
# # ### Histogram-based gradient classifier
# # params = {'early_stopping': [False],
# #           'l2_regularization': [0.0],
# #           'learning_rate': [0.1, .2, .5, 1],
# #           'loss': ['binary_crossentropy'],
# #           'max_bins': [50, 100, 255],
# #           'max_depth': [5, 20, None],
# #           'max_iter': [50, 100, 200],
# #           'max_leaf_nodes': [15, 31, None],
# #           'min_samples_leaf': [5, 50, 100],
# #           'monotonic_cst': [None],
# #           'n_iter_no_change': [10],
# #           'random_state': [99],
# #           'scoring': ['f1'],
# #           'tol': [1e-07],
# #           'validation_fraction': [0.1]}
#
# # ### Perceptron classifier
# # params = {'penalty': ['l2', 'l1', 'elasticnet', None],
# #           'alpha': [.00001, .0001, .001, .01],
# #           'fit_intercept': [True, False],
# #           'max_iter': [200, 1000, 5000],
# #           'tol': [None],
# #           'shuffle': [True, False],
# #           'eta0': [1, 2, 3],
# #           'random_state': [99],
# #           'early_stopping': [True],
# #           'class_weight': [None, 'balanced']}
#
# # ### Decision tree classifier parameters
# # params = {'criterion': ['gini', 'entropy'],
# #           'splitter': ['best', 'random'],
# #           'max_depth': [None, 5, 20],
# #           'min_samples_split': [2, 5, 10, 25],
# #           'min_samples_leaf': [1, 5, 10, 25],
# #           'min_weight_fraction_leaf': [0, .1, .25],
# #           'max_features': [None, 'auto', 'log2', 5, 25],
# #           'max_leaf_nodes': [None],
# #           'min_impurity_decrease': [0],
# #           'random_state': [69],
# #           'class_weight': [None, 'balanced'],
# #           'ccp_alpha': [0, .1, .25, .5]}
#
# # # ### ADABoost classifier parameters
# # params = {'base_estimator': [None],
# #           'n_estimators': [25, 50, 100, 200],
# #           'learning_rate': [.25, .5, .75, 1, 1.25, 1.5, 2],
# #           'algorithm': ['SAMME', 'SAMME.R'],
# #           'random_state': [99]}
#
# ###### SPLIT THE DATA INTO TRAIN AND TEST
# ### Do a time series split
# test_train_split = .80
# cv_splits_inner = 3
# n_random_params = 250
# skip_last_for_model = None # -1
# scoring_type = 'f1'
# # scoring_type = 'balanced_accuracy'
# # scoring_type = 'average_precision'
#
# ### Calculate number of possible parameter sets
# print('Number of unique param combinations: ', reduce(lambda x, y: x*y, [len(values) for key, values in params.items()]))
#
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
#
# ###### RUN THE MODEL CREATION
# for symbol, df in all_streaks.items():
#     print('Building model for symbol:', datetime.datetime.now(), symbol)
#     ### Split the dataset into X and y data
#     X = df.loc[:, df.columns[:-1]]#.values
#     y = df[df.columns[-1:][0]]#.values
#
#     ### Create test train indicies
#     test_train_split_index = int(test_train_split * len(X))
#     train_index = list(range(len(X)))[:test_train_split_index]
#     test_index = list(range(len(X)))[test_train_split_index:]
#
#     ### Use the indices to create X and y train and test sets using 1 of the splits
#     X_train = np.array([X.iloc[i, :] for i in train_index]).astype(float)
#     X_test = np.array([X.iloc[i, :] for i in test_index]).astype(float)
#     y_train = np.array([y.iloc[i] for i in train_index]).astype(int)
#     y_test = np.array([y.iloc[i] for i in test_index]).astype(int)
#
#     ### Select the model to use
#     # model = Perceptron()
#     model = RandomForestClassifier()
#     # model = HistGradientBoostingClassifier()
#     # model = GradientBoostingClassifier()
#     # model = Perceptron()
#     # model = DecisionTreeClassifier()
#     # model = AdaBoostClassifier()
#     # print(model.get_params())
#     best_model = model
#     final_model = model
#
#     ### Create the randomized search to find the best parameters
#     clf = RandomizedSearchCV(estimator=model,
#                              param_distributions=params,
#                              scoring=scoring_type,
#                              n_iter=n_random_params,
#                              n_jobs=1,
#                              cv=TimeSeriesSplit(n_splits=cv_splits_inner),
#                              verbose=0,
#                              return_train_score=True).fit(X_train, y_train)
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
#     sns.swarmplot(data=best_param_df, x='result', y='y_prob')
#     plt.autoscale()
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
#     pickle.dump(final_model, open('../downloads/models/'+symbol+'_model.pkl', 'wb'))
#     clear_plot()
#     plot_precision_recall_curve(estimator=final_model,
#                                 X=X.values[:skip_last_for_model],
#                                 y=y.values[:skip_last_for_model],
#                                 response_method='predict_proba',
#                                 name=symbol,
#                                 sample_weight=None)
#     plt.savefig('../downloads/model_data/'+symbol+'_precision_recall_curve.jpg')
#     clear_plot()
#     plot_roc_curve(estimator=final_model,
#                    X=X.values[:skip_last_for_model],
#                    y=y.values[:skip_last_for_model],
#                    drop_intermediate=False,
#                    response_method='predict_proba',
#                    name=symbol,
#                    sample_weight=None)
#     plt.savefig('../downloads/model_data/'+symbol+'_roc_curve.jpg')
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
# feature_importance_df.to_csv('../downloads/model_data/all_feature_importance_'+date+'.tsv', sep='\t')
#
# ### Plot feature importance
# clear_plot()
# cmap = sns.color_palette("rocket", as_cmap=True)
# sns.heatmap(feature_importance_df, vmin=0, vmax=1, cmap=cmap)
# plt.autoscale()
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
# sns.stripplot(data=best_parameters, x='result', y='y_prob')
# sns.violinplot(data=best_parameters, x='result', y='y_prob')
# plt.autoscale()
# plt.savefig('../downloads/model_data/best_parameters_'+date+'.jpg')
