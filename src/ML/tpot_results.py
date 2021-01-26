import itertools

# from sklearn.decomposition import PCA
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.cluster import k_means, spectral_clustering
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, OneHotEncoder, RobustScaler, KBinsDiscretizer
# from functools import reduce
import matplotlib
import numpy as np
import pandas as pd

# from tpot import TPOTClassifier
# from tpot import TPOTRegressor
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# sns.set_theme(style="white")
# sns.set_style(style='darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

results = pd.read_csv('../downloads/model_data/results.tsv', sep='\t', index_col=0, header=0)
results['symbol'] = results['date'].apply(lambda x: x.split('_')[1])
results['date'] = results['date'].apply(lambda x: x.split('_')[0])
all_dates = pd.DataFrame(
    list(
        itertools.product(
            results['date'].drop_duplicates().tolist(),
            results['symbol'].drop_duplicates().tolist())), columns=['date', 'symbol'])
hist_results = results.copy(deep=True)
results = results.merge(all_dates, how='right', on=['date', 'symbol'])
results = results.sort_values('date', ascending=True)
# print(results)
all_symbols = results['symbol'].drop_duplicates().tolist()
# print(all_symbols)
### Plot the y vs y_hat
fig, axs = plt.subplots(ncols=1, nrows=len(all_symbols), sharex=True, sharey=False, figsize=(30, 60))
for i, symbol in enumerate(all_symbols):
    print(i, symbol)
    data = results.loc[results['symbol'] == symbol, :].fillna(0)
    data.loc[:, 'SMA'] = data['d'].rolling(center=True, window=20, min_periods=1).mean() * 12
    axs[i].bar(data=data, height='d', x='date', color='black')
    axs[i].plot(data['date'], data['SMA'], color='red', ls='-', linewidth=1)
    axs[i].set_label(symbol)
    axs[i].set_ylim(bottom=min(results['d']) * 1.2, top=max(results['d']) * 1.2)
    axs[i].annotate(text=symbol, xy=(max(results['d']), 5))
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../downloads/model_data/delta_by_symbol.jpg')
# plt.show()


cmap = LinearSegmentedColormap.from_list('hot_or_cold', ['blue', 'yellow', 'red'], 256)
### Test deviation heatmap
hm_results = results.pivot(index='symbol', columns='date', values='d')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(60, 30))
im = ax.imshow(hm_results, cmap=cmap)
cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)
ax.set_xticks(np.arange(hm_results.shape[1]))
ax.set_xticklabels(hm_results.columns)
ax.set_yticks(np.arange(hm_results.shape[0]))
ax.set_yticklabels(hm_results.index)
plt.title('Predicted minus Actual streak')
ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
plt.setp(ax.get_xticklabels(), rotation=90)  # , ha="right", rotation_mode="anchor")
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
# plt.tight_layout(c)
plt.savefig('../downloads/model_data/delta_heatmap.jpg')
# plt.show()

### Plot the histogram of expected results
# results_grid = results.groupby(['y_pred', 'y_test']).count()

hist_results['y_pred'] = hist_results['y_pred'].apply(lambda x: int(np.floor(x)))
hist_results['y_test'] = hist_results['y_test'].apply(lambda x: int(x))
all_predictions = hist_results['y_pred'].drop_duplicates().sort_values().tolist()
print(hist_results)
fig, axs = plt.subplots(ncols=1, nrows=len(all_predictions), sharex=True, sharey=False, figsize=(8, 30))
for i, val in enumerate(all_predictions):
    print(i, val)
    data = hist_results.loc[hist_results['y_pred'] == val, :]
    axs[i].hist(x=data['y_test'],
                bins=np.arange(0, int(max(data['y_test'].tolist())) + 2),
                density=False,
                color='black')
    axs[i].set_ylim(bottom=0, top=len(data))
    axs[i].set_xlim(left=1, right=max(hist_results['y_test']))
    axs[i].set_ylabel(str(val))
    axs[i].set_xticks(np.arange(1, int(max(data['y_test'].tolist())) + 2))
    axs[i].set_xticklabels(np.arange(1, int(max(data['y_test'].tolist())) + 2))
    # axs[i].annotate(text=val, xy=(.1, 5))
# plt.xlabel('Date')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../downloads/model_data/ypred_vs_ytest__histogram.jpg')
# plt.show()
