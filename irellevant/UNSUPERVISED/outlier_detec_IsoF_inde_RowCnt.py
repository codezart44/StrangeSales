
# IMPORTS
import numpy as np
import pandas as pd
import pickle as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest


'''
NOTE
Ifall man anser att varje belopp (amount) är et normalt förekommande värde men vill undersöka hur
spridningen av RowCnt ser ut för det beloppet (identifiera FldVal-RowCnt par som är extrema ==> outlier)

'''




# practice groupby: https://sparkbyexamples.com/pandas/pandas-groupby-sum-examples/ 


with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)




### mean, median, min, max of RowCnt for each Amount
# ds_tot = df_amount.groupby('FldVal')

# ds_mean = ds_tot.mean('RowCnt')
# ds_min = ds_tot.min('RowCnt')
# ds_max = ds_tot.max('RowCnt')
# ds_median = ds_tot.median('RowCnt')


# df_info = pd.DataFrame(
#     data=np.c_[ds_mean, ds_median, ds_min, ds_max],
#     columns=['mean', 'min', 'max', 'median']
#     )
# print(df_info)

n_batches = df_amount.BatchInstId.nunique()

df_groups = df_amount.groupby('FldVal')['RowCnt'].apply(np.array)
print(df_groups)


for i in df_groups.index:
    X = df_groups.loc[i].reshape(-1, 1)

    # prevalence = len(X)/n_batches           # in what percentage of the batches does this amount occur
    # contamination = (1.1-prevalence)/2
    # print(i, prevalence, contamination)

    clf = IsolationForest(n_estimators=200, contamination=0.05, verbose=1)
    y_pred = clf.fit_predict(X)
    

    df_res = pd.DataFrame(data=np.c_[X.ravel(), y_pred], columns=['rowcnt', 'pred'])
    Xo = df_res.query('pred==-1').rowcnt
    Xi = df_res.query('pred==1').rowcnt
    plt.scatter([i]*len(Xo), Xo, c='red', s=20, edgecolors='k')
    plt.scatter([i]*len(Xi), Xi, c='black', s=20, edgecolors='k')
plt.show()




# ds_tot = df_amount.groupby('FldVal').sum('RowCnt')
# df_tot = pd.DataFrame(data=zip(ds_tot.index, ds_tot.values.ravel()), columns=['FldVal', 'RowCnt'])
# print(df_tot)


# plt.scatter(df_tot.FldVal, df_tot.RowCnt, c='black', s=20, edgecolors='k')
# plt.show()












