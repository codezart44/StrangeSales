
# IMPORTS
import pandas as pd
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV



# practice groupby: https://sparkbyexamples.com/pandas/pandas-groupby-sum-examples/ 


with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)


ds_rowcnt_grouped = df_amount.groupby('FldVal')['RowCnt'].apply(np.array)
ds_rowcnt_summed = df_amount.groupby('FldVal')['RowCnt'].sum()
ds_rowcnt_count = df_amount.groupby('FldVal')['RowCnt'].apply(np.count_nonzero)


n_batches = df_amount.BatchInstId.nunique()


ds_rowcnt_prev = ds_rowcnt_count.divide(n_batches)


amount = ds_rowcnt_prev.index                           # all unique amounts
total_rowcnt = ds_rowcnt_summed.values                  # rowcnt summed for each unique amount
prevalence = ds_rowcnt_prev.values                      # prevalence of each unique amount

X = np.c_[amount, prevalence, total_rowcnt]
X_train, X_test = train_test_split(X, test_size=0.3, random_state=2)

clf = IsolationForest(n_estimators=100, contamination=0.1, bootstrap=False)
y_pred = clf.fit_predict(X)

df_res = pd.DataFrame(data=zip(X[:, 0], X[:, 1], X[:, 2], y_pred), columns=['amount', 'prev', 'rowcnt', 'pred'])
print(df_res)


xi = df_res.query('pred==1')[['amount', 'prev']].values
xo = df_res.query('pred==-1')[['amount', 'prev']].values


plt.scatter(xi[:, 0], xi[:, 1], c='black', s=20, edgecolors='k')
plt.scatter(xo[:, 0], xo[:, 1], c='red', s=20, edgecolors='k')
plt.show()




