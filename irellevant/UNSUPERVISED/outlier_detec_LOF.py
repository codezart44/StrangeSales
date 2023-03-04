
# IMPORTS
import numpy as np
import pandas as pd
import pickle as pl

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV





with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)


X = np.c_[df_amount.FldVal, df_amount.RowCnt]
# X_train, X_test = train_test_split(X, train_size=0.7, random_state=1)


clf = LocalOutlierFactor(n_neighbors=100, novelty=False, contamination=0.05)
y_pred = clf.fit_predict(X)

df_res = pd.DataFrame(data=zip(X[:,0], X[:,1], y_pred), columns=['amount', 'rowcnt', 'pred'])
print(df_res)



df_anom = df_res.query('pred==-1')
df_true = df_res.query('pred==1')

# xx, yy = np.meshgrid(
#     np.linspace(min(df_amount.FldVal)*1.05, max(df_amount.FldVal)*1.05, 50),
#     np.linspace(min(df_amount.RowCnt)*0.95, max(df_amount.RowCnt)*1.05, 50),
# )
# Z = clf.fit_predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.bone)

inl = plt.scatter(df_anom.amount, df_anom.rowcnt, s=20, c='red', edgecolors='k')
out = plt.scatter(df_true.amount, df_true.rowcnt, s=20, c='black', edgecolors='k')

# plt.legend(
#     # [inl, out],
#     labels = ['inliers', 'outliers'],
#     loc='upper left'
# )
# plt.title('Decision function as altitude levels \nof Isolation Forest')
# plt.xlabel('amount')
# plt.ylabel('rowcnt')
plt.show()






