
# imports
import numpy as np
import pandas as pd
import pickle as pl

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL




with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)
df_trends = df_amount.query('Prevalence>0.5').reset_index(drop=True)        # remove short trends



dbscan = DBSCAN(eps=0.7, min_samples=3)
sc = StandardScaler(with_mean=True, with_std=True)




for amt in df_trends.FldVal.unique()[:]:
    df = df_amount.query('FldVal==@amt')[['Ordinal', 'RowCnt']].reset_index(drop=True)
    X_scaled = sc.fit_transform(df.values)

    # isof = IsolationForest(n_estimators=200, contamination=0.01)
    # y_pred = isof.fit_predict(X=X_scaled)

    dbscan.fit(X=X_scaled)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x=df.Ordinal, y=df.RowCnt, c=dbscan.labels_)
    ax.set_title(f'for amount {amt}')
    ax.set_xlim(-5, 95)
plt.show()

