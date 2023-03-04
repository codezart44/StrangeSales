
# imports
import pickle as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)

df_trends = df_amount.query('Prevalence>0.5').reset_index(drop=True)        # remove short trends


noise_std = 0.75
kernel = RBF(length_scale=20, length_scale_bounds=(1e-7, 1e7))
gpr = GaussianProcessRegressor(
    kernel=kernel, 
    alpha=noise_std**2,
    n_restarts_optimizer=12
    )



for amt in df_amount.FldVal.unique():
    print(amt)
    df_amt = df_amount.query('FldVal==@amt')
    X = df_amt.Ordinal.values
    y = df_amt.RowCnt.values

    sc = StandardScaler(with_mean=True, with_std=True)
    X_scaled = sc.fit_transform(X.reshape(-1, 1))
    y_scaled = sc.fit_transform(y.reshape(-1, 1))

    size = int(1*df_amt.shape[0])
    all_idx = np.arange(y.shape[0])
    train_idx = np.random.choice(all_idx, size=size, replace=False)
    X_train_scaled, y_train_scaled = X_scaled[train_idx], y_scaled[train_idx]
    gpr.fit(X_train_scaled, y_train_scaled)

    mean_pred, std_pred = gpr.predict(X_scaled, return_std=True)            # also has return_cov if wanted

    # plotting
    fig = plt.figure(figsize=(15,8))
    plt.scatter(X_scaled, y_scaled, alpha=0.6, label='full original dataset')
    plt.scatter(X_train_scaled, y_train_scaled, c='r', label='training data points')
    # plt.plot(X_scaled, y_scaled, label=f'RowCnt timeseries for amt {amt}', ls=':')
    # plt.errorbar(
    #     X _train_scaled,
    #     y_train_scaled,
    #     noise_std,
    #     c='tab:blue',
    #     marker='.',
    #     label='choosen observations'
    # )
    plt.plot(X_scaled, mean_pred, label='Mean prediction', ls='--')
    plt.fill_between(
        X_scaled.ravel(),
        mean_pred - 1.96 * std_pred,        # λ_0.025 = 1.96, where ɑ = 1-0.95 ==> 95% confidence level
        mean_pred + 1.96 * std_pred,
        color='tab:orange',
        alpha=0.4,                          # probalby for opacity
        label=r'95% confidence level'
    )
    plt.legend()
    plt.xlabel('$batch Ordinal$')
    plt.ylabel('$RowCnt$')
    plt.title('GaussianProcessRegression on 20 random data points')
    plt.show()
