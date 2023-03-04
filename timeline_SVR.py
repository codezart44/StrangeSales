
# imports
import numpy as np
import pandas as pd
import pickle as pl


import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.cluster import DBSCAN
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler


# functions and definitions

def conf_int(series:list | np.ndarray, λ_ɑ2:float=1.96) -> tuple:
    '''
    ## Create confidence interval for dataseries
    '''
    µ = np.mean(series)
    s = np.std(series)
    a1, a2 = µ-λ_ɑ2*s, µ+λ_ɑ2*s
    I_µ = [a1, a2]

    return (µ, s, I_µ)





# data retreival

srcfile_arrbr = 'SAR'
with open(f'PICKLED/df_{srcfile_arrbr}_FldKey.pl', 'rb') as f:
    package = pl.load(f)

df_FldKey: pd.DataFrame = package[0]
FldKey: str = package[1]




for value_set in df_FldKey.FldVal.unique():                     # value_set (sv. värdeförråd)
    df_value_set = df_FldKey.query('FldVal==@value_set')
    X = df_value_set.Ordinal.values
    y = df_value_set.RowCnt.values


    sc_X = StandardScaler(with_mean=True, with_std=True)
    sc_y = StandardScaler(with_mean=True, with_std=True)
    X_scaled = sc_X.fit_transform(X.reshape(-1, 1))
    y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

    svr_rbf = SVR(kernel='rbf', C=100, gamma=4, epsilon=0.1)
    svr_lin = SVR(kernel='linear', C=100, gamma=2)
    svr_poly = SVR(kernel='poly', C=100, gamma=2, degree=3, epsilon=0.1, coef0=1)


    svr_rbf.fit(X_scaled, y_scaled.ravel())
    svr_lin.fit(X_scaled, y_scaled.ravel())
    svr_poly.fit(X_scaled, y_scaled.ravel())


    y_pred_rbf = svr_rbf.predict(X_scaled)
    y_pred_lin = svr_lin.predict(X_scaled)
    y_pred_poly = svr_poly.predict(X_scaled)

    # get residuals for SVR_RBF fitted curve and observed data points 
    resid_rbf = y_scaled.ravel()-y_pred_rbf


    λ_ɑ2 = 2.58
    µg, S, I_µg = conf_int(series=resid_rbf, λ_ɑ2=λ_ɑ2)     # NOTE global confidence interval for rbf residuals
    pred_glo = [1 if I_µg[0] < x < I_µg[1] else -1 for x in resid_rbf]
    # r0 = resid[(I_µ[0] < resid) & (resid < I_µ[1])]


    
    ## Local confidence interval (point specific)
    ## Calc local uncertainty for each point


    pt_range = 5           # range stretching back- and forward, 2n+1 points in total
    X_inl, y_inl = np.array([]), np.array([])
    X_out, y_out = np.array([]), np.array([])
    s_arr = np.array([])

    for i, x in enumerate(X_scaled):
        y = y_scaled[i]
        y_rbf = y_pred_rbf[i]


        lower_lt = i-pt_range if i-pt_range>=0 else 0
        upper_lt = i+pt_range+1 if i+pt_range+1<=len(resid_rbf) else len(resid_rbf)+1
        pts = resid_rbf[lower_lt: upper_lt]
        pts = list(filter(lambda r: I_µg[0] <= r <= I_µg[1], pts))            # filter out points outside of global confidence interval when calc local confidence interval

        
        # local confidence interval for rbf residuals
        µl, s, I_µl = conf_int(series=pts, λ_ɑ2=λ_ɑ2)     # NOTE local confidenc interval for residuals
        s_arr = np.r_[s_arr, s]


        # plot residuals for points deemed as outliers
        if y_rbf-λ_ɑ2*s <= y <= y_rbf+λ_ɑ2*s:
            X_inl = np.r_[X_inl, x]
            y_inl = np.r_[y_inl, y]
        else:
            X_out = np.r_[X_out, x]
            y_out = np.r_[y_out, y]

    

    ## Forecast of amt sales series (abbr. fc)
    X_fc = X
    forecast_len = 10
    for _ in range(forecast_len):
        X_fc = np.r_[X_fc, X_fc[-1]+1]
    X_fc_scaled = sc_X.transform(X_fc.reshape(-1, 1))
    
    X_inl = X_inl.reshape(-1, 1)

    variance = S**2
    kernel = RBF(length_scale=20, length_scale_bounds=(1e-3, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=variance, n_restarts_optimizer=8)
    gpr.fit(X_inl, y_inl)
    µ_fc, S_fc = gpr.predict(X_fc_scaled, return_std=True)



    # Plotting - two graphs: global conf, local conf

    ##  FIG  ##
    fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    fig.tight_layout(pad=5.0)


    ##  AX1  ##
    # Global confidence interval
    ax1.scatter(X_scaled, y_scaled, c=pred_glo)
    ax1.plot(X_scaled, y_pred_rbf-λ_ɑ2*S, c='k', ls='--')                          # upper limit of 95% confidence interval
    ax1.plot(X_scaled, y_pred_rbf+λ_ɑ2*S, c='k', ls='--')                          # lower limit of 95% confidence interval
    ax1.plot(X_scaled, y_pred_rbf, label='radial basis function', ls=':', c='r')      # estimated trend of data points
    ax1.plot(X_scaled, y_pred_lin, label='linear function', ls='--', c='c')
    ax1.plot(X_scaled, y_pred_poly, label='polynomial function', ls='-', c='m')
    for i, x in enumerate(X_scaled):
        if pred_glo[i] == -1:
            ax1.plot([x, x], [y_scaled.ravel()[i], y_pred_rbf[i]], c='r', lw=0.8, ls='--', zorder=-1)

    ax1.set_title(f'RowCnt timeseries for ${FldKey}$ ${value_set}$ w. global conf')
    ax1.set_xlabel('$Ordinal$')
    ax1.set_ylabel('$RowCnt$')
    ax1.legend()


    
    ##  AX2  ##
    # plotting local confidence interval (for each point)
    for i, x in enumerate(X_scaled):
        y = y_scaled[i]
        y_rbf = y_pred_rbf[i]
        s = s_arr[i]
        ax2.plot([x, x], [y_rbf-λ_ɑ2*s, y_rbf+λ_ɑ2*s], c='y', ls=':')
        if x in X_out:
            ax2.plot([x, x], [y, y_rbf], c='r', ls='--', lw=0.8)                # plot residuals for points deemed as outliers



    # GPR forecast line with mean and conf int as upper lower bounds of fill
    ax2.plot(X_fc_scaled, µ_fc, c='m', ls='--', label='gpr pred w/o noise')
    ax2.fill_between(
        x=X_fc_scaled.ravel(), y1=µ_fc-λ_ɑ2*S_fc, y2=µ_fc+λ_ɑ2*S_fc,
        color='black', alpha=0.1, ls='-.', label='gpr conf int'
        )
    


    # SVR fitted curves with different kernels (rbf, lin, poly)
    ax2.plot(X_scaled, y_pred_rbf, c='r', ls=':', label='rbf pred')
    ax2.plot(X_scaled, y_pred_lin, c='c', ls='--', label='linear pred')
    ax2.plot(X_scaled, y_pred_poly, c='m', ls='-', label='polynomial pred')


    ax2.scatter(X_scaled, y_scaled, c='k', s=20, label='data points')

    # Other
    ax2.legend()
    ax2.set_title(f'RowCnt timeseries for ${FldKey}$ ${value_set}$ w. local conf')
    ax2.set_xlabel('$Ordinal$')
    ax2.set_ylabel('$RowCnt$')
    plt.show()








# NOTE!
# Dots outside of global conf are removed
# Local conf is calculated over pts within pt_range
# Pros: Tolerance scales with variance (letting more points slide if variance is expected to be high)
#   Good for avoiding false positives and 






### PLAN
# Use GPR to fill out missing points (label as generated / prsphetic)
# Use DBSCAN with large eps (generous) to identify it there are more than one cluster ==> wont be able to fit trendline (already indicative of outlier, strange behavoiur eg 1000)
# Use SVR_RBF / LIN / POLY as well as LOWESS to fit curve to points, calc residuals
# Find average, std and distribution and of residuals and identify residuals that deviate more than normal (this is a key param for this composite model)
# Points deemed as deviating too much are outliars, which there are a handful of in the amount series

## NOTE
# Calculate difference (residual) at each point find RMSE and minimize that with CVGS (cross validation grid search)
# Find optimal params for svr_rbf and 







# SCRAP

# min_idx = np.argmin([I_μg[1]-I_μg[0], I_μl[1]-I_μl[0]])
# min_I_µ = [I_μg, I_μl][min_idx]
# pred = 1 if min_I_μ[0] < resid[i] < min_I_μ[1] else -1
# labels.append(pred)
# if pred == -1:
#     plt.plot([x, x], [y_scaled[i], y_pred_rbf[i]], c='r', ls='--', lw=0.8)


# gbl_conf = [y_pred_rbf[i]-λ_ɑ2*D, y_pred_rbf[i]+λ_ɑ2*D]
# lcl_conf = [y_pred_rbf[i]-λ_ɑ2*d, y_pred_rbf[i]+λ_ɑ2*d]
# min_idx = np.argmin([gbl_conf[1]-gbl_conf[0], lcl_conf[1]-lcl_conf[0]])
# min_conf = [gbl_conf, lcl_conf][min_idx]
# c = 'k' if min_idx==0 else 'y'




# df_trends = df_amount.query('Prevalence==1').reset_index(drop=True)        # remove short trends
# df_bad_trends = df_amount.query('Prevalence<0.5').reset_index(drop=True)        # keep only short trends


# for fdk in df_CSD_FldKey.FldVal.unique():
#     df_fdk = df_CSD_FldKey.query('FldVal==@fdk')
#     X = df_fdk.Ordinal.values
#     y = df_fdk.Count.values



# ## plot DBSCAN cluster are descision boundary
        # n = 20                              # number of edges for the circle
        # t = np.linspace(0, 2*np.pi, n+1)
        # a = eps*np.cos(t)+x
        # b = eps*np.sin(t)+y
        # ax2.plot(a, b, c='r', lw=0.5)


# sns.kdeplot(ax=ax1, data=resid_rbf, color='m', label='Normal distribution of residuals', fill=True)
    # ax1.axvline(x=I_µg[0], c='b', ls='--')                 # plt.axhline() for horisontal line
    # ax1.axvline(x=I_µg[1], c='b', ls='--')
    # ax1.set_title('residuals GaussianDistr')
    # ax1.legend()