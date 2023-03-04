
# imports
import pickle as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def conf_int(series:list | np.ndarray, λ_ɑ2:float=1.96) -> tuple:
    '''
    ## Create confidence interval for dataseries
    '''
    µ = np.mean(series)
    s = np.std(series)
    a1, a2 = µ-λ_ɑ2*s, µ+λ_ɑ2*s
    I_µ = [a1, a2]

    return (µ, s, I_µ)



with open('PICKLED/df_SAR_FldKey.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)


df_trends = df_amount.query('Prevalence==1').reset_index(drop=True)        # remove short trends
df_bad_trends = df_amount.query('Prevalence<0.5').reset_index(drop=True)        # keep only short trends





for amt in df_amount.FldVal.unique():
    df_amt = df_amount.query('FldVal==@amt')

    ## PLOTTING & ANIMATION ##
    fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1, figsize=(12, 8))


    X = df_amt.Ordinal.values
    y = df_amt.RowCnt.values

    sc_X = StandardScaler(with_mean=True, with_std=True)
    sc_y = StandardScaler(with_mean=True, with_std=True)

    X_scaled = sc_X.fit_transform(X.reshape(-1, 1))
    y_scaled = sc_y.fit_transform(y.reshape(-1, 1))



    svr_rbf = SVR(kernel='rbf', C=100, gamma=4, epsilon=0.1)
    svr_lin = SVR(kernel='linear', C=100, gamma=2)
    svr_poly = SVR(kernel='poly', C=100, gamma=2, degree=3, epsilon=0.1, coef0=1)


    for i in range(df_amt.shape[0]-20):

        X_scaled_anim = X_scaled[:20+i]
        y_scaled_anim = y_scaled[:20+i]

        

        svr_rbf.fit(X_scaled_anim, y_scaled_anim.ravel())
        svr_lin.fit(X_scaled_anim, y_scaled_anim.ravel())
        svr_poly.fit(X_scaled_anim, y_scaled_anim.ravel())


        y_pred_rbf = svr_rbf.predict(X_scaled_anim)
        y_pred_lin = svr_lin.predict(X_scaled_anim)
        y_pred_poly = svr_poly.predict(X_scaled_anim)

        # get residuals for SVR_RBF fitted curve and observed data points 
        resid_rbf = y_scaled_anim.ravel()-y_pred_rbf
        resid_lin = y_scaled_anim.ravel()-y_pred_lin


        λ_ɑ2 = 2.58
        µg, S, I_µg = conf_int(series=resid_rbf, λ_ɑ2=λ_ɑ2)     # NOTE global confidence interval for rbf residuals
        pred_glo = [1 if I_µg[0] < x < I_µg[1] else -1 for x in resid_rbf]


        X_inl, y_inl = np.array([]), np.array([])
        X_out, y_out = np.array([]), np.array([])
        s_arr = np.array([])


        for i, x in enumerate(X_scaled_anim):
            y = y_scaled_anim[i]
            y_rbf = y_pred_rbf[i]

            pt_range = 5                                        # range stretching back- and forward, 2n+1 points in total
            lower_lt = i-pt_range if i-pt_range>=0 else 0
            upper_lt = i+pt_range+1 if i+pt_range+1<=len(resid_rbf) else len(resid_rbf)+1
            pts = resid_rbf[lower_lt: upper_lt]
            pts = list(filter(lambda r: I_µg[0] <= r <= I_µg[1], pts))      # filter out points outside of global confidence interval when calc local confidence interval

            
            # local confidence interval for rbf residuals
            µl, s, I_µl = conf_int(series=pts, λ_ɑ2=λ_ɑ2)       # NOTE local confidenc interval for residuals
            s_arr = np.r_[s_arr, s]


            # divide in- and outliers
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



        # plotting local confidence interval (for each point)
        for i, x in enumerate(X_scaled_anim):
            y = y_scaled_anim[i]
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
        ax2.plot(X_scaled_anim, y_pred_rbf, c='r', ls=':', label='rbf pred')
        ax2.plot(X_scaled_anim, y_pred_lin, c='c', ls='--', label='linear pred')
        ax2.plot(X_scaled_anim, y_pred_poly, c='m', ls='-', label='polynomial pred')

        ax2.scatter(X_scaled_anim, y_scaled_anim, c='k', s=20, label='data points')     # orig data points


        # ax2 window config
        ax2.legend()
        ax2.set_xlim(left=np.min(X_fc_scaled), right=np.max(X_fc_scaled))
        ax2.set_ylim(bottom=np.min(y_scaled)-1, top=np.max(y_scaled)+1)

        # animation
        plt.pause(0.15)
        ax1.cla()
        ax2.cla()

    plt.show()






