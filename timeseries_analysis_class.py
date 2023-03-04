
# imports
import numpy as np
import pandas as pd
import pickle as pl


import seaborn as sns
import matplotlib.pyplot as plt


# from sklearn.cluster import DBSCAN
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler




# functions and definitions






# create dict with {value_set: FldKey_timeseries}


class FldKey_timeseries:

    def __init__(self, df_FldKey: pd.DataFrame, FldKey: str, value_set, λ_ɑ2:float=1.96) -> None:

        self.value_set = value_set
        
        self.df_FldKey = df_FldKey
        self.FldKey = FldKey
        self.λ_ɑ2 = λ_ɑ2

        # data updated through loop
        self.sc_X = None
        self.X = self.y = self.X_scaled = self.y_scaled = None
        self.__scale_data(value_set=self.value_set)

        # svr fitted line
        self.y_pred_rbf = self.resid_rbf = None
        self.__svr_rbf()

        # global conf
        self.S = self.a1 = self.a2 = None
        self.pred_glo = np.array([])
        for _ in range(2):
            self.__global_conf()     # NOTE global confidence interval for rbf residuals
        
        # local conf
        self.X_inl = self.y_inl = self.X_out = self.y_out = None
        self.pred_loc = self.s_arr = np.array([])
        self.__local_conf()

        # forecast
        self.µ_fc = self.S_fc = self.X_fc_scaled = None
        self.__gpr_forecast()

        # plotting
        self.__add_plot()


    def __scale_data(self, value_set: str) -> None:
        # Scaling data
        self.sc_X = StandardScaler(with_mean=True, with_std=True)
        sc_y = StandardScaler(with_mean=True, with_std=True)

        df_value_set = self.df_FldKey.query('FldVal==@value_set')

        self.X = df_value_set.Ordinal.values
        self.y = df_value_set.RowCnt.values

        self.X_scaled = self.sc_X.fit_transform(self.X.reshape(-1, 1))
        self.y_scaled = sc_y.fit_transform(self.y.reshape(-1, 1))


    def __svr_rbf(self) -> None:
        svr_rbf = SVR(kernel='rbf', C=100, gamma=4, epsilon=0.1)
        svr_rbf.fit(self.X_scaled, self.y_scaled.ravel())
        self.y_pred_rbf = svr_rbf.predict(self.X_scaled)

        # get residuals for SVR_RBF fitted curve and observed data points 
        self.resid_rbf = self.y_scaled.ravel()-self.y_pred_rbf


    def __global_conf(self):
        ''''''
        res = self.resid_rbf

        if self.pred_glo.any():
            res = res[self.pred_glo==1]       # filter out outliers
        print(len(res))
        
        self.S = np.std(res)

        µ = np.mean(res)
        self.a1, self.a2 = µ-self.λ_ɑ2*self.S, µ+self.λ_ɑ2*self.S     # upper lower bounds

        self.pred_glo = np.array([1 if self.a1 <= x <= self.a2 else -1 for x in self.resid_rbf])


    def __local_conf(self) -> None:
        pt_range = 5                                    # range stretching back- and forward, 2n+1 points in total
        # X_inl = X_out = y_inl = y_out = s_arr = []

        for i, x in enumerate(self.X_scaled):
            y = self.y_scaled[i]
            y_rbf = self.y_pred_rbf[i]

            low = i-pt_range if i-pt_range>=0 else 0
            upp = i+pt_range+1 if i+pt_range+1<=len(self.resid_rbf) else len(self.resid_rbf)+1
            pts = self.resid_rbf[low: upp]
            pts = pts[(self.a1 <= pts) & (pts <= self.a2)]        # filter out points outside of global confidence interval when calc local confidence interval

            # local confidence interval for rbf residuals
            s = np.std(pts)
            self.s_arr = np.append(self.s_arr, s)

            # plot residuals for points deemed as outliers
            pred = 1 if y_rbf-self.λ_ɑ2*s <= y <= y_rbf+self.λ_ɑ2*s else -1
            self.pred_loc = np.append(self.pred_loc, pred)

        self.X_inl, self.X_out = self.X_scaled[self.pred_loc==1], self.X_scaled[self.pred_loc==-1]
        self.y_inl, self.y_out = self.y_scaled[self.pred_loc==1], self.y_scaled[self.pred_loc==-1]


    def __gpr_forecast(self) -> None:
        ## Forecast of amt sales series (abbr. fc)
        X_fc = self.X
        forecast_len = 10
        for _ in range(forecast_len):
            X_fc = np.r_[X_fc, X_fc[-1]+1]
        self.X_fc_scaled = self.sc_X.transform(X_fc.reshape(-1, 1))
        
        self.X_inl = self.X_inl.reshape(-1, 1)

        variance = self.S**2
        kernel = RBF(length_scale=20, length_scale_bounds=(1e-3, 1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=variance, n_restarts_optimizer=8)
        gpr.fit(self.X_inl, self.y_inl)
        self.µ_fc, self.S_fc = gpr.predict(self.X_fc_scaled, return_std=True)


    def __add_plot(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            figsize=(10, 7), nrows=2, ncols=1, sharex=True
        )
        self.fig.tight_layout(pad=5)

        self.ax1.scatter(self.X_scaled, self.y_scaled, c=self.pred_glo)
        self.ax1.plot(self.X_scaled, self.y_pred_rbf, label='radial basis function', ls=':', c='r')      # estimated trend of data points
        self.ax1.set_title(f'RowCnt timeseries for ${FldKey}$ ${self.value_set}$')
        self.ax1.set_xlabel('$Ordinal$')
        self.ax1.set_ylabel('$RowCnt$')

        self.ax2.scatter(self.X_scaled, self.y_scaled, c='k', s=20, label='data points')
        self.ax2.plot(self.X_scaled, self.y_pred_rbf, c='r', ls=':', label='rbf pred')
        self.ax2.set_title(f'RowCnt timeseries for ${FldKey}$ ${self.value_set}$')
        self.ax2.set_xlabel('$Ordinal$')
        self.ax2.set_ylabel('$RowCnt$')
    

    def plot_global_conf(self) -> None:
        ##  AX1  ##
        # Global confidence interval
        self.ax1.plot(self.X_scaled, self.y_pred_rbf-λ_ɑ2*self.S, c='k', ls='--')                          # upper limit of 95% confidence interval
        self.ax1.plot(self.X_scaled, self.y_pred_rbf+λ_ɑ2*self.S, c='k', ls='--')                          # lower limit of 95% confidence interval
        for i, x in enumerate(self.X_scaled):
            if self.pred_glo[i] == -1:
                self.ax1.plot([x, x], [self.y_scaled.ravel()[i], self.y_pred_rbf[i]], c='r', lw=0.8, ls='--', zorder=-1)

        self.ax1.set_title(f'RowCnt timeseries for ${self.FldKey}$ ${self.value_set}$ w. global conf')
        self.ax1.legend()


    def plot_local_conf(self) -> None:
        ##  AX2  ##
        # plotting local confidence interval (for each point)
        for i, x in enumerate(self.X_scaled):
            y = self.y_scaled[i]
            y_rbf = self.y_pred_rbf[i]
            s = self.s_arr[i]
            self.ax2.plot([x, x], [y_rbf-self.λ_ɑ2*s, y_rbf+self.λ_ɑ2*s], c='y', ls=':')
            if x in self.X_out:
                self.ax2.plot([x, x], [y, y_rbf], c='r', ls='--', lw=0.8)                # plot residuals for points deemed as outliers

        self.ax2.set_title(f'RowCnt timeseries for ${self.FldKey}$ ${self.value_set}$ w. local conf')


    def plot_gpr_forecast(self) -> None:
        # GPR forecast line with mean and conf int as upper lower bounds of fill
        self.ax2.plot(self.X_fc_scaled, self.µ_fc, c='m', ls='--', label='gpr pred w/o noise')
        self.ax2.fill_between(
            x=self.X_fc_scaled.ravel(), 
            y1=self.µ_fc-λ_ɑ2*self.S_fc, 
            y2=self.µ_fc+λ_ɑ2*self.S_fc,
            color='black', alpha=0.1, ls='-.', label='gpr conf int'
            )
        
    







##############################################################################################
##############################################################################################
##############################################################################################




# data retreival
srcfile_arrbr = 'SAR'
with open(f'PICKLED/df_{srcfile_arrbr}_FldKey.pl', 'rb') as f:
    package = pl.load(f)

df_FldKey: pd.DataFrame = package[0]
FldKey: str = package[1]




λ_ɑ2 = 2.58
for value_set in df_FldKey.FldVal.unique():
    print(value_set)
    series = FldKey_timeseries(df_FldKey=df_FldKey, FldKey=FldKey, value_set=value_set, λ_ɑ2=λ_ɑ2)
    series.plot_global_conf()
    series.plot_local_conf()
    series.plot_gpr_forecast()
    plt.show()
    # plt.cla()




# Types of predictions
# - forecasted prediction with gpr
# - historic prediction with scr (global and local conf)
# 
# new methods:
# - get latest ordinal (used to ender new datapoint ordinal)
# - train model on accumulaed data
# - warning method - triggered when too many (eg 10) data points have been entered without refitting model
# - separate vactors (data trained on) vs (accumulated data inl. non fitted points)
# - predict methods according to above (fc, glo, loc) - all can be run simulataneously
# 





