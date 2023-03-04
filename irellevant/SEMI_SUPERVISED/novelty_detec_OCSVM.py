
# IMPORTS
import pandas as pd
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split



with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)

# OCSVM hyperparams
nu = 0.01               # nu (greek letter ν) Upper bound for fraction of outliers we want to allow
gamma = 'scale'
rs = 1                  # random state



X = df_amount[['FldVal', 'RowCnt']].values
X1_outl = np.random.uniform(low=min(X[:, 0]), high=max(X[:, 0]), size=50)
X2_outl = np.random.uniform(low=min(X[:, 1]), high=max(X[:, 1]), size=50)
X_outliers = np.c_[X1_outl, X2_outl]

X_train, X_test = train_test_split(X, test_size=0.3, random_state=rs)
X_test = np.r_[X_test, X_outliers]




sc = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Normal OCSVM
#__________________________________________________________________________________________________
clf = OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
clf.fit(X_train_scaled)
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

df_res_train = pd.DataFrame(data=zip(X_train[:,0], X_train[:, 1], y_pred_train), columns=['amount', 'rowcnt', 'pred'])
df_res_test = pd.DataFrame(data=zip(X_test[:,0], X_test[:, 1], y_pred_test), columns=['amount', 'rowcnt', 'pred'])


df_anom = df_res_test.query('pred==-1')
df_true = df_res_test.query('pred==1')

xx, yy = np.meshgrid(
    np.linspace(min(X_test[:, 0])*1.05, max(X_test[:, 0])*1.05, 50),
    np.linspace(min(X_test[:, 1]), max(X_test[:, 1])*1.05, 50)
    )
Z = clf.decision_function(sc.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone)



inl = plt.scatter(df_anom.amount, df_anom.rowcnt, s=20, c='red', edgecolors='k')
out = plt.scatter(df_true.amount, df_true.rowcnt, s=20, c='black', edgecolors='k')
plt.show()








## SÄTT UPP EN ENKEL MODELL FÖR BRA / DÅLIGA VÄRDEN PÅ ENBART AMOUNT - NOVELTY DETECTION
## SVM ==> lyft till högre dimensioner, testa endast på AMOUNT







# OCSVM w. SGD
#__________________________________________________________________________________________________
# transform = Nystroem(gamma=gamma, random_state=rs)
# clf_sgd = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, random_state=rs, tol=1e-4)
# clf_sgd.fit(X_train)
# y_pred_test_sgd = clf_sgd.predict(X_test)

# df_res_sgd = pd.DataFrame(data=zip(X_test[:, 0], X_test[:, 1], y_pred_test_sgd), columns=['amount', 'rowcnt', 'pred'])
# print(df_res_sgd)



# df_anom_sgd = df_res_sgd.query('pred==-1')
# df_true_sgd = df_res_sgd.query('pred==1')

# print(df_anom_sgd)
# print(df_true_sgd)


# xx, yy = np.meshgrid(
#     np.linspace(min(X_test[:, 0])*1.05, max(X_test[:, 0])*1.05, 50),
#     np.linspace(min(X_test[:, 1]), max(X_test[:, 1])*1.05, 50)
#     )
# Z = clf_sgd.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)


# plt.contourf(xx, yy, Z, cmap=plt.cm.bone)
# plt.scatter(df_anom_sgd.amount, df_anom_sgd.rowcnt, s=20, c='red', edgecolors='k')
# plt.scatter(df_true_sgd.amount, df_true_sgd.rowcnt, s=20, c='black', edgecolors='k')
# plt.show()