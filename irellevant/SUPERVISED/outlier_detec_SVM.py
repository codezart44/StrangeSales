
# IMPORTS
import pandas as pd
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV



with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)

# SVC hyperparams
gamma = 1.0
C = 0.1
kernel = 'rbf'
rs=1


# Data setup
#__________________________________________________________________________________________________
X = df_amount[['FldVal', 'RowCnt']].values
X1_outl = np.random.uniform(low=min(X[:, 0]), high=max(X[:, 0]), size=100)
X2_outl = np.random.uniform(low=min(X[:, 1]), high=max(X[:, 1]), size=100)
X_outliers = np.c_[X1_outl, X2_outl]



X_train, X_test = train_test_split(X, test_size=0.3, random_state=rs)
X_train_outl, X_test_outl = train_test_split(X_outliers, test_size=0.3, random_state=rs)


y_train = np.r_[np.ones(len(X_train)), np.ones(len(X_train_outl))*-1]
X_train = np.r_[X_train, X_train_outl]

y_test = np.r_[np.ones(len(X_test)), np.ones(len(X_test_outl))*-1]
X_test = np.r_[X_test, X_test_outl]


sc = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)





# Supervised SVC model
#__________________________________________________________________________________________________
param_grid = {
    'gamma': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 100],
    'C': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 100]
}
gscv = GridSearchCV(
    estimator=SVC(kernel=kernel),
    param_grid=param_grid,
    scoring='accuracy',
    refit='accuracy',
    cv=5,
    verbose=2
)

gscv.fit(X_train_scaled, y_train)
svm: SVC = gscv.best_estimator_
print(gscv.best_params_)
y_pred = svm.predict(X_test_scaled)


df_res = pd.DataFrame(data=zip(X_test[:, 0], X_test[:, 1], y_pred), columns=['amount', 'rowcnt', 'pred'])


df_anom = df_res.query('pred==-1')
df_true = df_res.query('pred==1')

xx, yy = np.meshgrid(
    np.linspace(min(df_amount.FldVal)*1.05, max(df_amount.FldVal)*1.05, 50),
    np.linspace(min(df_amount.RowCnt)*0.95, max(df_amount.RowCnt)*1.05, 50),
)
Z = svm.predict(sc.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone)

inl = plt.scatter(df_anom.amount, df_anom.rowcnt, s=20, c='red', edgecolors='k')
out = plt.scatter(df_true.amount, df_true.rowcnt, s=20, c='black', edgecolors='k')
plt.show()