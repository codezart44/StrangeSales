

# Excessive purchases, seen through FldKey 'Amount'. Usually values between -10,000 and 10,000
# Single batch model
# model: Isof, classify deviating amounts as outliers

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle as pl


inputfiles = os.listdir('CSVfiles2')

# print(len(trainfiles), len(testfile))
# print(inputfiles)


# NOTE ATT GÅ IGENOM:
# 1 senaste verision av amount outlier detection
#   - joint eval. Bedömer amount och rowCnt sammansatt och viktat. Antingen predictions el. score
#   - viktad bedömning bygger på att det genom 'score' finns svaga outlier och starka 'outliers'
#   - feature med högre inlier score kommer överväga feature med lågt outlier score (som frekvenser som stör ut alt. förstärker)
# 2 Andra versioner (avkommentera)
# 3 Pickle - hur mar sparar modeller (alla typer av objekt)


# model: KMeans-clusters
# Classify outliers by amount * RowCnt, plotting normal distr of X, Xo, Xi.
#__________________________________________________________________________________________________________________________________________
# FldVals = amountFldValRowCnt.FldVal.values
# RowCnts = amountFldValRowCnt.RowCnt.values

# X = np.c_[RowCnts, FldVals]


# clf = KMeans(n_clusters=2, random_state=0)
# clf.fit(X)
# clf.labels_
# y_pred_train = clf.predict(X)

# anom_score = clf.predict(X)



# models: Isof
# Joint evaluation by separate classification. Joint outlier prediction | joint anomaly score
#__________________________________________________________________________________________________________________________________________
with open('PICKLED/NBdf.pl', 'rb') as f:
    dfSales = pl.load(f).reset_index(drop=True)

dfSales.FldVal = list(map(lambda x: float(x.replace(',', '.')), dfSales.FldVal))

scaler = StandardScaler()
scaler.fit(dfSales)
scaled_Data = scaler.transform(dfSales)

X_train, X_test, df_train, df_test = train_test_split(scaled_Data, dfSales, test_size=0.2, random_state=1)

clf = IsolationForest(n_estimators=30, max_features=0.1, contamination=0.01, bootstrap=True, random_state=1)
clf.fit(X_train)



y_pred = clf.predict(X_test)
print(y_pred)

# Joint outlier prediction
# Small amount many times is expected
# Large amount many times is however not expected   -  outlier if (FldVal_pred==-1,  RowCnt_pred==-1)
# Small amount few times is expected
# Large amount few times is expected

result_df = pd.DataFrame(zip(df_test.FldVal.values, df_test.RowCnt.values, y_pred), columns=['FldVal', 'RowCnt', 'Outlier'])
print(result_df)

    
xi = result_df.query('Outlier==1')['FldVal'].values
xo = result_df.query('Outlier==-1')['FldVal'].values
yi = result_df.query('Outlier==1')['RowCnt'].values
yo = result_df.query('Outlier==-1')['RowCnt'].values


marginal = 500
xx, yy = np.meshgrid(
    np.linspace(min(dfSales.FldVal.values)-marginal, max(dfSales.FldVal.values)+marginal, 50), 
    np.linspace(min(dfSales.RowCnt.values)-300, max(dfSales.RowCnt.values)+marginal, 50)
    )

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone)                   # cmap.plt.cm.bone for white/black color map


b = plt.scatter(xi, yi, c='green', s=20, edgecolor="k")       # plt inlier points
c = plt.scatter(xo, yo, c='red', s=20, edgecolor="k")         # plt outlier points
# plt.scatter(X[:, 0], X[:, 1], c='b', s=30, edgecolors='k')                            # plt all points
plt.legend(
    [b, c],
    ['inliers', 'outliers'],
    loc='upper left'
)
plt.title('Number of sales to Sales amount')
plt.xlabel('amount')
plt.ylabel('No. sales')
plt.show()





# model: Isof
# Classify ouliers by RowCnt, inverse plotted (many data points with RowCnt=1 but these repr. the true outliers)
#__________________________________________________________________________________________________________________________________________
# # DATA PREP
# FldVals = amountFldValRowCnt.FldVal.values
# RowCnts = amountFldValRowCnt.RowCnt.values
# # sc = preprocessing.StandardScaler()
# # weights = sc.fit_transform(FldVals.reshape(-1 ,1))

# X_train = RowCnts.reshape(-1, 1)
# X_test = X_train

# # Weight input data, outliers are heavily favoured
# amountMean = np.mean(np.abs(FldVals))
# weights = np.log(abs(FldVals))

# # MODEL FITTING
# clf = IsolationForest(n_estimators=50, max_features=1, random_state=1)
# clf.fit(X_train, sample_weight=None)
# anom_score = clf.decision_function(X_test)
# y_pred_test = clf.predict(X_test)

# result_df = pd.DataFrame(zip(X_train.ravel(), FldVals, anom_score, y_pred_test), columns=['RowCnt', 'FldVal', 'score', 'prediction'])
# print(result_df)

# inliers = result_df.query('prediction==1')
# outliers = result_df.query('prediction==-1')
# X_inliers = np.c_[inliers.RowCnt.values, inliers.FldVal.values]
# X_outliers = np.c_[outliers.RowCnt.values, outliers.FldVal.values]

# # PLOTTING
# xx, yy = np.meshgrid(np.linspace(-500, 17500, 50), np.linspace(-3500, 1500, 50))
# Z = clf.decision_function(xx.reshape(-1, 1))
# Z = Z.reshape(xx.shape)                                     # Z height values only scaled along x_axis (contourf)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Greens_r)               # cmap.plt.cm.bone for white/black color map

# b = plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='r', s=20, edgecolor="k")      # plt inlier points
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='g', s=20, edgecolor="k")       # plt outlier points
# # # plt.scatter(X[:, 0], X[:, 1], c='b', s=30, edgecolors='k')                            # plt all points
# plt.legend(
#     [c, b],
#     ['inliers', 'outliers'],
#     loc='lower right'
# )
# plt.title('Sales amount to Number of Sales')
# plt.xlabel('No. sales')
# plt.ylabel('amount')

# # sns.kdeplot(X_inliers[:, 0], X_inliers[:, 1], shade=True, color='r')
# # sns.kdeplot(X_outliers[:, 0], X_outliers[:, 1], shade=True, color='g')

# plt.show()





# model: Isof
# Classify ouliers by FldVal, NOTE weighted by log(RowCount)
#__________________________________________________________________________________________________________________________________________
# FldVals = amountFldValRowCnt.FldVal.values
# RowCnts = amountFldValRowCnt.RowCnt.values
# sc = preprocessing.StandardScaler()


# X_train = np.array(FldVals).reshape(-1, 1)
# # RowCnts = sc.fit_transform(RowCnts.reshape(-1, 1))
# X_test = X_train

# # Weight input data, outliers are heavily favoured
# # weights = list(map(lambda x: -1/np.log(x) if x>1 else x, RowCnts))
# weights = np.log(RowCnts)                                               # NOTE Very exciting! Turns RowCnts of value 1 to log(1)==0, negatively favours odd occurences
# # weights = list(map(lambda x: (1/x+1)**3 if np.log(x) < 5 else 0, RowCnts))


# clf = IsolationForest(n_estimators=50, max_features=1, random_state=1)
# clf.fit(X_train, sample_weight=weights)
# anom_score = clf.decision_function(X_test)
# y_pred_test = clf.predict(X_test)


# result_df = pd.DataFrame(zip(X_train.ravel(), RowCnts, weights, anom_score, y_pred_test), columns=['FldVal', 'RowCnt', 'weights', 'score', 'prediction'])
# print(result_df)


# inliers = result_df.query('prediction==1')
# outliers = result_df.query('prediction==-1')
# X_inliers = np.c_[inliers.FldVal.values, inliers.RowCnt.values]
# X_outliers = np.c_[outliers.FldVal.values, outliers.RowCnt.values]


# xx, yy = np.meshgrid(np.linspace(-3500, 1500, 50), np.linspace(-500, 17500, 50))
# Z = clf.decision_function(xx.reshape(-1, 1))
# Z = Z.reshape(xx.shape)                                     # Z height values only scaled along x_axis (contourf)
# plt.contourf(xx, yy, Z, cmap=plt.cm.bone)               # cmap.plt.cm.bone for white/black color map


# b = plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='black', s=20, edgecolor="k")      # plt inlier points
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='r', s=20, edgecolor="k")       # plt outlier points
# # plt.scatter(X[:, 0], X[:, 1], c='b', s=30, edgecolors='k')                            # plt all points
# plt.legend(
#     [b, c],
#     ['inliers', 'outliers'],
#     loc='upper left'
# )
# plt.title('NUmber of sales to Sales amount')
# plt.xlabel('amount')
# plt.ylabel('No. sales')
# plt.show()







# model: Isof
# Classify outliers by amount vector with RowCnt occurences respectively. plotting normal distr of X, Xo, Xi.
#__________________________________________________________________________________________________________________________________________
# batch_df = pd.read_csv(f'CSVfiles/{testBatches[1]}.csv')
# X_test = (batch_df
# .query("FldKey=='Amount'")
# [['FldVal', 'RowCnt']]
# )
# X_test.FldVal = np.array(list(map(lambda amount: float(amount.replace(',', '.')), X_test.FldVal)))
# X_train = X_test


# amountFldValSeries = []
# for i, count in enumerate(X_train.RowCnt):
#     amountFldValSeries += [X_train.FldVal.values[i]] * count
# amountFldValSeries = np.array(amountFldValSeries)
# print(len(amountFldValSeries))
# X_train = amountFldValSeries.reshape(-1, 1)

# X2_train = X_test.FldVal.values.reshape(-1, 1)

# # X_train, X_test = model_selection.train_test_split(amountFldValSeries.reshape(-1, 1), test_size=0.2, random_state=7)

# clf = IsolationForest(n_estimators=200, max_features=1, random_state=1)
# clf2 = IsolationForest(n_estimators=200, max_features=1, random_state=1)


# clf.fit(X_train)
# clf2.fit(X2_train)

# anom_score = clf.decision_function(X_train)
# anom_score2 = clf2.decision_function(X2_train)
# y_pred_test = clf.predict(X_train)
# y_pred_test2 = clf2.predict(X2_train)


# result_df = pd.DataFrame(zip(X_train.ravel(), anom_score, y_pred_test), columns=['FldVal', 'score', 'prediction'])
# result_df = result_df.drop_duplicates()
# result_df['RowCnt'] = X_test['RowCnt'].values
# result_df['prediction'] = (result_df.prediction.values + y_pred_test2)/2
# result_df = result_df.sort_values('FldVal')
# print(result_df)


# inliers = result_df.query('prediction==0 | prediction==1')
# outliers = result_df.query('prediction==-1')




# X_inliers = np.c_[inliers.FldVal.values, inliers.RowCnt.values]
# X_outliers = np.c_[outliers.FldVal.values, outliers.RowCnt.values]


# xx, yy = np.meshgrid(
#     np.linspace(
#         -3500,
#         2000,
#         50
#         ), 
#     np.linspace(
#         -300,
#         18000,
#         50
#         ))
# Z1 = clf.decision_function(xx.reshape(-1, 1))
# Z2 = clf2.decision_function(yy.reshape(-1, 1))
# Z = (Z1+Z2).reshape(-1, 50)                                       # Z height values only scaled along x_axis (contourf)
# plt.contourf(xx, yy, Z, cmap=plt.cm.bone)                   # cmap.plt.cm.bone for white/black color map


# b = plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='black', s=20, edgecolor="k")       # plt inlier points
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='r', s=20, edgecolor="k")         # plt outlier points
# # plt.scatter(X[:, 0], X[:, 1], c='b', s=30, edgecolors='k')                            # plt all points
# plt.legend(
#     [b, c],
#     ['inliers', 'outliers'],
#     loc='upper left'
# )
# plt.title('Number of sales to Sales amount')
# plt.xlabel('amount')
# plt.ylabel('No. sales')
# plt.show()




# NOTES
#__________________________________________________________________________________________________________________________________________
# Vad hände om man lägger till ett udda värde som är innanför intervallet av inliers men är unikt? 
#
# NOTE Hur reagerar modellen på data som är distribuerad på olika sätt?
#   - amountFldValSeries verkar ha en relativt jämn normalfördelning med en mindre fördelning till vänster
#   - Amount har få värden per batch. Skulle behöva samla amount från flera batches för att ha mkt data att träna programmet med. 
#   - Kan simulera data med normalfördelningar. Stor size ger högre upplösning i distributionsgrafen. Behövs speciellt med X1 och X2 tillsammans.
#   - Men endast en normaldistribuerad vektor är man lätt se fördelningen av modellens inliers/outliers även för mindre size.
#   - När en sammansatt normalditribuerad vektor instroduceras har programmet svårare att klassificera in-/outliers. 






# max_outlier = np.max(result_df.query('outliers==-1')['data'])[0]
# min_outlier = np.min(result_df.query('outliers==-1')['data'])[0]
# max_inlier = np.max(result_df.query('outliers==1')['data'])[0]
# min_inlier = np.min(result_df.query('outliers==1')['data'])[0]
# print(f'''
#     outliers in range: [{min_outlier}, {min_inlier}) U ({max_inlier}, {max_outlier}]
#     inliers in range:  [{min_inlier}, {max_inlier}]
# ''')


# Distribution of X data
# sns.kdeplot(data=result_df.query('prediction==-1')['data'], color='r')
# sns.kdeplot(data=result_df.query('prediction==1')['data'], color='g')
# sns.kdeplot(data=X_train, color='b')