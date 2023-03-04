

# imports
import numpy as np
import pandas as pd
import pickle as pl

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



# formatting
with open('PICKLED/df_amount.pl', 'rb') as f:
    df_amount: pd.DataFrame = pl.load(f)

ordinal = []
for i, batchId in enumerate(df_amount['BatchInstId'].unique().ravel()):        # batches are ordered according to Input file (chronological)
    ordinal += [i]*df_amount.query('BatchInstId==@batchId').shape[0]
df_amount['Ordinal'] = ordinal

X = df_amount[['FldVal', 'RowCnt', 'Ordinal']]

sc = StandardScaler(with_mean=True, with_std=True)
X_scaled = sc.fit_transform(X)

df_scaled_metrics = pd.DataFrame(data=X_scaled, columns=X.columns)
# print(df_scaled_metrics.describe())




# modeling
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(X_scaled)


def plot_elbow_method(X) -> None:
    plt.figure(figsize=(10,5))
    nn = NearestNeighbors(n_neighbors=2)        # The two nearest neightbors, the first being the point itselt, the other the closest point other than itself
    nn.fit(X_scaled)
    distances, idx = nn.kneighbors(X_scaled)    # Returns distances from each point to itself (thus 0) and a series of the distancs to the closest point (thus we choose column 1)
    distances = np.sort(distances, axis=0)
    print(distances)
    plt.plot(distances[:,1])
    plt.show()            # The plot shows that epsilon should be choosen to about 0.07

def iterative_silhouette(X, upper_eps=0.1, upper_samples=5) -> pd.DataFrame:
    eps_range = np.arange(0.01, upper_eps, 0.01)
    min_samples_range = np.arange(2, upper_samples)

    eps_values = []
    min_sample = []
    n_clusters = []
    sil_scores = []

    # SILHOUETTE SCORE ANALYSIS
    for eps in eps_range:
        for min_samples in min_samples_range:

            eps, min_samples = np.round(eps, 2), np.round(min_samples)          # Round of errors otherwise
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X)

            eps_values.append(eps)
            min_sample.append(min_samples)
            n_clusters.append(len(np.unique(dbscan.labels_)))
            sil_scores.append(silhouette_score(X=X, labels=dbscan.labels_))

    sil_cluster_data = zip(eps_values, min_sample, n_clusters, sil_scores)
    df_sil_cluster = pd.DataFrame(
        data=sil_cluster_data, 
        columns=['epsilon_values', 'min_sampl_core', 'n_clusters', 'silhouette_score']
        )
    return df_sil_cluster

def plot_scatter3D_clustering(labels) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs=df_amount.Ordinal, ys=df_amount.FldVal, zs=df_amount.RowCnt, c=labels)
    plt.show()

def plot_3perspective_clustering(labels) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0))
    ax2 = plt.subplot2grid(shape=(1, 3), loc=(0, 1))
    ax3 = plt.subplot2grid(shape=(1, 3), loc=(0, 2))

    ax1.scatter(x=df_amount.Ordinal, y=df_amount.FldVal, c=labels, s=20)
    ax1.set_xlabel('batch ordinal'), ax1.set_ylabel('sales amount'), ax1.set_title('Amount over time')
    ax1.grid(linewidth=0.2), ax1.set_axisbelow(True)

    ax2.scatter(x=df_amount.Ordinal, y=df_amount.RowCnt, c=labels, s=20)
    ax2.set_xlabel('batch ordinal'), ax2.set_ylabel('sales frequency'), ax2.set_title('Frequency over time')
    ax2.grid(linewidth=0.2), ax2.set_axisbelow(True)

    ax3.scatter(x=df_amount.FldVal, y=df_amount.RowCnt, c=labels, s=20)
    ax3.set_xlabel('sales amount'), ax2.set_ylabel('sales frequency'), ax3.set_title('Frequency to Amount')
    ax3.grid(linewidth=0.2), ax3.set_axisbelow(True)
    plt.show()


# plot_elbow_method(X=X_scaled)
# df_sil = iterative_silhouette(data=X_scaled ,upper_eps=0.12, upper_samples=6)
# print(df_sil)

# plot_scatter3D_clustering(labels=dbscan.labels_)

# plot_3perspective_clustering(labels=dbscan.labels_)



# df_amount = df_amount.drop(columns=['BatchInstId'])
# df_grouped = df_amount.groupby('FldVal').agg({'RowCnt':lambda x: list(x), 'Ordinal':lambda x: list(x)})
# print(df_grouped)




n_batches = df_amount.BatchInstId.nunique()
df_prev = df_amount.groupby('FldVal').agg({'RowCnt': lambda x: len(list(x))/n_batches}).rename(columns={'RowCnt':'Prevalence'})
transcription_dict = dict(zip(df_prev.index, df_prev.Prevalence))
df_amount['Prevalence'] = df_amount['FldVal'].map(transcription_dict)


# for amt in df_prev.index:
#     df = df_amount.query('FldVal==@amt')
#     fig, ax = plt.subplots(figsize=(10, 7))
#     ax.scatter(x=df.Ordinal, y=df.RowCnt)
#     ax.set_xlim(-5, 95)
# plt.show()






dbscan = DBSCAN(eps=0.1, min_samples=5)
for i in range(2):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    if i==0:
        X2 = df_amount[['RowCnt', 'Ordinal', 'Prevalence']].values
        X2_scaled = sc.fit_transform(X2)
        dbscan.fit(X2_scaled)
        df_amount['labels'] = dbscan.labels_
        df_noise_free = df_amount.query('labels!=-1')
        ax.scatter3D(xs=df_noise_free.Ordinal, ys=df_noise_free.Prevalence, zs=df_noise_free.RowCnt, c=df_noise_free.labels)
        # ax.scatter3D(xs=df_amount.Ordinal, ys=df_amount.Prevalence, zs=df_amount.RowCnt, c=dbscan.labels_)
    if i==1:
        X2 = df_amount[['FldVal', 'Ordinal', 'Prevalence']].values
        X2_scaled = sc.fit_transform(X2)
        dbscan.fit(X2_scaled)
        df_amount['labels'] = dbscan.labels_
        df_noise_free = df_amount.query('labels!=-1')
        ax.scatter3D(xs=df_noise_free.Ordinal, ys=df_noise_free.FldVal, zs=df_noise_free.Prevalence, c=df_noise_free.labels)
        # ax.scatter3D(xs=df_amount.Ordinal, ys=df_amount.FldVal, zs=df_amount.Prevalence, c=dbscan.labels_)

    plt.show()




