import pandas as pd
from dbcv import dbcv
import numpy as np

record_df = pd.read_csv("./results-umap-hdbscan/record.csv")

dbcv_scores = []
n_clusters = []
taux_noise = []
for i in range(len(record_df)):
    index_to_load = record_df.index[i]
    df = pd.read_csv(f"../../results-umap-hdbscan/results-{index_to_load}.csv")
    X = np.array([df["x"].to_list(), df["y"].to_list()]).T
    y = df["hdbscan_clusters"].to_numpy()
    dbcv_scores += [dbcv(X, y)]
    n_clusters += [
        len([c_index for c_index in df["hdbscan_clusters"].unique() if c_index != -1])
    ]
    taux_noise += [(df["hdbscan_clusters"] == -1).mean()]

record_df.loc[:, "dbcv_scores"] = dbcv_scores
record_df.loc[:, "n_clusters"] = n_clusters
record_df.loc[:, "taux_noise"] = taux_noise

record_df.to_csv("./results-umap-hdbscan/record-with-scores.csv", index = False)
