import json

from datasets import load_from_disk
from hdbscan import HDBSCAN
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from umap import UMAP


def log(text):
    time = pd.Timestamp.now()
    with open("./src/logs.txt", "a") as file:
        file.write(f"[LOG] {time.strftime('%Y-%m-%d %X')} -- {text}\n")


UMAP_n_neighbors_list = [15, 30, 45, 60]
HDBSCAN_min_cluster_size_list = [5, 10, 20]
HDBSCAN_min_samples_list = [5, 10, 20]
models = {
    "fr": ["alibaba", "camembertav2", "qwen06b"],
    "en": ["alibaba", "f2llm", "qwen06b"],
}

index = 0
record = []
for language_short in ["fr", "en"]:
    for model_used in models[language_short]:
        log("#" * 50)
        log(f"{language_short} - {model_used}")
        log("#" * 50)

        ds = load_from_disk(f"./embeddings/embeddings-{language_short}-{model_used}")
        embeddings = np.array(ds["embedding"])
        log("Dataset loaded")

        for n_neighbors, min_cluster_size, min_sample in tqdm(
            product(
                UMAP_n_neighbors_list,
                HDBSCAN_min_cluster_size_list,
                HDBSCAN_min_samples_list,
            )
        ):
            record.append(
                {
                    "index": index,
                    "model": model_used,
                    "n_neighbors": n_neighbors,
                    "min_cluster_size": min_cluster_size,
                    "min_sample": min_sample,
                }
            )
            log(
                f"neighbor: {n_neighbors} | cluster size: {min_cluster_size} | "
                f"min sample: {min_sample}" + "#" * 20
            )

            umap_model = UMAP(
                n_neighbors=n_neighbors,
                metric="cosine",
                n_components=2,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size, min_samples=min_sample
            )

            Xp = umap_model.fit_transform(embeddings)
            log("UMAP computed")
            y = hdbscan_model.fit_predict(Xp)
            log("HDBSCAN computed")

            df = pd.DataFrame(
                {
                    "x": Xp[:, 0],
                    "y": Xp[:, 1],
                    "oai_first_name": ds["oai_first_name"],
                    "sujets_rameau_fr": ds["sujets_rameau_fr"],
                    "hdbscan_clusters": y,
                }
            )

            df.to_csv(f"./umap_hdbscan_results/results-{index}.csv", index=False)
            index += 1

            if index == 3:
                break

pd.DataFrame("./results-umap-hdbscan/record.csv", index=False)
