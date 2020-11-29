import umap
import hdbscan

import numpy as np


def get_umap(node_embs):
    reducer = umap.UMAP()
    return reducer.fit_transform(node_embs)

def compute_clusterings(umap_embs):
    cl_sizes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    collected = []
    
    for cl_sz in cl_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_sz, min_samples=cl_sz)
        clusterer.fit(umap_embs)
        valid = [i for i, x in enumerate(clusterer.labels_) if x != -1]
        collected.append((clusterer.labels_, valid))
        
    return collected