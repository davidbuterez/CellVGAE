import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict

def mean_of_attention_heads(matrix, out_dim):
    chunks = torch.split(matrix, out_dim, dim=1)
    return torch.mean(torch.stack(chunks), dim=0)

def latent_dim_participation_in_clusters(latent_data, labels):
    latent_diff = np.zeros(shape=(latent_data.shape[1], len(set(labels)) + 1))
    
    for l_dim in range(latent_data.shape[1]):
        cells_in_dim = latent_data[:, l_dim]
        l_dim_mean = np.mean(cells_in_dim)
        l_dim_std = np.std(cells_in_dim)

        variable_cells_larger = np.where(cells_in_dim > l_dim_mean + l_dim_std)
        variable_cells_smaller = np.where(cells_in_dim < l_dim_mean - l_dim_std)
        
        labels_larger = labels[variable_cells_larger]
        labels_smaller = labels[variable_cells_smaller]
        
        variable_labels = np.concatenate((labels_larger, labels_smaller), axis=None)
        
        cluster_count = {x: list(variable_labels).count(x) for x in labels}
        counter_per_cluster = np.array(list(cluster_count.values())) / len(variable_labels)
        counter_per_cluster = np.around(counter_per_cluster * 100.0, decimals=2)
        
        latent_diff[l_dim][1:] = counter_per_cluster
        latent_diff[l_dim][0] = int(l_dim)
        
    cluster_label = [str(i) for i in np.unique(labels)]

    latent_diff = pd.DataFrame(latent_diff, columns=['Latent dimension'] + cluster_label)
    latent_diff['Latent dimension'] = latent_diff['Latent dimension'].astype(int)

    latent_diff = latent_diff.melt(id_vars=['Latent dimension'], value_vars=cluster_label, var_name='Cluster',
                                   value_name='Percentage')
    sns.set(font_scale=2.5)
    sns.set_style("whitegrid")
    g = sns.catplot(x='Cluster', y='Percentage', col='Latent dimension', data=latent_diff, palette=sns.color_palette("hls", len(set(labels))), col_wrap=5,
                       kind="bar", ci=None, aspect=1.3, legend_out=True)

    for ax in g.axes:
        ax.set_xticklabels(sorted(set(labels)))
        plt.setp(ax.get_xticklabels(), visible=True)
        
    
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    return latent_diff


def _indices_of_top_k(arr, k):
    return np.argpartition(arr, -k)[-k:]


def select_genes_by_latent_dim(matrix, latent_dim, top_k):
    corresponding_to_latent_dim = matrix[:, latent_dim]
    return _indices_of_top_k(corresponding_to_latent_dim.detach().numpy(), top_k)


def merged_count(list_of_tuples):
    counter = defaultdict(int)
    for lst in list_of_tuples:
        for tup in lst:
            counter[tup[0]] += tup[1]
    return counter