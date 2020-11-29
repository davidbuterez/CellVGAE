# CellVGAE

An unsupervised scRNA-seq analysis workflow with graph attention networks

![](figures/workflow.png)



CellVGAE uses the connectivity between cells (such as *k*-nearest neighbour graphs or KNN) with gene expression values as node features to learn high-quality cell representations in a lower-dimensional space, with applications in downstream analyses like (density-based) clustering, visualisation, gene set enrichment analysis and others. CellVGAE leverages both the variational graph autoencoder and graph attention networks to offer a powerful and more interpretable machine learning approach. It is implemented in PyTorch using the PyTorch Geometric library.

## Requirements

The following packages are required to be able to run everything in this repository (included are the versions we used):

```bash
pandas==1.0.5
torch_geometric==1.6.1
seaborn==0.10.1
matplotlib==3.2.2
numpy==1.18.5
hdbscan==0.8.26
torch==1.6.0
umap_learn==0.4.6
graph_tool==2.11
numpy==1.19.4
scikit_learn==0.23.2
umap==0.1.1
```
The used CUDA toolkit version is 10.2. We also used CellVGAE successfully with PyTorch 1.7.0 and PyTorch Geometric 1.7.0 (CUDA 11.0). 

[graph-tool](https://graph-tool.skewed.de/) is currently only available on Linux/Mac OS.

The version of R we used is 4.0.2. The following libraries are required:

`Seurat 3`, `scran`, `SingleCellExperiment`. `scRNAseq`, `BiocSingular`, `igraph`, `dplyr` and `textshape`.

## Usage

The `train.py` file can be invoked with the following options:

```
train [-h] [--hvg_file HVG_FILE] [--graph_file GRAPH_FILE] [--num_hidden_layers NUM_HIDDEN_LAYERS] [--num_heads NUM_HEADS] [--hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]] [--dropout [DROPOUT [DROPOUT ...]]] [--latent_dim LATENT_DIM] [--loss {kl,mmd}] [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--val_split VAL_SPLIT] [--node_out NODE_OUT] [--save_trained_model SAVE_TRAINED_MODEL]

Train CellVGAE

optional arguments:
  -h, --help            show this help message and exit
  --hvg_file HVG_FILE   
  						HVG file (log-normalised)
  --graph_file GRAPH_FILE
                        Graph specified as an edge list (one per line, separated by whitespace)
  --num_hidden_layers NUM_HIDDEN_LAYERS
                        Number of hidden layers (must be 2 or 3)
  --num_heads NUM_HEADS
                        Number of attention heads
  --hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]
                        Output dimension for each hidden layer (only 2 or 3 layers allowed)
  --dropout [DROPOUT [DROPOUT ...]]
                        Dropout for each hidden layer (only 2 or 3 layers allowed)
  --latent_dim LATENT_DIM
                        Latent dimension (output dimension for node embeddings)
  --loss {kl,mmd}       
  						Loss function (KL or MMD)
  --lr LR               
  						Learning rate for Adam
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       
  						Number of training epochs
  --val_split VAL_SPLIT
                        Validation split e.g. 0.1
  --node_out NODE_OUT   
  						Output file name and path for the computed node embeddings (saved in numpy .npy format)
  --save_trained_model SAVE_TRAINED_MODEL
                        Path to save PyTorch model
```



We recommend running CellVGAE with the default architectural parameters of 2 hidden layers of dimension 128 and 0.2 dropout each and the number of latent dimension set to 50; we used this architecture for all datasets (only changing the number of attention heads):

```bash
python train.py --hvg_file HVG/Loh_HVG_500.csv --graph_file KNN/Loh_HVG_500_KNN_k5_d50.txt --num_hidden_layers 2 --num_heads 10
--hidden_dims 128 128 --dropout 0.2 0.2 --latent_dim 50 --lr 0.0001 --batch_size 64
```

The default settings are as above, and additionally the default number of epochs is set to 250, the validation split to 0, the loss to KL divergence (more memory efficient in the current implementation, for most of the experiments we have used the MMD). By default, the node embeddings are saved in a file `node_embs.npy` in the current directory, and the model to `model.pt` in the current directory.

### Preprocessing steps

1. Highly variable genes (HVG) selection is performed using the included`preprocessing/hvg_proprocessing.ipynb` notebook, where a count matrix (after quality control) is loaded and processed. User inputs are the project name and number of HVGs (250 or 500).
2. Graph generation (either KNN or Pearson) is typically done on the HVGs matrix from the last step (other options are on the entire gene expression matrix, or on some HVG matrix of other dimension, e.g. using 1000 HVGs for graph generation but 250 HVGs for node features). User inputs are the number of neighbours *k* and optionally, for KNN, the number of dimensions of *PCA*.

## Repository structure

We provide the repository structure highlighting the directories and the Jupyter notebooks that accompany the data:

```
├──README.md
├──requirements.txt
├──train.py
├──tree.txt
├──figures
│  ├──attention_visualisation
│  │  ├──Darmanis
│  │  ├──PBMC3k
│  │  ├──attn_graph_darmanis.ipynb
│  │  ├──attn_graph_pbmc.ipynb
│  │  ├──attn_heatmap_darmanis.ipynb
│  │  └──attn_heatmap_pbmc.ipynb
|  |
│  ├──baron4
│  │  ├──CellVGAE
│  │  ├──DiffVAE
│  │  ├──SAM
│  │  └──clusters.ipynb
|  |
│  ├──macrophages
│  │  ├──CellVGAE
│  │  ├──DiffVAE
│  │  ├──SAM
│  │  └──clusters.ipynb
|  |
│  ├──muraro
│  │  ├──CellVGAE
│  │  ├──DiffVAE
│  │  ├──SAM
│  │  └──clusters.ipynb
|  |
│  ├──pbmc3k
│  │  ├──clustering
│  │  │  └──pbmc3k_clusters.ipynb
|  |  |
│  │  └──expression
│  │     └──pbmc3k_expression.ipynb
|  |
│  └──schisto
│     ├──CellVGAE
│     ├──DiffVAE
│     ├──SAM
│     ├──Seurat
│     └──schisto_clusters.ipynb
|
├──misc
│  ├──benchmark_sam.ipynb
│  ├──clean_seger.ipynb
│  ├──clean_wang.ipynb
│  └──curate_baron.ipynb
|
├──models
│  ├──CellVGAE_Encoder.py
│  ├──CellVGAE.py
│  └──mmd.py
|
├──preprocessing
│  ├──generate_knn.ipynb
│  ├──generate_pearson_graph.ipynb
│  └──hvg_proprocessing.ipynb
|
├──saved_embeddings
│  ├──inspect_saved.ipynb
│  ├──baron1
│  ├──baron2
│  ├──baron3
│  ├──baron4
│  ├──darmanis
│  ├──loh
│  ├──muraro
│  ├──segerstolpe
│  └──wang
|
├──top_genes
│  ├──schisto_example
│  │  ├──CellVGAE
│  │  ├──HVG
│  │  └──layer_weights
|  |
│  └──top_genes.ipynb
|
└──utils
   ├──attn_graph.py
   ├──cluster.py
   └──top_genes.py
```

Brief explanations for each subdirectory:

- `models` includes the model definition and logic for CellVGAE
- `utils` contains various helper functions used throughout the project
- `saved_embeddings` includes saved node embeddings, 2D UMAP representations and clusters for the 9 benchmarks and a notebook for visualisation
- `preprocessing` has small notebooks used to generate the inputs to CellVGAE, i.e. the highly variable genes files and the graphs
- `top_genes` gives an example notebook on how to identify the high-weight genes, corresponds to Section 5.1.2 *Finding marker genes*
- `misc` contains code to curate or clean some of the datasets (actual data not included in the repository because of size)
- `figures` includes the code necessary to generate all the figures and the code for the respective analyses:
  - `attention_visualisation` corresponds to Section 5.4 *Interpretability*
  - `baron4` corresponds to Appendix D *Clustering on the Baron4 dataset*
  - `macrophages` corresponds to Appendix A *The macrophages dataset*
  - `muraro` corresponds to Section 5.3.1 *Results*
  - `pbmc3k/clustering` corresponds to Section 5.2.1 *Clustering performance*
  - `pbmc3k/expression` corresponds to Section 5.2.2 *Visualising gene expression*
  - `schisto`  corresponds to Section 5.1 *The Schistosoma mansoni dataset*
