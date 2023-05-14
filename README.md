# CellVGAE

An unsupervised scRNA-seq analysis workflow with graph attention networks

![](figures/workflow.png)



CellVGAE uses the connectivity between cells (such as *k*-nearest neighbour graphs or KNN) with gene expression values as node features to learn high-quality cell representations in a lower-dimensional space, with applications in downstream analyses like (density-based) clustering, visualisation, gene set enrichment analysis and others. CellVGAE leverages both the variational graph autoencoder and graph attention networks to offer a powerful and more interpretable machine learning approach. It is implemented in PyTorch using the PyTorch Geometric library.

## Requirements

Installing CellVGAE with pip will attempt to install PyTorch and PyTorch Geometric, however it is recommended that the appropriate GPU/CPU versions are installed manually beforehand. For Linux:

1. Install PyTorch GPU: 

   ```conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia```

   or PyTorch CPU:  

   ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```

   
   
2. Install PyTorch Geometric:  

   `conda install pyg -c pyg -c conda-forge`
   
   

3. (Optional) Install Faiss CPU:  

   `conda install -c pytorch faiss-cpu`

   

   Faiss is only required if using the option `--graph_type "KNN Faiss"` .  It is a soft dependency as it is not available for some platforms (currently Apple M1). Attempting to use CellVGAE with Faiss without installing it will result in an exception.

   A GPU version of Faiss for CUDA 11.1 is not yet available.

   

4. Install CellVGAE with pip:

   `pip install cellvgae --pre`
   
   

5. (Optional) For the attention graph visualisations of Figure 6, `igraph` is required:

   `pip install python-igraph`




If using the R preprocessing code, we recommend installing the following:

`Seurat 3`, `scran`, `SingleCellExperiment`. `scRNAseq`, `BiocSingular`, `igraph`, `dplyr` and `textshape`.



## Example use

Using the example files in this repo (.h5ad file is the same as downloaded by Scanpy 1.8.1):

```bash
python -m cellvgae --input_gene_expression_path "example_data/paul15_myeloid_scanpy.h5ad" --graph_file_path "example_data/paul15_Faiss_KNN_K3_KHVG2500.txt" --graph_convolution "GAT" --num_hidden_layers 2 --hidden_dims 128 128 --num_heads 3 3 3 3 --dropout 0.4 0.4 0.4 0.4 --latent_dim 50 --epochs 50 --model_save_path "model_saved_out"
```

Other examples are available in `examples/cellvgae_example_scripts.txt`

(also consult the help section below)



## Usage

Invoke the training script with `python -m cellvgae` with the arguments detailed below:

```
usage: train [-h] [--input_gene_expression_path INPUT_GENE_EXPRESSION_PATH] [--hvg HVG] [--khvg KHVG] [--graph_type {KNN Scanpy,KNN Faiss,PKNN}] [--k K] [--graph_n_pcs GRAPH_N_PCS]
             [--graph_metric {euclidean,manhattan,cosine}] [--graph_distance_cutoff_num_stds GRAPH_DISTANCE_CUTOFF_NUM_STDS] [--save_graph] [--raw_counts] [--faiss_gpu]
             [--hvg_file_path HVG_FILE_PATH] [--khvg_file_path KHVG_FILE_PATH] [--graph_file_path GRAPH_FILE_PATH] [--graph_convolution {GAT,GATv2,GCN}] [--num_hidden_layers {2,3}]
             [--num_heads [NUM_HEADS [NUM_HEADS ...]]] [--hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]] [--dropout [DROPOUT [DROPOUT ...]]] [--latent_dim LATENT_DIM] [--loss {kl,mmd}] [--lr LR]
             [--epochs EPOCHS] [--val_split VAL_SPLIT] [--test_split TEST_SPLIT] [--transpose_input] [--use_linear_decoder] [--decoder_nn_dim1 DECODER_NN_DIM1] [--name NAME] --model_save_path MODEL_SAVE_PATH [--umap] [--hdbscan]

Train CellVGAE.

optional arguments:
  -h, --help            show this help message and exit
  --input_gene_expression_path INPUT_GENE_EXPRESSION_PATH
                        Input gene expression file path.
  --hvg HVG             Number of HVGs.
  --khvg KHVG           Number of KHVGs.
  --graph_type {KNN Scanpy,KNN Faiss,PKNN}
                        Type of graph.
  --k K                 K for KNN or Pearson (PKNN) graph.
  --graph_n_pcs GRAPH_N_PCS
                        Use this many Principal Components for the KNN (only Scanpy).
  --graph_metric {euclidean,manhattan,cosine}
  --graph_distance_cutoff_num_stds GRAPH_DISTANCE_CUTOFF_NUM_STDS
                        Number of standard deviations to add to the mean of distances/correlation values. Can be negative.
  --save_graph          Save the generated graph to the output path specified by --model_save_path.
  --raw_counts          Enable preprocessing recipe for raw counts.
  --faiss_gpu           Use Faiss on the GPU (only for KNN Faiss).
  --hvg_file_path HVG_FILE_PATH
                        HVG file if not using command line options to generate it.
  --khvg_file_path KHVG_FILE_PATH
                        KHVG file if not using command line options to generate it. Can be the same file as --hvg_file_path if HVG = KHVG.
  --graph_file_path GRAPH_FILE_PATH
                        Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma), if not using command line options to generate it.
  --graph_convolution {GAT,GATv2,GCN}
  --num_hidden_layers {2,3}
                        Number of hidden layers (must be 2 or 3).
  --num_heads [NUM_HEADS [NUM_HEADS ...]]
                        Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.
  --hidden_dims [HIDDEN_DIMS [HIDDEN_DIMS ...]]
                        Output dimension for each hidden layer. Input is a list that matches --num_hidden_layers in length.
  --dropout [DROPOUT [DROPOUT ...]]
                        Dropout for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.
  --latent_dim LATENT_DIM
                        Latent dimension (output dimension for node embeddings).
  --loss {kl,mmd}       Loss function (KL or MMD).
  --lr LR               Learning rate for Adam.
  --epochs EPOCHS       Number of training epochs.
  --val_split VAL_SPLIT
                        Validation split e.g. 0.1.
  --test_split TEST_SPLIT
                        Test split e.g. 0.1.
  --transpose_input     Specify if inputs should be transposed.
  --use_linear_decoder  Turn on a neural network decoder, similar to traditional VAEs.
  --decoder_nn_dim1 DECODER_NN_DIM1
                        First hidden dimenson for the neural network decoder, if specified using --use_linear_decoder.
  --name NAME           Name used for the written output files.
  --model_save_path MODEL_SAVE_PATH
                        Path to save PyTorch model and output files. Will create the entire path if necessary.
  --umap                Compute and save the 2D UMAP embeddings of the output node features.
  --hdbscan             Compute and save different HDBSCAN clusterings.
```

## Citation

```
@article{10.1093/bioinformatics/btab804,
    author = {Buterez, David and Bica, Ioana and Tariq, Ifrah and Andrés-Terré, Helena and Liò, Pietro},
    title = "{CellVGAE: an unsupervised scRNA-seq analysis workflow with graph attention networks}",
    journal = {Bioinformatics},
    volume = {38},
    number = {5},
    pages = {1277-1286},
    year = {2021},
    month = {12},
    abstract = "{Single-cell RNA sequencing allows high-resolution views of individual cells for libraries of up to millions of samples, thus motivating the use of deep learning for analysis. In this study, we introduce the use of graph neural networks for the unsupervised exploration of scRNA-seq data by developing a variational graph autoencoder architecture with graph attention layers that operates directly on the connectivity between cells, focusing on dimensionality reduction and clustering. With the help of several case studies, we show that our model, named CellVGAE, can be effectively used for exploratory analysis even on challenging datasets, by extracting meaningful features from the data and providing the means to visualize and interpret different aspects of the model.We show that CellVGAE is more interpretable than existing scRNA-seq variational architectures by analysing the graph attention coefficients. By drawing parallels with other scRNA-seq studies on interpretability, we assess the validity of the relationships modelled by attention, and furthermore, we show that CellVGAE can intrinsically capture information such as pseudotime and NF-ĸB activation dynamics, the latter being a property that is not generally shared by existing neural alternatives. We then evaluate the dimensionality reduction and clustering performance on 9 difficult and well-annotated datasets by comparing with three leading neural and non-neural techniques, concluding that CellVGAE outperforms competing methods. Finally, we report a decrease in training times of up to × 20 on a dataset of 1.3 million cells compared to existing deep learning architectures.The CellVGAE code is available at https://github.com/davidbuterez/CellVGAE.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab804},
    url = {https://doi.org/10.1093/bioinformatics/btab804},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/5/1277/49009403/btab804.pdf},
}
```
