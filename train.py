# """Train CellVGAE

# Usage:
#   train.py --hvg_file <hvg> --graph_file <graph> --num_hidden_layers (2 | 3) [--num_heads=<heads>] [-hd <hd> ]... [--d <d>]... [--latent_dim=<latent_dim>]

# Options:
#   -h --help     Show this screen.
#   --version     Show version.
# """

import os
import argparse
from multiprocessing.sharedctypes import Value
from random import choices
from pathlib import Path
import torch

import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import faiss

import umap
import hdbscan
from tqdm.auto import tqdm
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import train_test_split_edges, to_undirected
from sklearn.preprocessing import MinMaxScaler

from models import CellVGAE, CellVGAE_Encoder
from models import mmd

def prepare_training_data(args):
    if args['input_gene_expression_path'][-5:] == '.h5ad':
        adata = anndata.read_h5ad(args['input_gene_expression_path'])
    elif args['input_gene_expression_path'][-4:] == '.csv':
        adata = anndata.read_csv(args['input_gene_expression_path'])

    if args['raw_counts']:
        adata_pp = sc.pp.recipe_seurat(adata, log=True, copy=True)
    else:
        adata_pp = sc.pp.log1p(adata, copy=True)

    adata_hvg = sc.pp.highly_variable_genes(adata_pp, n_top_genes=args['hvg'], inplace=False, flavor='seurat')
    adata_khvg = sc.pp.highly_variable_genes(adata_pp, n_top_genes=args['khvg'], inplace=False, flavor='seurat')

    X_hvg = np.array(adata[:, adata_hvg[adata_hvg['highly_variable']].index.values].X)
    X_khvg = np.array(adata[:, adata_khvg[adata_khvg['highly_variable']].index.values].X)

    if args['transpose_input']:
        X_hvg = X_hvg.T
        X_khvg = X_khvg.T

    return adata_hvg, adata_khvg, X_hvg, X_khvg


def knn_faiss(data_numpy, k, metric='euclidean', use_gpu=False):    
    data_numpy = data_numpy.astype(np.float32)
    data_numpy = data_numpy.copy(order='C')

    if use_gpu:
        print('Using GPU for Faiss...')
        res = faiss.StandardGpuResources()
    else:
        print('Using CPU for Faiss...')
    
    if metric == 'euclidean':
        index = faiss.IndexFlatL2(data_numpy.shape[1])
    elif metric == 'manhattan':
        index = faiss.IndexFlat(data_numpy.shape[1], faiss.METRIC_L1)
    elif metric == 'cosine':
        index = faiss.IndexFlat(data_numpy.shape[1], faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(data_numpy)

    if use_gpu:
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
    index.train(data_numpy)
    assert index.is_trained

    index.add(data_numpy)
    nprobe = data_numpy.shape[0]
    index.nprobe = nprobe
    distances, neighbors = index.search(data_numpy, k)
            
    return distances, neighbors


def correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy)
    corr = df.corr(method=corr_type)
    nlargest = k + 1
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    
    return corr, neighbors
    


def prepare_graphs(adata_khvg, X_khvg, args):
    if args['graph_type'] == 'KNN Scanpy':
        distances = sc.pp.neighbors(adata_khvg, n_neighbors=args['k'] + 1, n_pcs=args['graph_n_pcs'], knn=True, metric=args['graph_metric'], copy=True).obsp['distances'].A
        neighbors = np.nonzero(distances)[1].reshape(-1, args['k'] + 1)
    elif args['graph_type'] == 'KNN Faiss':
        distances, neighbors = knn_faiss(data_numpy=X_khvg, k=args['k'] + 1, metric=args['graph_metric'], use_gpu=args['faiss_gpu'])
    elif args['graph_type'] == 'PKNN':
        distances, neighbors = correlation(data_numpy=X_khvg, k=args['k'] + 1)

    if args['graph_distance_cutoff_num_stds']:
        cutoff = np.mean(distances) + float(args['graph_distance_cutoff_num_stds']) * np.std(distances)


    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(1, neighbors.shape[1]):
            pair = (str(i), str(neighbors[i][j]))
            if args['graph_distance_cutoff_num_stds']:
                distance = distances[i][j]
                if distance < cutoff:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)
            else:
                if i != neighbors[i][j]:
                    edgelist.append(pair)


    if args['graph_save_dir']:
        Path(args['graph_save_dir']).mkdir(exist_ok=True)

        num_hvg = X_khvg.shape[1]
        k_file = args['k'] - 1
        if args['graph_type'] == 'KNN Scanpy':
            graph_name = 'Scanpy'
        elif args['graph_type'] == 'KNN Faiss':
            graph_name = 'Faiss'
        elif args['graph_type'] == 'PKNN':
            graph_name = 'Pearson'

        filename = f'{graph_name}_KNN_K{k_file}_KHVG_{num_hvg}.txt'

        final_path = os.path.join(args['graph_save_dir'], filename)
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist


def train(model, optimizer, train_loader, loss, device, use_decoder_loss=False):
    model = model.train()

    epoch_loss = 0.0

    saves = []

    for batch_idx, batch in enumerate(train_loader):
        x, edge_index = batch.x.to(torch.float).to(device), batch.edge_index.to(torch.long).to(device)

        optimizer.zero_grad()
        z, _ = model.encode(x, edge_index)
        reconstruction_loss = model.recon_loss(z, edge_index)

        if loss == 'mmd':
            true_samples = Variable(torch.randn(x.shape[0], LATENT_DIM), requires_grad=False)
            mmd_loss = mmd.compute_mmd(true_samples.to(device), z)

            loss = reconstruction_loss + mmd_loss
        else:
            num_features = len(train_loader.dataset)
            loss = reconstruction_loss + (1 / num_features) * model.kl_loss()

        decoder_loss = 0.0
        if use_decoder_loss:
            reconstructed_features = model.decoder_nn(z)
            decoder_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
            loss += decoder_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss, decoder_loss


def test(x, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(x.to(torch.float), train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def setup(args):
    adata_hvg, adata_khvg, X_hvg, X_khvg = prepare_training_data(args)
    edgelist = prepare_graphs(adata_khvg, X_khvg, args)

    edge_index = np.array(edgelist).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes=X_hvg.shape[0])

    scaler = MinMaxScaler()
    scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))

    data_obj = Data(edge_index=edge_index, x=scaled_x)
    data_obj.num_nodes = X_hvg.shape[0]

    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None

    # Can set validation ratio
    data = train_test_split_edges(data_obj, val_ratio=args['val_split'], test_ratio=0)
    x, train_pos_edge_index = data.x.to(torch.double), data.train_pos_edge_index

    num_features = data_obj.num_features
    train_loader = DataLoader([Data(edge_index=train_pos_edge_index, x=x)], batch_size=1)

    if args['graph_convolution'] in ['GAT', 'GATv2']:
        num_heads = {}
        if len(args['num_heads']) == 4:
            num_heads['first'] = args['num_heads'][0]
            num_heads['second'] = args['num_heads'][1]
            num_heads['mean'] = args['num_heads'][2]
            num_heads['std'] = args['num_heads'][3]
        elif len(args['num_heads']) == 5:
            num_heads['first'] = args['num_heads'][0]
            num_heads['second'] = args['num_heads'][1]
            num_heads['third'] = args['num_heads'][2]
            num_heads['mean'] = args['num_heads'][3]
            num_heads['std'] = args['num_heads'][4]

        encoder = CellVGAE_Encoder.CellVGAE_Encoder(
            in_channels=num_features, num_hidden_layers=args['num_hidden_layers'],
            num_heads=num_heads,
            hidden_dims=args['hidden_dims'],
            dropout=args['dropout'],
            latent_dim=args['latent_dim'],
            v2=args['graph_convolution'] == 'GATv2',
            concat={'first': True, 'second': True})
    else:
        encoder = CellVGAE_Encoder.CellVGAE_GCNEncoder(
            in_channels=num_features,
            num_hidden_layers=args['num_hidden_layers'],
            hidden_dims=args['hidden_dims'],
            latent_dim=args['latent_dim'])

    model = CellVGAE.CellVGAE(encoder=encoder, decoder_nn_dim1=args['decoder_nn_dim1'], gcn_or_gat=args['graph_convolution'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    model = model.to(device)

    return model, optimizer, train_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train', description='Train CellVGAE')

    parser.add_argument('--input_gene_expression_path', help='Input gene expression file path')
    parser.add_argument('--hvg', type=int, help='Number of HVGs')
    parser.add_argument('--khvg', type=int, help='Number of KHVGs')
    parser.add_argument('--graph_type', choices=['KNN Scanpy', 'KNN Faiss', 'PKNN'], help='Type of graph')
    parser.add_argument('--k', type=int, help='K for KNN or Pearson (PKNN) graph')
    parser.add_argument('--graph_n_pcs', type=int, help='Use this many Principal Components for the KNN (only Scanpy)')
    parser.add_argument('--graph_metric', choice=['euclidean', 'manhattan', 'cosine'])
    parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0, help='Number of standard deviations to add to the mean of distances for KNN graphs (both Scanpy and Faiss). Can be negative')
    parser.add_argument('--graph_save_dir', help='(Optional) save the generated graph to this path')
    parser.add_argument('--raw_counts', action='store_true')
    parser.add_argument('--faiss_gpu', action='store_true', help='Use Faiss on the GPU (only for KNN Faiss)')
    # parser.add_argument('--log_normalise', action=argparse.BooleanOptionalAction)
    parser.add_argument('--hvg_file', type=int, help='HVG file if not using command line options to generate it')
    parser.add_argument('--graph_file', help='Graph specified as an edge list (one per line, separated by whitespace)')
    parser.add_argument('--graph_convolution', choices=['GAT', 'GATv2', 'GCN'])
    parser.add_argument('--num_hidden_layers', help='Number of hidden layers (must be 2 or 3)', default=2, type=int)
    parser.add_argument('--num_heads', help='Number of attention heads', type=int, nargs='*')
    parser.add_argument('--hidden_dims', help='Output dimension for each hidden layer (only 2 or 3 layers allowed)', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dropout', help='Dropout for each hidden layer (only 2 or 3 layers allowed)', type=float, nargs='*', default=[0.2, 0.2])
    parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings)', default=50, type=int)
    parser.add_argument('--loss', help='Loss function (KL or MMD)', choices=['kl', 'mmd'], default='kl')
    parser.add_argument('--lr', help='Learning rate for Adam', default=0.0001, type=float)
    parser.add_argument('--epochs', help='Number of training epochs', default=250, type=int)
    parser.add_argument('--val_split', help='Validation split e.g. 0.1', default=0.0, type=float)
    parser.add_argument('--node_out', help='Output file name and path for the computed node embeddings (saved in numpy .npy format)', default='node_embs.npy')
    parser.add_argument('--save_trained_model', help='Path to save PyTorch model', default='model.pt')
    parser.add_argument('--name', type=str, help='Project name')
    parser.add_argument('--transpose_input', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_linear_decoder', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--decoder_nn_dim1', type=int)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = vars(args)
    num_hidden_layers = args['num_hidden_layers']
    hidden_dims = args['hidden_dims']
    num_heads = args['num_heads']
    dropout = args['dropout']
    conv_type = args['graph_convolution']

    if (args['hvg_file'] is not None) and (args['hvg'] is not None or args['khvg'] is not None):
        raise ValueError('Cannot use custom HVG file when --hvg or --khvg is specified.')

    if (args['graph_file'] is not None) and (args['graph_type'] is not None):
        raise ValueError('Cannot use custom HVG file when --hvg or --khvg is specified.')

    if (conv_type == 'GCN') and (num_heads is not None or dropout is not None):
        raise ValueError('GCN convolution not available with --num_heads or --dropout.')

    if (args['graph_type'] != 'Faiss KNN') and (args['faiss_gpu'] is not None):
        raise ValueError('Must use Faiss KNN if providing --faiss_gpu.')

    assert (len(dropout) == num_hidden_layers + 2), 'Number of hidden dropout values must match number of hidden layers.'
    assert (len(num_heads) == num_hidden_layers + 2), 'Number of hidden output dimensions must match number of hidden layers.'
    assert (len(hidden_dims) == num_hidden_layers), 'Number of hidden output dimensions must match number of hidden layers.'
    assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'Number of hidden layers must be 2 or 3.'

    # model, optimizer, train_loader, data_list = setup(args)
    model, optimizer, train_loader = setup(args)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(device))
    print(model)

    if args['use_linear_decoder']:
        print('Using linear feature decoder.')
    else:
        print('No feature decoder used.')

    for epoch in tqdm(range(1, args['epochs'] + 1)):
        epoch_loss, decoder_loss = train(model, optimizer, train_loader, args['loss'], device=device, use_decoder_loss=args['use_linear_decoder'])
        print('Epoch {:03d} -- {:.4f} -- {:.4f}'.format(epoch, epoch_loss, decoder_loss))

        # Uncomment if using validation
        #     auc, ap = test(x.to(torch.float), data.val_pos_edge_index, data.val_neg_edge_index)
        #     print('Epoch: {:03d} -- AUC: {:.4f} -- AP: {:.4f}'.format(epoch, auc, ap))

    if args['node_out']:
        model = model.eval()
        node_embeddings = []

        for batch_idx, batch in enumerate(train_loader):
            x, edge_index = batch.x.to(torch.float).to(device), batch.edge_index.to(torch.long).to(device)
            z_nodes, _ = model.encode(x, edge_index)
            node_embeddings.append(z_nodes.cpu().detach().numpy())

        node_embeddings = np.array(node_embeddings)
        node_embeddings = node_embeddings.squeeze()
        np.save(args['node_out'], node_embeddings)

    if args['save_trained_model']:
        torch.save(model.state_dict(), args['save_trained_model'])

    print('Exiting...')
