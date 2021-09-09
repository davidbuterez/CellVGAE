# """Train CellVGAE

# Options:
#   -h --help     Show this screen.
#   --version     Show version.
# """

import sys
import os
import argparse
from multiprocessing.sharedctypes import Value
from random import choices
from pathlib import Path
from tkinter import E
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
from termcolor import colored

from models import CellVGAE, CellVGAE_Encoder
from models import mmd


def preprocess_raw_counts(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def load_input_file(path):
    if path[-5:] == '.h5ad':
        adata = anndata.read_h5ad(path)
    elif path[-4:] == '.csv':
        adata = anndata.read_csv(path)
    return adata


def prepare_training_data(args):
    print('Preparing training data...')
    adata = load_input_file(args['input_gene_expression_path'])
    print(f'Original data shape: {adata.shape}')

    if args['transpose_input']:
        print(f'Transposing input to {adata.shape[::-1]}...')
        adata = adata.copy().transpose()

    adata_pp = adata.copy()
    if args['raw_counts']:
        print('Applying raw counts preprocessing...')
        # sc.pp.recipe_seurat(adata_pp, log=True, copy=False)
        preprocess_raw_counts(adata_pp)
    else:
        print('Applying log-normalisation...')
        sc.pp.log1p(adata_pp, copy=False)


    adata_hvg = adata_pp.copy()
    adata_khvg = adata_pp.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=args['hvg'], inplace=True, flavor='seurat')
    sc.pp.highly_variable_genes(adata_khvg, n_top_genes=args['khvg'], inplace=True, flavor='seurat')

    # X_hvg = np.array(adata[:, adata_hvg[adata_hvg['highly_variable']].index.values].X)
    # X_khvg = np.array(adata[:, adata_khvg[adata_khvg['highly_variable']].index.values].X)

    # adata_hvg = adata[:, adata_hvg[adata_hvg['highly_variable']].index.values]
    # adata_khvg = adata[:, adata_khvg[adata_khvg['highly_variable']].index.values]

    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    adata_khvg = adata_khvg[:, adata_khvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    X_khvg = adata_khvg.X
    # X_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values].X
    # X_khvg = adata_khvg[:, adata_khvg.var['highly_variable'].values].X

    # if args['transpose_input']:
    #     X_hvg = X_hvg.T
    #     X_khvg = X_khvg.T
    #     adata_hvg = adata_hvg.copy().transpose()
    #     adata_khvg = adata_khvg.copy().transpose()
    
    print(f'HVG adata shape: {adata_hvg.shape}')
    print(f'KHVG adata shape: {adata_khvg.shape}')

    return adata_hvg, adata_khvg, X_hvg, X_khvg


def load_separate_hvg(hvg_path):
    adata = load_input_file(hvg_path)
    return adata


def load_separate_graph_edgelist(edgelist_path):
    edgelist = []
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist


def knn_faiss(data_numpy, k, metric='euclidean', use_gpu=False):    
    data_numpy = data_numpy.astype(np.float32)
    data_numpy = data_numpy.copy(order='C')
    data_numpy = np.ascontiguousarray(data_numpy, dtype=np.float32)

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

    data_numpy = np.ascontiguousarray(data_numpy, dtype=np.float32)
    index.train(data_numpy)
    assert index.is_trained

    index.add(data_numpy)
    nprobe = data_numpy.shape[0]
    index.nprobe = nprobe
    distances, neighbors = index.search(data_numpy, k)
            
    return distances, neighbors


def correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    
    return corr, neighbors
    

def prepare_graphs(adata_khvg, X_khvg, args):
    if args['graph_type'] == 'KNN Scanpy':
        print('Computing KNN Scanpy graph ("{}" metric)...'.format(args['graph_metric']))
        distances = sc.pp.neighbors(adata_khvg, n_neighbors=args['k'] + 1, n_pcs=args['graph_n_pcs'], knn=True, metric=args['graph_metric'], copy=True).obsp['distances'].A
        
        # Scanpy might not always return neighbors for all graph nodes. Missing nodes have a -1 in the neighbours matrix.
        neighbors = np.full(distances.shape, fill_value=-1)
        neighbors[np.nonzero(distances)] = distances[np.nonzero(distances)]

        # neighbors = distances[np.nonzero(distances)[0]].reshape(adata_khvg.shape[0], args['k'])
        # print(neighbors.shape)
        # from collections import Counter
        # print(np.nonzero(distances)[0])
        # c = Counter(np.nonzero(distances)[0])
        # print(c)
        # print({print(k) : v for k, v in dict(c).items() if v != 10})
        # print(distances)
        # print(np.nonzero(distances)[1][:25])
        # neighbors = np.nonzero(distances)[1].reshape(np.nonzero(distances)[1].shape[0], args['k'])
    elif args['graph_type'] == 'KNN Faiss':
        print('Computing KNN Faiss graph ("{}" metric)...'.format(args['graph_metric']))
        distances, neighbors = knn_faiss(data_numpy=X_khvg, k=args['k'] + 1, metric=args['graph_metric'], use_gpu=args['faiss_gpu'])
    elif args['graph_type'] == 'PKNN':
        print('Computing PKNN graph...')
        distances, neighbors = correlation(data_numpy=X_khvg, k=args['k'] + 1)

    if args['graph_distance_cutoff_num_stds']:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(args['graph_distance_cutoff_num_stds']) * np.std(np.nonzero(distances), axis=None)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if args['graph_distance_cutoff_num_stds']:
                    distance = distances[i][j]
                    if distance < cutoff:
                        # print(distance, cutoff)
                        if i != neighbors[i][j]:
                            # print('appended')
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    print(f'The graph has {len(edgelist)} edges.')

    if args['graph_save_path']:
        Path(args['graph_save_path']).mkdir(parents=True, exist_ok=True)

        num_hvg = X_khvg.shape[1] if args['transpose_input'] else X_khvg.shape[0]
        k_file = args['k']
        if args['graph_type'] == 'KNN Scanpy':
            graph_name = 'Scanpy'
        elif args['graph_type'] == 'KNN Faiss':
            graph_name = 'Faiss'
        elif args['graph_type'] == 'PKNN':
            graph_name = 'Pearson'

        if args['name']:
            filename = f'{args["name"]}_{graph_name}_KNN_K{k_file}_KHVG_{num_hvg}.txt'
        else:
            filename = f'{graph_name}_KNN_K{k_file}_KHVG_{num_hvg}.txt'
        if args['graph_n_pcs']:
            filename = filename.split('.')[0] + f'_d_{args["graph_n_pcs"]}.txt'
        if args['graph_distance_cutoff_num_stds']:
            filename = filename.split('.')[0] + '_cutoff_{:.4f}.txt'.format(cutoff)

        final_path = os.path.join(args['graph_save_path'], filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist


def train(model, optimizer, train_loader, loss, device, use_decoder_loss=False, conv_type='GAT'):
    model = model.train()

    epoch_loss = 0.0

    saves = []

    for batch_idx, batch in enumerate(train_loader):
        x, edge_index = batch.x.to(torch.float).to(device), batch.edge_index.to(torch.long).to(device)

        optimizer.zero_grad()
        
        if conv_type in ['GAT', 'GATv2']:
            z, _ = model.encode(x, edge_index)
        else:
            z = model.encode(x, edge_index)
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
            try:
                reconstructed_features = model.decoder_nn(z)
            except AttributeError as ae:
                print()
                print(colored('Exception: ' + str(ae), 'red'))
                print('Need to provide the first hidden dimension for the decoder with --decoder_nn_dim1.')
                sys.exit(1)
                
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
    if not args['hvg_file_path']:
        adata_hvg, adata_khvg, X_hvg, X_khvg = prepare_training_data(args)
    else:
        assert args['khvg_file_path'] is not None
        adata_hvg = load_separate_hvg(hvg_path=args['hvg_file_path'])
        adata_khvg = load_separate_hvg(hvg_path=args['khvg_file_path'])
        if args['transpose_input']:
            print(f'Transposing input HVG file to {adata_hvg.shape[::-1]}...')
            adata_hvg = adata_hvg.copy().transpose()
            print(f'Transposing input KHVG file to {adata_khvg.shape[::-1]}...')
            adata_khvg = adata_khvg.copy().transpose()
        
        X_hvg = adata_hvg.X
        X_khvg = adata_khvg.X

    if not args['graph_file_path']:
        try:
            edgelist = prepare_graphs(adata_khvg, X_khvg, args)
        except ValueError as ve:
            print()
            print(colored('Exception: ' + str(ve), 'red'))
            print('Might need to transpose input with the --transpose_input argument.')
            sys.exit(1)
    else:
        edgelist = load_separate_graph_edgelist(args['graph_file_path'])

    num_nodes = X_hvg.shape[0] if args['transpose_input'] else X_hvg.shape[1]
    print(f'Number of nodes in graph: {num_nodes}.')
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)

    scaler = MinMaxScaler()
    scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))

    data_obj = Data(edge_index=edge_index, x=scaled_x)
    data_obj.num_nodes = X_hvg.shape[0]

    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None

    # Can set validation ratio
    try:
        data = train_test_split_edges(data_obj, val_ratio=args['val_split'], test_ratio=0)
    except IndexError as ie:
        print()
        print(colored('Exception: ' + str(ie), 'red'))
        print('Might need to transpose input with the --transpose_input argument.')
        sys.exit(1)

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
    parser = argparse.ArgumentParser(prog='train', description='Train CellVGAE.')

    parser.add_argument('--input_gene_expression_path', help='Input gene expression file path.')
    parser.add_argument('--hvg', type=int, help='Number of HVGs.')
    parser.add_argument('--khvg', type=int, help='Number of KHVGs.')
    parser.add_argument('--graph_type', choices=['KNN Scanpy', 'KNN Faiss', 'PKNN'], help='Type of graph')
    parser.add_argument('--k', type=int, help='K for KNN or Pearson (PKNN) graph.')
    parser.add_argument('--graph_n_pcs', type=int, help='Use this many Principal Components for the KNN (only Scanpy).')
    parser.add_argument('--graph_metric', choices=['euclidean', 'manhattan', 'cosine'], default='euclidean')
    parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0, help='Number of standard deviations to add to the mean of distances/correlation values. Can be negative.')
    parser.add_argument('--graph_save_path', help='(Optional) save the generated graph to this path. Will create the entire path if necessary.')
    parser.add_argument('--raw_counts', action='store_true', default=False)
    parser.add_argument('--faiss_gpu', action='store_true', help='Use Faiss on the GPU (only for KNN Faiss).', default=False)
    parser.add_argument('--hvg_file_path', help='HVG file if not using command line options to generate it.')
    parser.add_argument('--khvg_file_path', help='KHVG file if not using command line options to generate it. Can be the same file as --hvg_file_path if HVG = KHVG.')
    parser.add_argument('--graph_file_path', help='(Optional) Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma).')
    parser.add_argument('--graph_convolution', choices=['GAT', 'GATv2', 'GCN'], default='GAT')
    parser.add_argument('--num_hidden_layers', help='Number of hidden layers (must be 2 or 3).', choices=[2, 3], type=int)
    parser.add_argument('--num_heads', help='Number of attention heads', type=int, nargs='*')
    parser.add_argument('--hidden_dims', help='Output dimension for each hidden layer (only 2 or 3 layers allowed).', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dropout', help='Dropout for each hidden layer (only 2 or 3 layers allowed).', type=float, nargs='*')
    parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings).', default=50, type=int)
    parser.add_argument('--loss', help='Loss function (KL or MMD).', choices=['kl', 'mmd'], default='kl')
    parser.add_argument('--lr', help='Learning rate for Adam.', default=0.0001, type=float)
    parser.add_argument('--epochs', help='Number of training epochs.', default=50, type=int)
    parser.add_argument('--val_split', help='Validation split e.g. 0.1', default=0.0, type=float)
    parser.add_argument('--transpose_input', action='store_true', default=False, help='Specify if inputs should be transposed.')
    parser.add_argument('--use_linear_decoder', action='store_true', default=False, help='Turn on a neural network decoder, similar to traditional VAEs.')
    parser.add_argument('--decoder_nn_dim1', help='First hidden dimenson for the neural network decoder, if specified using --use_linear_decoder.', type=int)
    parser.add_argument('--name', help='Name used for the written output files.', type=str)
    parser.add_argument('--node_embeddings_save_path', help='(Optional) Output path for the computed node embeddings (saved in numpy .npy format). Will create the entire path if necessary.', type=str)
    parser.add_argument('--model_save_path', help='(Optional) Path to save PyTorch model. Will create the entire path if necessary.', type=str)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = vars(args)
    num_hidden_layers = args['num_hidden_layers']
    hidden_dims = args['hidden_dims']
    num_heads = args['num_heads']
    dropout = args['dropout']
    conv_type = args['graph_convolution']

    assert args['latent_dim'] > 0, 'Number of latent dimensions must be greateer than 0.'
    assert args['epochs'] > 0, 'Number of epochs must be greateer than 0.'
    assert args['lr'] > 0, 'Learning rate must be greateer than 0.'
    assert args['val_split'] >= 0, 'Negative values for the validation split are not allowed.'

    if args['decoder_nn_dim1']:
        assert args['decoder_nn_dim1'] > 0, 'Number of latent dimensions must be greateer than 0.'

    if args['graph_n_pcs']:
        assert args['graph_n_pcs'] >= 0, 'Negative values for the Scanpy argument "graph_n_pcs" are not allowed.'

    if args['k']:
        assert args['k'] > 0, 'The value of k for (P)KNN graphs must be larger than 0.'

    if (args['hvg_file_path'] is not None) and (args['input_gene_expression_path'] is not None):
        raise ValueError('Cannot use custom HVG file when --input_gene_expression_path is specified.')

    if (args['hvg_file_path'] is not None) and (args['hvg'] is not None or args['khvg'] is not None):
        raise ValueError('Cannot use custom HVG file when --hvg or --khvg is specified.')

    if (args['khvg_file_path'] is not None) and (args['input_gene_expression_path'] is not None):
        raise ValueError('Cannot use custom KHVG file when --input_gene_expression_path is specified.')

    if (args['khvg_file_path'] is not None) and (args['hvg'] is not None or args['khvg'] is not None):
        raise ValueError('Cannot use custom KHVG file when --hvg or --khvg is specified.')

    if (args['hvg_file_path'] and not args['khvg_file_path']) or (args['hvg_file_path'] and not args['khvg_file_path']):
        raise ValueError('--hvg_file_path and --khvg_file_path must be specified simultaneously.')


    if (args['graph_file_path'] is not None) and (args['graph_type'] is not None):
        raise ValueError('Cannot use custom graph file when --graph_type is specified.')

    if (args['graph_file_path'] is not None) and (args['k'] is not None):
        raise ValueError('Cannot use custom graph file when --k is specified.')

    if (args['graph_file_path'] is not None) and (args['graph_save_path'] is not None):
        raise ValueError('Cannot use custom graph file when --graph_save_path is specified.')

    if (conv_type == 'GCN') and (num_heads is not None or dropout is not None):
        raise ValueError('GCN convolution not available with --num_heads or --dropout.')

    if (args['graph_type'] != 'KNN Faiss') and (args['faiss_gpu']):
        raise ValueError('Must use Faiss KNN if providing --faiss_gpu.')

    if (args['hvg'] and not args['khvg']) or (args['khvg'] and not args['hvg']):
        raise ValueError('--hvg and --khvg must be specified simultaneously.')

    if (args['graph_n_pcs']) and (args['graph_type'] != 'KNN Scanpy'):
        raise ValueError('--graph_n_pcs is valid only for the "KNN Scanpy" graph type.')

    if args['dropout']:
        assert (len(dropout) == num_hidden_layers + 2), 'Number of hidden dropout values must match number of hidden layers.'
    if args['num_heads']:
        assert (len(num_heads) == num_hidden_layers + 2), 'Number of hidden output dimensions must match number of hidden layers.'
    assert (len(hidden_dims) == num_hidden_layers), 'Number of hidden output dimensions must match number of hidden layers.'
    assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'Number of hidden layers must be 2 or 3.'

    if (args['decoder_nn_dim1']) and (not args['use_linear_decoder']):
        print()
        print(colored('WARNING: --decoder_nn_dim1 provided but --use_linear_decoder is not set. Ignoring --decoder_nn_dim1.\n', 'yellow'))

    # model, optimizer, train_loader, data_list = setup(args)
    model, optimizer, train_loader = setup(args)
    if torch.cuda.is_available():
        print(f'CUDA available, using {torch.cuda.get_device_name(device)}.')
    print('Neural model details: \n')
    print(model)
    print()

    print(f'Using {args["latent_dim"]} latent dimensions.')
    if args['use_linear_decoder']:
        print('Using linear feature decoder.\n')
    else:
        print('No feature decoder used.\n')

    for epoch in tqdm(range(1, args['epochs'] + 1)):
        epoch_loss, decoder_loss = train(model, optimizer, train_loader, args['loss'], device=device, use_decoder_loss=args['use_linear_decoder'], conv_type=args['graph_convolution'])
        if args['use_linear_decoder']:
            print('Epoch {:03d} -- Total epoch loss: {:.4f} -- NN decoder epoch loss: {:.4f}'.format(epoch, epoch_loss, decoder_loss))
        else:
            print('Epoch {:03d} -- Total epoch loss: {:.4f}'.format(epoch, epoch_loss))

        # Uncomment if using validation
        #     auc, ap = test(x.to(torch.float), data.val_pos_edge_index, data.val_neg_edge_index)
        #     print('Epoch: {:03d} -- AUC: {:.4f} -- AP: {:.4f}'.format(epoch, auc, ap))

    if args['node_embeddings_save_path']:
        model = model.eval()
        node_embeddings = []

        for batch_idx, batch in enumerate(train_loader):
            x, edge_index = batch.x.to(torch.float).to(device), batch.edge_index.to(torch.long).to(device)
            if args['graph_convolution'] in ['GAT', 'GATv2']:
                z_nodes, _ = model.encode(x, edge_index)
            else:
                z_nodes = model.encode(x, edge_index)
            node_embeddings.append(z_nodes.cpu().detach().numpy())

        node_embeddings = np.array(node_embeddings)
        node_embeddings = node_embeddings.squeeze()

        Path(args['node_embeddings_save_path']).mkdir(parents=True, exist_ok=True)

        if args['name']:
            filename = f'{args["name"]}_cellvgae_node_embeddings.npy'
        else:
            filename = 'cellvgae_node_embeddings.npy'

        node_filepath = os.path.join(args['node_embeddings_save_path'], filename)
        np.save(node_filepath, node_embeddings)

    if args['model_save_path']:
        Path(args['model_save_path']).mkdir(parents=True, exist_ok=True)
        
        if args['name']:
            filename = f'{args["name"]}_cellvgae_model.npy'
        else:
            filename = 'cellvgae_model.npy'

        model_filepath = os.path.join(args['model_save_path'], filename)
        torch.save(model.state_dict(), model_filepath)

    print('Exiting...')
