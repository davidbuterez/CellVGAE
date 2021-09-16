# """Train CellVGAE

# Options:
#   -h --help     Show this screen.
#   --version     Show version.
# """

import sys
import os
import argparse
from pathlib import Path
import torch

import seaborn as sns
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import faiss
import random

import umap
import hdbscan
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch_geometric.transforms as T
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

from cellvgae import CellVGAE, CellVGAE_Encoder, CellVGAE_GCNEncoder, compute_mmd


def _user_prompt() -> bool:
    # Credit for function: https://stackoverflow.com/a/50216611
    """ Prompt the yes/no-*question* to the user. """
    from distutils.util import strtobool

    while True:
        user_input = input("[y/n]: ")
        try:
            return bool(strtobool(user_input))
        except ValueError:
            print("Please use y/n or yes/no.\n")


def _preprocess_raw_counts(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def _load_input_file(path):
    if path[-5:] == '.h5ad':
        adata = anndata.read_h5ad(path)
    elif path[-4:] == '.csv':
        adata = anndata.read_csv(path)
    return adata


def _prepare_training_data(args):
    print('Preparing training data...')
    adata = _load_input_file(args['input_gene_expression_path'])
    print(f'Original data shape: {adata.shape}')

    if (np.abs(adata.shape[0] - 20000) < 5000) and (adata.shape[0] / adata.shape[1] > 0.4) and not args['transpose_input']:
        print(colored('WARNING: --transpose_input not provided but input data might have genes in the fist dimension. Are you sure you want to continue?', 'yellow'))
        answer = _user_prompt()
        if not answer:
            sys.exit(0)

    if args['transpose_input']:
        print(f'Transposing input to {adata.shape[::-1]}...')
        adata = adata.copy().transpose()

    adata_pp = adata.copy()
    if args['raw_counts']:
        print('Applying raw counts preprocessing...')
        _preprocess_raw_counts(adata_pp)
    else:
        print('Applying log-normalisation...')
        sc.pp.log1p(adata_pp, copy=False)


    adata_hvg = adata_pp.copy()
    adata_khvg = adata_pp.copy()
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=args['hvg'], inplace=True, flavor='seurat')
    sc.pp.highly_variable_genes(adata_khvg, n_top_genes=args['khvg'], inplace=True, flavor='seurat')

    adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable'].values]
    adata_khvg = adata_khvg[:, adata_khvg.var['highly_variable'].values]
    X_hvg = adata_hvg.X
    X_khvg = adata_khvg.X

    print(f'HVG adata shape: {adata_hvg.shape}')
    print(f'KHVG adata shape: {adata_khvg.shape}')

    return adata_hvg, adata_khvg, X_hvg, X_khvg


def _load_separate_hvg(hvg_path):
    adata = _load_input_file(hvg_path)
    return adata


def _load_separate_graph_edgelist(edgelist_path):
    edgelist = []
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist


def _knn_faiss(data_numpy, k, metric='euclidean', use_gpu=False):    
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


def _correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    
    return corr, neighbors
    

def _prepare_graphs(adata_khvg, X_khvg, args):
    if args['graph_type'] == 'KNN Scanpy':
        print('Computing KNN Scanpy graph ("{}" metric)...'.format(args['graph_metric']))
        distances = sc.pp.neighbors(adata_khvg, n_neighbors=args['k'] + 1, n_pcs=args['graph_n_pcs'], knn=True, metric=args['graph_metric'], copy=True).obsp['distances'].A
        
        # Scanpy might not always return neighbors for all graph nodes. Missing nodes have a -1 in the neighbours matrix.
        neighbors = np.full(distances.shape, fill_value=-1)
        neighbors[np.nonzero(distances)] = distances[np.nonzero(distances)]

    elif args['graph_type'] == 'KNN Faiss':
        print('Computing KNN Faiss graph ("{}" metric)...'.format(args['graph_metric']))
        distances, neighbors = _knn_faiss(data_numpy=X_khvg, k=args['k'] + 1, metric=args['graph_metric'], use_gpu=args['faiss_gpu'])
    elif args['graph_type'] == 'PKNN':
        print('Computing PKNN graph...')
        distances, neighbors = _correlation(data_numpy=X_khvg, k=args['k'] + 1)
    # else:
    #     print(colored('Graph generation enabled but graph type not provided. Exiting...', 'red'))
    #     sys.exit(1)

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
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    print(f'The graph has {len(edgelist)} edges.')

    if args['save_graph']:
        Path(args['model_save_path']).mkdir(parents=True, exist_ok=True)

        num_hvg = X_khvg.shape[1]
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

        final_path = os.path.join(args['model_save_path'], filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist


def _train(model, optimizer, train_data, loss, device, use_decoder_loss=False, conv_type='GAT'):
    model = model.train()

    epoch_loss = 0.0

    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)

    optimizer.zero_grad()
    
    if conv_type in ['GAT', 'GATv2']:
        z, _ = model.encode(x, edge_index)
    else:
        z = model.encode(x, edge_index)
    reconstruction_loss = model.recon_loss(z, train_data.pos_edge_label_index)

    if loss == 'mmd':
        true_samples = Variable(torch.randn(x.shape[0], args['latent_dim']), requires_grad=False)
        mmd_loss = compute_mmd(true_samples.to(device), z)

        loss = reconstruction_loss + mmd_loss
    else:
        loss = reconstruction_loss + (1 / train_data.num_nodes) * model.kl_loss()

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


@torch.no_grad()
def _test(model, device, data, graph_conv):
    model.eval()
    if graph_conv in ['GAT', 'GATv2']:
        z, _ = model.encode(data.x.to(torch.float).to(device), data.edge_index.to(torch.long).to(device))
    else:
        z = model.encode(data.x.to(torch.float).to(device), data.edge_index.to(torch.long).to(device))
    return model.test(z, data.pos_edge_label_index.to(torch.long).to(device), data.neg_edge_label_index.to(torch.long).to(device))  


def _setup(args, device):
    if not args['hvg_file_path']:
        adata_hvg, adata_khvg, X_hvg, X_khvg = _prepare_training_data(args)
    else:
        assert args['khvg_file_path'] is not None
        adata_hvg = _load_separate_hvg(hvg_path=args['hvg_file_path'])
        adata_khvg = _load_separate_hvg(hvg_path=args['khvg_file_path'])
        if args['transpose_input']:
            print(f'Transposing input HVG file to {adata_hvg.shape[::-1]}...')
            adata_hvg = adata_hvg.copy().transpose()
            print(f'Transposing input KHVG file to {adata_khvg.shape[::-1]}...')
            adata_khvg = adata_khvg.copy().transpose()
        
        X_hvg = adata_hvg.X
        X_khvg = adata_khvg.X

    if not args['graph_file_path']:
        try:
            edgelist = _prepare_graphs(adata_khvg, X_khvg, args)
        except ValueError as ve:
            print()
            print(colored('Exception: ' + str(ve), 'red'))
            print('Might need to transpose input with the --transpose_input argument.')
            sys.exit(1)
    else:
        edgelist = _load_separate_graph_edgelist(args['graph_file_path'])

    num_nodes = X_hvg.shape[0]
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
        transform = T.RandomLinkSplit(num_val=args['val_split'], num_test=args['test_split'], is_undirected=True, add_negative_train_samples=False, split_labels=True)
        train_data, val_data, test_data = transform(data_obj)
    except IndexError as ie:
        print()
        print(colored('Exception: ' + str(ie), 'red'))
        print('Might need to transpose input with the --transpose_input argument.')
        sys.exit(1)

    num_features = data_obj.num_features

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

        encoder = CellVGAE_Encoder(
            in_channels=num_features, num_hidden_layers=args['num_hidden_layers'],
            num_heads=num_heads,
            hidden_dims=args['hidden_dims'],
            dropout=args['dropout'],
            latent_dim=args['latent_dim'],
            v2=args['graph_convolution'] == 'GATv2',
            concat={'first': True, 'second': True})
    else:
        encoder = CellVGAE_GCNEncoder(
            in_channels=num_features,
            num_hidden_layers=args['num_hidden_layers'],
            hidden_dims=args['hidden_dims'],
            latent_dim=args['latent_dim'])

    model = CellVGAE(encoder=encoder, decoder_nn_dim1=args['decoder_nn_dim1'], gcn_or_gat=args['graph_convolution'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    model = model.to(device)

    return model, optimizer, train_data, val_data, test_data


def _get_filepath(args, number_or_name, edge_or_weights):
    layer_filename = f'GAT_Layer_{number_or_name}_{edge_or_weights}.pt'
    return os.path.join(args['model_save_path'], layer_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train', description='Train CellVGAE.')

    parser.add_argument('--input_gene_expression_path', help='Input gene expression file path.')
    parser.add_argument('--hvg', type=int, help='Number of HVGs.')
    parser.add_argument('--khvg', type=int, help='Number of KHVGs.')
    parser.add_argument('--graph_type', choices=['KNN Scanpy', 'KNN Faiss', 'PKNN'], help='Type of graph.')
    parser.add_argument('--k', type=int, help='K for KNN or Pearson (PKNN) graph.')
    parser.add_argument('--graph_n_pcs', type=int, help='Use this many Principal Components for the KNN (only Scanpy).')
    parser.add_argument('--graph_metric', choices=['euclidean', 'manhattan', 'cosine'], default='euclidean')
    parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0, help='Number of standard deviations to add to the mean of distances/correlation values. Can be negative.')
    parser.add_argument('--save_graph', action='store_true', default=False, help='Save the generated graph to the output path specified by --model_save_path.')
    parser.add_argument('--raw_counts', action='store_true', default=False, help='Enable preprocessing recipe for raw counts.')
    parser.add_argument('--faiss_gpu', action='store_true', help='Use Faiss on the GPU (only for KNN Faiss).', default=False)
    parser.add_argument('--hvg_file_path', help='HVG file if not using command line options to generate it.')
    parser.add_argument('--khvg_file_path', help='KHVG file if not using command line options to generate it. Can be the same file as --hvg_file_path if HVG = KHVG.')
    parser.add_argument('--graph_file_path', help='Graph specified as an edge list (one edge per line, nodes separated by whitespace, not comma), if not using command line options to generate it.')
    parser.add_argument('--graph_convolution', choices=['GAT', 'GATv2', 'GCN'], default='GAT')
    parser.add_argument('--num_hidden_layers', help='Number of hidden layers (must be 2 or 3).', choices=[2, 3], type=int)
    parser.add_argument('--num_heads', help='Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.', type=int, nargs='*')
    parser.add_argument('--hidden_dims', help='Output dimension for each hidden layer. Input is a list that matches --num_hidden_layers in length.', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--dropout', help='Dropout for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.', type=float, nargs='*')
    parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings).', default=50, type=int)
    parser.add_argument('--loss', help='Loss function (KL or MMD).', choices=['kl', 'mmd'], default='kl')
    parser.add_argument('--lr', help='Learning rate for Adam.', default=0.0001, type=float)
    parser.add_argument('--epochs', help='Number of training epochs.', default=50, type=int)
    parser.add_argument('--val_split', help='Validation split e.g. 0.1.', default=0.0, type=float)
    parser.add_argument('--test_split', help='Test split e.g. 0.1.', default=0.0, type=float)
    parser.add_argument('--transpose_input', action='store_true', default=False, help='Specify if inputs should be transposed.')
    parser.add_argument('--use_linear_decoder', action='store_true', default=False, help='Turn on a neural network decoder, similar to traditional VAEs.')
    parser.add_argument('--decoder_nn_dim1', help='First hidden dimenson for the neural network decoder, if specified using --use_linear_decoder.', type=int)
    parser.add_argument('--name', help='Name used for the written output files.', type=str)
    parser.add_argument('--model_save_path', required=True, help='Path to save PyTorch model and output files. Will create the entire path if necessary.', type=str)
    parser.add_argument('--umap', action='store_true', default=False, help='Compute and save the 2D UMAP embeddings of the output node features.')
    parser.add_argument('--hdbscan', action='store_true', default=False, help='Compute and save different HDBSCAN clusterings.')

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
    assert args['test_split'] >= 0, 'Negative values for the test split are not allowed.'

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

    if (args['graph_file_path'] is not None) and (args['save_graph']):
        raise ValueError('Cannot use custom graph file when --save_graph is specified.')

    if (not args['graph_file_path']) and (not args['k']):
        raise ValueError('Graph generation enabled but --k not specified.')

    if (not args['graph_file_path']) and (not args['graph_type']):
        raise ValueError('Graph generation enabled but --graph_type not specified.')

    if (conv_type == 'GCN') and (num_heads is not None or dropout is not None):
        raise ValueError('GCN convolution not available with --num_heads or --dropout.')

    if (args['graph_type'] != 'KNN Faiss') and (args['faiss_gpu']):
        raise ValueError('Must use Faiss KNN if providing --faiss_gpu.')

    if (args['hvg'] and not args['khvg']) or (args['khvg'] and not args['hvg']):
        raise ValueError('--hvg and --khvg must be specified simultaneously.')

    if (args['graph_n_pcs']) and (args['graph_type'] != 'KNN Scanpy'):
        raise ValueError('--graph_n_pcs is valid only for the "KNN Scanpy" graph type.')

    if (args['hdbscan']) and (not args['umap']):
        raise ValueError('--hdbscan is valid only if --umap is specified.')

    if args['dropout']:
        assert (len(dropout) == num_hidden_layers + 2), 'Number of hidden dropout values must match number of hidden layers.'
    if args['num_heads']:
        assert (len(num_heads) == num_hidden_layers + 2), 'Number of hidden output dimensions must match number of hidden layers.'
    assert (len(hidden_dims) == num_hidden_layers), 'Number of hidden output dimensions must match number of hidden layers.'
    assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'Number of hidden layers must be 2 or 3.'

    if (args['decoder_nn_dim1']) and (not args['use_linear_decoder']):
        print()
        print(colored('WARNING: --decoder_nn_dim1 provided but --use_linear_decoder is not set. Ignoring --decoder_nn_dim1.\n', 'yellow'))

    model, optimizer, train_data, val_data, test_data = _setup(args, device=device)
    if torch.cuda.is_available():
        print(f'\nCUDA available, using {torch.cuda.get_device_name(device)}.')
    print('Neural model details: \n')
    print(model)
    print()

    print(f'Using {args["latent_dim"]} latent dimensions.')
    if args['use_linear_decoder']:
        print('Using linear feature decoder.')
    else:
        print('No feature decoder used.')
    if args['loss'] == 'kl':
        print('Using KL loss.')
    else:
        print('Using MMD loss.')

    print(f'Number of train edges {train_data.edge_index.shape[1]}.\n')

    if args['val_split']:
        print(f'Using validation split of {args["val_split"]}, number of validation edges: {val_data.pos_edge_label_index.shape[1]}.')

    if args['test_split']:
        print(f'Using test split of {args["test_split"]}, number of test edges: {test_data.pos_edge_label_index.shape[1]}.')

    # Train/val/test code
    for epoch in tqdm(range(1, args['epochs'] + 1)):
        epoch_loss, decoder_loss = _train(model, optimizer, train_data, args['loss'], device=device, use_decoder_loss=args['use_linear_decoder'], conv_type=args['graph_convolution'])
        if args['use_linear_decoder']:
            print('Epoch {:03d} -- Total epoch loss: {:.4f} -- NN decoder epoch loss: {:.4f}'.format(epoch, epoch_loss, decoder_loss))
        else:
            print('Epoch {:03d} -- Total epoch loss: {:.4f}'.format(epoch, epoch_loss))

        if args['val_split']:
            auroc, ap = _test(val_data)
            print('Validation AUROC {:.4f} -- AP {:.4f}.'.format(auroc, ap))

    if args['test_split']:
        auroc, ap = _test(test_data)
        print('Test AUROC {:.4f} -- AP {:.4f}.'.format(auroc, ap))

    # Save node embeddings
    model = model.eval()
    node_embeddings = []

    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device)
    if args['graph_convolution'] in ['GAT', 'GATv2']:
        z_nodes, attn_w = model.encode(x, edge_index)
    else:
        z_nodes = model.encode(x, edge_index)
    node_embeddings.append(z_nodes.cpu().detach().numpy())

    node_embeddings = np.array(node_embeddings)
    node_embeddings = node_embeddings.squeeze()

    Path(args['model_save_path']).mkdir(parents=True, exist_ok=True)

    if args['name']:
        filename = f'{args["name"]}_CellVGAE_node_embeddings.npy'
    else:
        filename = 'cellvgae_node_embeddings.npy'

    node_filepath = os.path.join(args['model_save_path'], filename)
    np.save(node_filepath, node_embeddings)

    # Save model        
    if args['name']:
        filename = f'{args["name"]}_CellVGAE_model.pt'
    else:
        filename = 'cellvgae_model.pt'

    model_filepath = os.path.join(args['model_save_path'], filename)
    torch.save(model.state_dict(), model_filepath)

    # Save attention weights
    if args['graph_convolution'] in ['GAT', 'GATv2']:
        for i in range(args['num_hidden_layers']):
            edge_index, attention_weights = attn_w[i]
            edge_index, attention_weights = edge_index.detach().cpu(), attention_weights.detach().cpu()

            edges_filepath = _get_filepath(args, number_or_name=i+1, edge_or_weights='edge_index')
            attn_w_filepath = _get_filepath(args, number_or_name=i+1, edge_or_weights='attention_weights')

            torch.save(edge_index, edges_filepath)
            torch.save(attention_weights, attn_w_filepath)
        
        edge_index, attention_weights = attn_w[-2]
        edge_index, attention_weights = edge_index.detach().cpu(), attention_weights.detach().cpu()
        mu_edges_filepath = _get_filepath(args, number_or_name='mu', edge_or_weights='edge_index')
        mu_attn_w_filepath = _get_filepath(args, number_or_name='mu', edge_or_weights='attention_weights')
        torch.save(edge_index, mu_edges_filepath)
        torch.save(attention_weights, mu_attn_w_filepath)

        edge_index, attention_weights = attn_w[-1]
        edge_index, attention_weights = edge_index.detach().cpu(), attention_weights.detach().cpu()
        sigma_edges_filepath = _get_filepath(args, number_or_name='sigma', edge_or_weights='edge_index')
        sigma_attn_w_filepath = _get_filepath(args, number_or_name='sigma', edge_or_weights='attention_weights')
        torch.save(edge_index, sigma_edges_filepath)
        torch.save(attention_weights, sigma_attn_w_filepath)

    # UMAP
    if args['umap']:
        filename = args['name'] + '_' if args['name'] else ''
        umap_output_path = os.path.join(args['model_save_path'], filename + '2D_UMAP_embeddings.npy')
        print(f'Computing UMAP representation and saving to {umap_output_path}...')
        umap_reducer = umap.UMAP()
        u = umap_reducer.fit_transform(node_embeddings)
        np.save(umap_output_path, u)

    # HDBSCAN
    if args['hdbscan']:
        hdbscan_save_dir = os.path.join(args['model_save_path'], 'hdbscan_clusters/')
        print(f'Computing HDBSCAN clusterings and saving to {hdbscan_save_dir}...')
        cl_sizes = [10, 25, 50, 100]
        min_samples = [5, 10, 25, 50]

        hdbscan_dict = {}
        for cl_size in cl_sizes:
            for min_sample in min_samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size, min_samples=min_sample)
                clusterer.fit(u)
                hdbscan_dict[(cl_size, min_sample)] = clusterer.labels_
        Path(hdbscan_save_dir).mkdir(parents=True, exist_ok=True)

        for k, clusters in hdbscan_dict.items():
            cl_size, min_sample = k
            np.save(os.path.join(hdbscan_save_dir, f'hdbscan-clusters-min_cluster_size={cl_size}-min_samples={min_sample}.npy'), clusters)

    
    # Save plots
    if args['umap']:
        filename = args['name'] + '_' if args['name'] else ''
        umap_save_path = os.path.join(args['model_save_path'], filename + 'UMAP_plot.pdf')
        print(f'Saving UMAP plot to {umap_save_path}...')

        c1 = '7dba84-81171B-f4743b-073b3a-fe4a49'.split('-')
        c2 = '473198-FFA552-FAA916-16DB65-CBEF43'.split('-')
        c3 = 'FA198B-35A7FF-C8553D-7A306C-216869'.split('-')
        c4 = 'A2D6F9-B88E8D-EA638C-30343F-6320EE'.split('-')
        c5 = 'CAD5CA-FFE5D4-BFD3C1-F7AF9D-FF5D73'.split('-')
        c6 = '3E517A-000000-A98743-611C35-BC8034'.split('-')
        c = c1 + c2 + c3 + c4 + c5 + c6
        c = ['#' + c_i for c_i in c]

        def style_axs(axs, labelpad=3):
            for ax in axs:
                ax.spines['left'].set_linewidth(0.25)
                ax.spines['top'].set_linewidth(0.25)
                ax.spines['bottom'].set_linewidth(0.25)
                ax.spines['right'].set_linewidth(0.25)
                ax.set_xlabel('UMAP_1', fontsize=14, labelpad=labelpad)
                ax.set_ylabel('UMAP_2', fontsize=14, labelpad=labelpad)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([], minor=True)
                ax.set_yticks([], minor=True)
                ax.xaxis.set_ticks_position('none') 
                ax.yaxis.set_ticks_position('none') 
                plt.xticks([])
                plt.yticks([])
                ax.axis('equal')

        fig = plt.figure(figsize=(12, 12))
        plt.scatter(x=u[:, 0], y=u[:, 1], s=8, linewidths=0)
        plt.gca().set_aspect('equal')
        ax = fig.get_children()[1]
        style_axs([ax])
        plt.xticks([])
        plt.yticks([])
        fig.tight_layout()
        plt.savefig(umap_save_path, dpi=300)

    if args['hdbscan']:
        filename = args['name'] + '_' if args['name'] else ''
        hdbscan_plot_save_path = os.path.join(args['model_save_path'], filename + 'UMAP_HDBSCAN_plots.pdf')
        print(f'Saving UMAP plots with HDBSCAN clusters to {hdbscan_plot_save_path}...')
        fig, axs = plt.subplots(4, 4, figsize=(18, 18))
        plt.tight_layout(w_pad=2, h_pad=1.5)

        for i in range(4):
            for j in range(4):
                ax = axs[i][j]
                clusters = hdbscan_dict[(cl_sizes[i], min_samples[j])]
                clusters_set = set(clusters)

                for attempt in range(10):
                    try:
                        colours = random.sample(c, len(clusters_set))   
                    except ValueError:
                        c = c + c
                    else:
                        break
                else:
                    print('Could not plot after 10 attempts due to insufficient number of colours.')
                    sys.exit(1)

                cluster_to_colour = dict(zip(clusters_set, colours))
                colours_to_plot = [cluster_to_colour[clstr] for clstr in clusters]
                
                sns.scatterplot(x=u[:, 0], y=u[:, 1], hue=clusters, palette=colours, s=8, linewidth=0.0, ax=ax)
                ax.legend(prop={'size': 3}, bbox_to_anchor=(1.05, 0.98), borderaxespad=-1.5, labelspacing=1, frameon=False, handletextpad=0.75)
                ax.set_title(f'min_cluster_size={cl_sizes[i]}, min_samples={min_samples[j]}', y=1.0, fontdict={'fontsize': 12})

        style_axs(axs.flat, labelpad=1)
        plt.xticks([])
        plt.yticks([])
        fig.tight_layout()
        plt.savefig(hdbscan_plot_save_path, dpi=300)

    print('Exiting...')
