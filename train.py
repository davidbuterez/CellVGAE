# """Train CellVGAE

# Usage:
#   train.py --hvg_file <hvg> --graph_file <graph> --num_hidden_layers (2 | 3) [--num_heads=<heads>] [-hd <hd> ]... [--d <d>]... [--latent_dim=<latent_dim>] 

# Options:
#   -h --help     Show this screen.
#   --version     Show version.
# """

import argparse
import torch

import pandas as pd
import numpy as np

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import train_test_split_edges, to_undirected
from sklearn.preprocessing import MinMaxScaler

from models import CellVGAE, CellVGAE_Encoder
from models import mmd

parser = argparse.ArgumentParser(prog='train', description='Train CellVGAE')

parser.add_argument('--hvg_file', help='HVG file (log-normalised)')
parser.add_argument('--graph_file', help='Graph specified as an edge list (one per line, separated by whitespace)')
parser.add_argument('--num_hidden_layers', help='Number of hidden layers (must be 2 or 3)', default=2, type=int)
parser.add_argument('--num_heads', help='Number of attention heads', default=10, type=int)
parser.add_argument('--hidden_dims', help='Output dimension for each hidden layer (only 2 or 3 layers allowed)', type=int, nargs='*', default=[128, 128])
parser.add_argument('--dropout', help='Dropout for each hidden layer (only 2 or 3 layers allowed)', type=float, nargs='*', default=[0.2, 0.2])
parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings)', default=50, type=int)
parser.add_argument('--loss', help='Loss function (KL or MMD)', choices=['kl', 'mmd'], default='kl')
parser.add_argument('--lr', help='Learning rate for Adam', default=0.0001, type=float)
parser.add_argument('--batch_size', help='Batch size', default=64, type=int)
parser.add_argument('--epochs', help='Number of training epochs', default=250, type=int)
parser.add_argument('--val_split', help='Validation split e.g. 0.1', default=0.0, type=float)
parser.add_argument('--node_out', help='Output file name and path for the computed node embeddings (saved in numpy .npy format)', default='node_embs.npy')
parser.add_argument('--save_trained_model', help='Path to save PyTorch model', default='model.pt')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, train_loader, loss):
    model = model.train()

    epoch_loss = 0.0
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

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

def test(x, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z, _ = model.encode(x.to(torch.float), train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

def setup(args):
    df = pd.read_csv(args['hvg_file'])
    df.drop(columns=['Unnamed: 0'], inplace=True)

    X_list = []
    for i in range(df.shape[1]):
        X_list.append(df.iloc[:, i].values)

    X_features = np.stack(X_list)
    edgelist = []

    with open(args['graph_file'], 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    edge_index = np.array(edgelist).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes=X_features.shape[0])

    data_list = []

    scaler = MinMaxScaler()
    scaled_x = torch.from_numpy(scaler.fit_transform(X_features))

    data_obj = Data(edge_index=edge_index, x=scaled_x)
    data_obj.num_nodes = X_features.shape[0]

    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    
    # Can set validation ratio
    data = train_test_split_edges(data_obj, val_ratio=args['val_split'], test_ratio=0)
    x, train_pos_edge_index = data.x.to(torch.double).to(device), data.train_pos_edge_index.to(device)

    data_list.append(Data(edge_index=edge_index, x=scaled_x))

    num_features = data_obj.num_features
    train_loader = DataLoader([Data(edge_index=train_pos_edge_index, x=x)], batch_size=args['batch_size'])

    encoder = CellVGAE_Encoder.CellVGAE_Encoder(in_channels=num_features, num_hidden_layers=args['num_hidden_layers'], num_heads=args['num_heads'],
                hidden_dims=args['hidden_dims'], dropout=args['dropout'], latent_dim=args['latent_dim'])
    model = CellVGAE.CellVGAE(encoder=encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    model = model.to(device)

    return model, optimizer, train_loader, data_list

if __name__ == '__main__':
    args = vars(args)
    num_hidden_layers = args['num_hidden_layers']
    assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'Number of hidden layers must be 2 or 3.'

    hidden_dims = args['hidden_dims']
    assert (len(hidden_dims) == num_hidden_layers), 'Number of hidden output dimensions must match number of hidden layers.'

    dropout = args['dropout']
    assert (len(dropout) == num_hidden_layers), 'Number of hidden dropout values must match number of hidden layers.'

    model, optimizer, train_loader, data_list = setup(args)

    print(torch.cuda.get_device_name(device))

    for epoch in range(1, args['epochs'] + 1):
        train(model, optimizer, train_loader, args['loss'])
        print('Epoch: {:03d}'.format(epoch))

        # Uncomment if using validation
        #     auc, ap = test(x.to(torch.float), data.val_pos_edge_index, data.val_neg_edge_index)
        #     print('Epoch: {:03d} -- AUC: {:.4f} -- AP: {:.4f}'.format(epoch, auc, ap))


    model = model.eval()
    node_embeddings = []

    for graph_data in data_list:
        x, edge_index = graph_data.x.to(torch.float).to(device), graph_data.edge_index.to(torch.long).to(device)
        z_nodes, _ = model.encode(x, edge_index)
        node_embeddings.append(z_nodes.cpu().detach().numpy())

    node_embeddings = np.array(node_embeddings)
    node_embeddings = node_embeddings.squeeze()

    if args['node_out']:
        np.save(args['node_out'], node_embeddings)

    if args['save_trained_model']:
        torch.save(model.state_dict(), args['save_trained_model'])

    print('Exiting...')
