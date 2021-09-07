import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from functools import partial


class CellVGAE_Encoder(nn.Module):
    def __init__(self, num_hidden_layers, num_heads, in_channels, hidden_dims, latent_dim, dropout, concat, v2=False):
        super(CellVGAE_Encoder, self).__init__()
        assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'The number of hidden layers must be 2 or 3.'
        assert (num_hidden_layers == len(hidden_dims) and num_hidden_layers == len(dropout)
                ), 'The number of hidden layers must match the number of hidden output dimensions and the number of hidden dropout values.'

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.v2 = v2

        # self.conv = partial(GATv2Conv, shared_weights=True) if self.v2 else GATConv
        self.conv = GATv2Conv if self.v2 else GATConv

        self.hidden_layer1 = self.conv(
            in_channels=in_channels, out_channels=hidden_dims[0],
            heads=self.num_heads['first'],
            dropout=dropout[0],
            concat=concat['first'])
        in_dim2 = hidden_dims[0] * self.num_heads['first'] if concat['first'] else hidden_dims[0]

        self.hidden_layer2 = self.conv(
            in_channels=in_dim2, out_channels=hidden_dims[1],
            heads=self.num_heads['second'],
            dropout=dropout[1],
            concat=concat['second'])

        self.hidden_layer3 = None
        if num_hidden_layers == 3:
            in_dim3 = hidden_dims[1] * self.num_heads['second'] if concat['second'] else hidden_dims[1]
            self.hidden_layer3 = self.conv(
                in_channels=in_dim3, out_channels=hidden_dims[2],
                heads=self.num_heads['third'],
                dropout=dropout[2],
                concat=concat['third'])

        if num_hidden_layers == 2:
            in_dim_final = hidden_dims[-1] * self.num_heads['second'] if concat['second'] else hidden_dims[-1]
        elif num_hidden_layers == 3:
            in_dim_final = hidden_dims[-1] * self.num_heads['third'] if concat['third'] else hidden_dims[-1]

        self.conv_mean = self.conv(in_channels=in_dim_final, out_channels=latent_dim,
                                   heads=self.num_heads['mean'], concat=False, dropout=0.2)
        self.conv_log_std = self.conv(in_channels=in_dim_final, out_channels=latent_dim,
                                      heads=self.num_heads['std'], concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        hidden_out1, attn_w_1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)
        hidden_out1 = F.relu(hidden_out1)
        hidden_out2, attn_w_2 = self.hidden_layer2(hidden_out1, edge_index, return_attention_weights=True)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        hidden_out3, attn_w_3 = None, None
        if self.hidden_layer3:
            hidden_out3, attn_w_3 = self.hidden_layer3(hidden_out2, edge_index, return_attention_weights=True)
        last_out = hidden_out3 if self.hidden_layer3 else hidden_out2
        z_mean, attn_w_mean = self.conv_mean(last_out, edge_index, return_attention_weights=True)
        z_log_std, attn_w_log_std = self.conv_log_std(last_out, edge_index, return_attention_weights=True)

        if self.hidden_layer3:
            return z_mean, z_log_std, (attn_w_1, attn_w_2, attn_w_3, attn_w_mean, attn_w_log_std)
        return z_mean, z_log_std, (attn_w_1, attn_w_2, attn_w_mean, attn_w_log_std)


class CellVGAE_GCNEncoder(nn.Module):
    def __init__(self, num_hidden_layers, in_channels, hidden_dims, latent_dim):
        super(CellVGAE_GCNEncoder, self).__init__()
        assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'The number of hidden layers must be 2 or 3.'
        assert (num_hidden_layers == len(hidden_dims)), 'The number of hidden layers must match the number of hidden output dimensions and the number of hidden dropout values.'

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        self.hidden_layer1 = GCNConv(in_channels=in_channels, out_channels=hidden_dims[0])
        self.hidden_layer2 = GCNConv(in_channels=hidden_dims[0], out_channels=hidden_dims[1])
        self.hidden_layer3 = None
        if num_hidden_layers == 3:
            self.hidden_layer3 = GCNConv(in_channels=hidden_dims[1], out_channels=hidden_dims[2])
        self.conv_mean = GCNConv(in_channels=hidden_dims[-1], out_channels=latent_dim)
        self.conv_log_std = GCNConv(in_channels=hidden_dims[-1], out_channels=latent_dim)

    def forward(self, x, edge_index):
        hidden_out1 = self.hidden_layer1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        hidden_out2 = self.hidden_layer2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        hidden_out3 = None, None
        if self.hidden_layer3:
            hidden_out3 = self.hidden_layer3(hidden_out2, edge_index)
        last_out = hidden_out3 if self.hidden_layer3 else hidden_out2
        z_mean = self.conv_mean(last_out, edge_index)
        z_log_std = self.conv_log_std(last_out, edge_index)

        if self.hidden_layer3:
            return z_mean, z_log_std
        return z_mean, z_log_std