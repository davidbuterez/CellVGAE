import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

class CellVGAE_Encoder(nn.Module):
    def __init__(self, num_hidden_layers, num_heads, in_channels, hidden_dims, latent_dim, dropout):
        super(CellVGAE_Encoder, self).__init__()
        assert (num_hidden_layers == 2 or num_hidden_layers == 3), 'The number of hidden layers must be 2 or 3.'
        assert (num_hidden_layers == len(hidden_dims) and num_hidden_layers == len(dropout)), 'The number of hidden layers must match the number of hidden output dimensions and the number of hidden dropout values.'

        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers

        self.hidden_layer1 = GATConv(in_channels=in_channels, out_channels=hidden_dims[0], heads=self.num_heads, dropout=dropout[0])
        self.hidden_layer2 = GATConv(in_channels=hidden_dims[0] * self.num_heads, out_channels=hidden_dims[1], heads=self.num_heads, dropout=dropout[1])
        self.hidden_layer3 = None
        if num_hidden_layers == 3:
            self.hidden_layer3 = GATConv(in_channels=hidden_dims[1] * self.num_heads, out_channels=hidden_dims[2], heads=self.num_heads, dropout=dropout[2])
        self.conv_mean = GATConv(in_channels=hidden_dims[-1] * self.num_heads, out_channels=latent_dim, heads=self.num_heads, concat=False, dropout=0.2)
        self.conv_log_std = GATConv(in_channels=hidden_dims[-1] * self.num_heads, out_channels=latent_dim, heads=self.num_heads, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        hidden_out1, attn_w_1 = self.hidden_layer1(x, edge_index, return_attention_weights=True)
        hidden_out1 = F.relu(hidden_out1)
        hidden_out2, attn_w_2 = self.hidden_layer2(hidden_out1, edge_index, return_attention_weights=True)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        hidden_out3, attn_w_3 = None, None
        if self.hidden_layer3:
            hidden_out3, attn_w_3 = self.hidden_layer3(hidden_out2, edge_index, return_attention_weights=True)
        last_out = hidden_out3 if hidden_out3 else hidden_out2
        z_mean, attn_w_mean = self.conv_mean(last_out, edge_index, return_attention_weights=True)
        z_log_std, attn_w_log_std = self.conv_log_std(last_out, edge_index, return_attention_weights=True)

        if self.hidden_layer3:
            return z_mean, z_log_std, (attn_w_1, attn_w_2, attn_w_3, attn_w_mean, attn_w_log_std)
        return z_mean, z_log_std, (attn_w_1, attn_w_2, attn_w_mean, attn_w_log_std)