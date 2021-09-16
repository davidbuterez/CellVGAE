import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GAE, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

EPS = 1e-15
MAX_LOGVAR = 10

# Based on VGAE class in PyTorch Geometric


class CellVGAE(GAE):
    def __init__(self, encoder, decoder_nn_dim1=None, decoder=None, gcn_or_gat='GAT'):
        super(CellVGAE, self).__init__(encoder, decoder)
        assert gcn_or_gat in ['GAT', 'GATv2', 'GCN'], 'Convolution must be "GCN", "GAT", or "GATv2.'
        self.decoder = InnerProductDecoder() if decoder is None else decoder

        self.decoder_nn_dim1 = decoder_nn_dim1
        self.decoder_nn_dim2 = self.encoder.in_channels
        self.gcn_or_gat = gcn_or_gat

        if decoder_nn_dim1:
            self.decoder_nn = Sequential(
                Linear(in_features=self.encoder.latent_dim, out_features=self.decoder_nn_dim1),
                BatchNorm1d(self.decoder_nn_dim1),
                ReLU(),
                Dropout(0.4),
                Linear(in_features=self.decoder_nn_dim1, out_features=self.decoder_nn_dim2),
            )

    def reparametrize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        if self.gcn_or_gat in ['GAT', 'GATv2']:
            self.__mu__, self.__logvar__, attn_w = self.encoder(*args, **kwargs)
        else:
            self.__mu__, self.__logvar__ = self.encoder(*args, **kwargs)

        self.__logvar__ = self.__logvar__.clamp(max=MAX_LOGVAR)
        z = self.reparametrize(self.__mu__, self.__logvar__)

        if self.gcn_or_gat in ['GAT', 'GATv2']:
            return z, attn_w
        else:
            return z

    def kl_loss(self, mu=None, logvar=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
