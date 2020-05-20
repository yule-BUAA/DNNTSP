from typing import List
import torch.nn as nn
import torch
import dgl
from dgl import init as g_init


class weighted_graph_conv(nn.Module):
    """
        Apply graph convolution over an input signal.
    """
    def __init__(self, in_features: int, out_features: int):
        super(weighted_graph_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, graph, node_features, edge_weights):
        r"""Compute weighted graph convolution.
        -----
        Input:
        graph : DGLGraph, batched graph.
        node_features : torch.Tensor, input features for nodes (n_1+n_2+..., in_features) or (n_1+n_2+..., T, in_features)
        edge_weights : torch.Tensor, input weights for edges  (T, n_1^2+n_2^2+..., n^2)

        Output:
        shape: (N, T, out_features)
        """
        graph = graph.local_var()
        # multi W first to project the features, with bias
        # (N, F) / (N, T, F)
        graph.ndata['n'] = node_features
        # edge_weights, shape (T, N^2)
        graph.edata['e'] = edge_weights.t()
        graph.send(graph.edges(), self.gcn_message)
        graph.recv(graph.nodes(), self.gcn_reduce)
        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        return output

    @staticmethod
    def gcn_message(edges: dgl.EdgeBatch):
        if edges.src['n'].dim() == 2:
            # (B, T, 1) (B, 1, F),  matmul ->  matmul (B, T, F)
            return {'msg': torch.matmul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'].unsqueeze(dim=1))}

        elif edges.src['n'].dim() == 3:
            # (B, T, 1) (B, T, F),  mul -> (B, T, F)
            return {'msg': torch.mul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'])}

        else:
            raise ValueError(f"wrong shape for edges.src['n'], the length of shape is {edges.src['n'].dim()}")

    @staticmethod
    def gcn_reduce(nodes: dgl.NodeBatch):
        # propagate, the first dimension is batch
        # h, tensor, shape, (B, N, T, F) -> (B, T, F)
        return {'h': torch.sum(nodes.mailbox['msg'], 1)}


class weighted_GCN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(weighted_GCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # layers for hidden_size
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(weighted_graph_conv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        # output layer
        gcns.append(weighted_graph_conv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, edges_weight: torch.Tensor):
        """
        :param graph: a graph
        :param node_features: shape (n_1+n_2+..., n_features)
               edges_weight: shape (T, n_1^2+n_2^2+...)
        :return:
        """
        graph.set_n_initializer(g_init.zero_initializer)
        graph.set_e_initializer(g_init.zero_initializer)
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            # (n_1+n_2+..., T, features)
            h = gcn(graph, h, edges_weight)
            h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h


class stacked_weighted_GCN_blocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(stacked_weighted_GCN_blocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, nodes_feature, edge_weights = input
        h = nodes_feature
        for module in self:
            h = module(g, h, edge_weights)
        return h
