from typing import List
import torch.nn as nn
import torch
import dgl
import dgl.function as fn


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
        # one way: use dgl.function is faster and less requirement of GPU memory
        graph.edata['e'] = edge_weights.t().unsqueeze(dim=-1)  # (E, T, 1)
        graph.update_all(fn.u_mul_e('n', 'e', 'msg'), fn.sum('msg', 'h'))

        # another way: use user defined function, needs more GPU memory
        # graph.edata['e'] = edge_weights.t()
        # graph.update_all(self.gcn_message, self.gcn_reduce)

        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        return output

    @staticmethod
    def gcn_message(edges):
        if edges.src['n'].dim() == 2:
            # (E, T, 1) (E, 1, F),  matmul ->  matmul (E, T, F)
            return {'msg': torch.matmul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'].unsqueeze(dim=1))}

        elif edges.src['n'].dim() == 3:
            # (E, T, 1) (E, T, F),  mul -> (E, T, F)
            return {'msg': torch.mul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'])}

        else:
            raise ValueError(f"wrong shape for edges.src['n'], the length of shape is {edges.src['n'].dim()}")

    @staticmethod
    def gcn_reduce(nodes):
        # propagate, the first dimension is nodes num in a batch
        # h, tensor, shape, (N, neighbors, T, F) -> (N, T, F)
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
