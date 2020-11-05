import torch
import torch.nn as nn


class aggregate_nodes_temporal_feature(nn.Module):

    def __init__(self, item_embed_dim):
        """
        :param item_embed_dim: the dimension of input features
        """
        super(aggregate_nodes_temporal_feature, self).__init__()

        self.Wq = nn.Linear(item_embed_dim, 1, bias=False)

    def forward(self, graph, lengths, nodes_output):
        """
        :param graph: batched graphs, with the total number of nodes is `node_num`,
                        including `batch_size` disconnected subgraphs
        :param lengths: tensor, (batch_size, )
        :param nodes_output: the output of self-attention model in time dimension, (n_1+n_2+..., T_max, F)
        :return: aggregated_features, (n_1+n_2+..., F)
        """
        nums_nodes, id = graph.batch_num_nodes(), 0
        aggregated_features = []
        for num_nodes, length in zip(nums_nodes, lengths):
            # get each user's length, tensor, shape, (user_nodes, user_length, item_embed_dim)
            output_node_features = nodes_output[id:id + num_nodes, :length, :]
            # weights for each timestamp, tensor, shape, (user_nodes, 1, user_length)
            # (user_nodes, user_length, 1) transpose to -> (user_nodes, 1, user_length)
            weights = self.Wq(output_node_features).transpose(1, 2)
            # (user_nodes, 1, user_length) matmul (user_nodes, user_length, item_embed_dim)
            # -> (user_nodes, 1, item_embed_dim) squeeze to (user_nodes, item_embed_dim)
            # aggregated_feature, tensor, shape, (user_nodes, item_embed_dim)
            aggregated_feature = weights.matmul(output_node_features).squeeze(dim=1)
            aggregated_features.append(aggregated_feature)
            id += num_nodes
        # (n_1+n_2+..., item_embed_dim)
        aggregated_features = torch.cat(aggregated_features, dim=0)
        return aggregated_features
