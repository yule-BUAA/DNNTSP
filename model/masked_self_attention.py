import torch
import torch.nn as nn
import numpy as np


class masked_self_attention(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads=4, attention_aggregate="concat"):
        super(masked_self_attention, self).__init__()
        # aggregate multi-heads by concatenation or mean
        self.attention_aggregate = attention_aggregate

        # the dimension of each head is dq // n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_heads = n_heads

        if attention_aggregate == "concat":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim // n_heads
        elif attention_aggregate == "mean":
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim
        else:
            raise ValueError(f"wrong value for aggregate {attention_aggregate}")

        self.Wq = nn.Linear(input_dim, n_heads * self.dq, bias=False)
        self.Wk = nn.Linear(input_dim, n_heads * self.dk, bias=False)
        self.Wv = nn.Linear(input_dim, n_heads * self.dv, bias=False)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: tensor, shape (nodes_num, T_max, features_num)
        Returns:
            output: tensor, shape (nodes_num, T_max, output_dim = features_num)
        """
        seq_length = input_tensor.shape[1]
        # tensor, shape (nodes_num, T_max, n_heads * dim_per_head)
        Q = self.Wq(input_tensor)
        K = self.Wk(input_tensor)
        V = self.Wv(input_tensor)
        # multi_head attention
        # Q, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        Q = Q.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dq).transpose(1, 2)
        # K after transpose, tensor, shape (nodes_num, n_heads, dim_per_head, T_max)
        K = K.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dk).permute(0, 2, 3, 1)
        # V, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        V = V.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dv).transpose(1, 2)

        # scaled attention_score, tensor, shape (nodes_num, n_heads, T_max, T_max)
        attention_score = Q.matmul(K) / np.sqrt(self.per_head_dim)

        # attention_mask, tensor, shape -> (T_max, T_max)  -inf in the top and right
        attention_mask = torch.zeros(seq_length, seq_length).masked_fill(
            torch.tril(torch.ones(seq_length, seq_length)) == 0, -np.inf).to(input_tensor.device)
        # attention_mask will be broadcast to (nodes_num, n_heads, T_max, T_max)
        attention_score = attention_score + attention_mask
        # (nodes_num, n_heads, T_max, T_max)
        attention_score = torch.softmax(attention_score, dim=-1)

        # multi_result, tensor, shape (nodes_num, n_heads, T_max, dim_per_head)
        multi_head_result = attention_score.matmul(V)
        if self.attention_aggregate == "concat":
            # multi_result, tensor, shape (nodes_num, T_max, n_heads * dim_per_head = output_dim)
            # concat multi-head attention results
            output = multi_head_result.transpose(1, 2).reshape(input_tensor.shape[0],
                                                               seq_length, self.n_heads * self.per_head_dim)
        elif self.attention_aggregate == "mean":
            # multi_result, tensor, shape (nodes_num, T_max, dim_per_head = output_dim)
            # mean multi-head attention results
            output = multi_head_result.transpose(1, 2).mean(dim=2)
        else:
            raise ValueError(f"wrong value for aggregate {self.attention_aggregate}")

        return output
