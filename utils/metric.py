import numpy as np
import torch
from tqdm import tqdm
from utils.util import convert_all_data_to_gpu
import datetime


def recall_score(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1)
    # end_time = datetime.datetime.now()
    # print("recall_score cost %d seconds" % (end_time - start_time).seconds)
    return (tp.float() / t.float()).mean().item()


def dcg(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):

    Returns:

    """
    _, predict_indices = y_pred.topk(k=top_k)
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg_score(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):
    Returns:

    """
    # start_time = datetime.datetime.now()
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    # end_time = datetime.datetime.now()
    # print("ndcg cost %d seconds" % (end_time - start_time).seconds)
    return (dcg_score / idcg_score).mean().item()


def PHR(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    hit_num = torch.mul(predict, truth).sum(dim=1).nonzero().shape[0]
    # end_time = datetime.datetime.now()
    # print("PHR cost %d seconds" % (end_time - start_time).seconds)
    return hit_num / truth.shape[0]


def get_metric(y_true, y_pred):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
        Returns:
            scores: dict
    """
    result = {}
    for top_k in [10, 20, 30, 40]:
        result.update({
            f'recall_{top_k}': recall_score(y_true, y_pred, top_k=top_k),
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k)
        })
    return result


def evaluate(model, data_loader):
    """
    Args:
        model: nn.Module
        data_loader: DataLoader
    Returns:
        scores: dict
    """
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                tqdm(data_loader)):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            predict_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

            # predict_data shape (batch_size, baskets_num, items_total)
            # truth_data shape (batch_size, baskets_num, items_total)
            y_pred.append(predict_data.detach().cpu())
            y_true.append(truth_data.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        return get_metric(y_true=y_true, y_pred=y_pred)
