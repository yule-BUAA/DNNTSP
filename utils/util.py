import json
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from utils.load_config import get_attribute


# convert data from cpu to gpu, accelerate the running speed
def convert_to_gpu(data):
    if get_attribute('cuda') != -1 and torch.cuda.is_available():
        data = data.to(get_attribute('cuda'))
    return data


def convert_all_data_to_gpu(*data):
    res = []
    for item in data:
        item = convert_to_gpu(item)
        res.append(item)
    return tuple(res)


def convert_train_truth_to_gpu(train_data, truth_data):
    train_data = [[convert_to_gpu(basket) for basket in baskets] for baskets in train_data]
    truth_data = convert_to_gpu(truth_data)
    return train_data, truth_data


# load parameters of model
def load_model(model_object, model_file_path):
    model_object.load_state_dict(torch.load(model_file_path))
    return model_object


# save parameters of the model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def get_truth_data(truth_data):
    """
    Args:
        truth_data: list, shape (baskets_num, items_num)
    Returns:
        turth: tensor, shape (baskets_num, items_total)
    """
    truth_list = []
    for basket in truth_data:
        one_hot_items = F.one_hot(basket, num_classes=get_attribute('items_total'))
        one_hot_basket, _ = torch.max(one_hot_items, dim=0)
        truth_list.append(one_hot_basket)
    truth = torch.stack(truth_list)

    return truth


# padding for all users
def pad_sequence(data_list):
    """
    :param data_list: shape (batch_users, baskets, item_embed_dim)
    :return:
    """
    length = [len(sq) for sq in data_list]
    # pad in time dimension
    data = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
    return data, length


def get_class_weights(data_path):
    with open(data_path, 'r') as file:
        data_dict = json.load(file)
    train_data = data_dict['train']
    item_frequency = torch.ones(get_attribute('items_total'))
    num_baskets = 0
    for user, baskets in train_data.items():
        for basket in baskets:
            num_baskets += 1
            for item in basket:
                item_frequency[item] += 1
    item_frequency /= num_baskets
    max_item_frequency = torch.ones(get_attribute('items_total')) * torch.max(item_frequency)

    weights = max_item_frequency / item_frequency

    weights = weights / torch.max(weights)

    return weights
