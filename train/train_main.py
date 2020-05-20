import torch
import torch.nn as nn
import sys
import os
import shutil

from utils.load_config import get_attribute
from utils.data_container import get_data_loader
from utils.loss import BPRLoss, WeightMSELoss
from utils.util import get_class_weights

from model.temporal_set_prediction import temporal_set_prediction
from train.train_model import train_model


def create_model():

    print(f"{get_attribute('data')}/{get_attribute('save_model_folder')}")

    model = temporal_set_prediction(items_total=get_attribute('items_total'),
                                    item_embedding_dim=get_attribute('item_embed_dim'))

    return model


def create_loss(loss_type):
    if loss_type == 'bpr_loss':
        loss_func = BPRLoss()
    elif loss_type == 'mse_loss':
        loss_func = WeightMSELoss()
    elif loss_type == 'weight_mse_loss':
        loss_func = WeightMSELoss(weights=get_class_weights(get_attribute('data_path')))
    elif loss_type == "multi_label_soft_loss":
        loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
    else:
        raise ValueError("Unknown loss function.")
    return loss_func


def train():
    model = create_model()
    # 创建data_loader
    train_data_loader = get_data_loader(data_path=get_attribute('data_path'),
                                        data_type='train',
                                        batch_size=get_attribute('batch_size'),
                                        item_embedding_matrix=model.item_embedding)
    valid_data_loader = get_data_loader(data_path=get_attribute('data_path'),
                                        data_type='validate',
                                        batch_size=get_attribute('batch_size'),
                                        item_embedding_matrix=model.item_embedding)
    loss_func = create_loss(loss_type=get_attribute('loss_function'))

    # 训练
    model_folder = f"../save_model_folder/{get_attribute('data')}/{get_attribute('save_model_folder')}"
    tensorboard_folder = f"../runs/{get_attribute('data')}/{get_attribute('save_model_folder')}"

    shutil.rmtree(model_folder, ignore_errors=True)
    os.makedirs(model_folder, exist_ok=True)
    shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder, exist_ok=True)

    if get_attribute("optim") == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=get_attribute("learning_rate"),
                                     weight_decay=get_attribute("weight_decay"))
    elif get_attribute("optim") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=get_attribute("learning_rate"),
                                    momentum=0.9)
    else:
        raise NotImplementedError()

    train_model(model=model,
                train_data_loader=train_data_loader,
                valid_data_loader=valid_data_loader,
                loss_func=loss_func,
                epochs=get_attribute('epochs'),
                optimizer=optimizer,
                model_folder=model_folder,
                tensorboard_folder=tensorboard_folder)


if __name__ == '__main__':
    train()
    sys.exit()
