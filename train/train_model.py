import datetime
import json

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.util import save_model, convert_to_gpu, convert_graph_to_gpu, convert_all_data_to_gpu
from utils.metric import get_metric

from tqdm import tqdm
import warnings
import numpy as np

def train_model(model: nn.Module,
                train_data_loader: DataLoader,
                valid_data_loader: DataLoader,
                loss_func,
                epochs,
                optimizer,
                model_folder,
                tensorboard_folder):
    """
    Args:
        model: nn.Module
        train_data_loader: DataLoader
        valid_data_loader: DataLoader
        loss_func: nn.Module
        epochs: int
        optimizer: Optimizer
        model_folder: str
        tensorboard_folder: str
    """
    warnings.filterwarnings('ignore')

    print(model)
    print(optimizer)

    writer = SummaryWriter(tensorboard_folder)
    writer.add_text('Welcome', 'Welcome to tensorboard!')

    model = convert_to_gpu(model)
    model.train()
    loss_func = convert_to_gpu(loss_func)

    start_time = datetime.datetime.now()

    validate_max_ndcg = 0
    name_list = ["train", "validate"]

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(epochs):
        loss_dict, metric_dict = {name: 0.0 for name in name_list}, {name: dict() for name in name_list}
        data_loader_dic = {"train": train_data_loader, "validate": valid_data_loader}

        for name in name_list:
            # training
            if name == "train":
                model.train()
            # validate
            else:
                model.eval()

            y_true = []
            y_pred = []
            total_loss = 0.0
            tqdm_loader = tqdm(data_loader_dic[name])
            for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(tqdm_loader):
                g = convert_graph_to_gpu(g) 
                nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                    convert_all_data_to_gpu(nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

                with torch.set_grad_enabled(name == 'train'):
                    # (B, N)
                    output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
                    loss = loss_func(output, truth_data.float())
                    total_loss += loss.cpu().data.numpy()
                    if name == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    y_pred.append(output.detach().cpu())
                    y_true.append(truth_data.detach().cpu())
                    tqdm_loader.set_description(f'{name} epoch: {epoch}, {name} loss: {total_loss / (step + 1)}')

            loss_dict[name] = total_loss / (step + 1)
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)

            print(f'{name} metric ...')
            scores = get_metric(y_true=y_true, y_pred=y_pred)
            scores = sorted(scores.items(), key=lambda item: item[0], reverse=False)
            scores = {item[0]: item[1] for item in scores}
            print(json.dumps(scores, indent=4))
            metric_dict[name] = scores

            # save best model
            if name == "validate":
                validate_ndcg_list = []
                for key in metric_dict["validate"]:
                    if key.startswith("ndcg_"):
                        validate_ndcg_list.append(metric_dict["validate"][key])
                validate_ndcg = np.mean(validate_ndcg_list)
                if validate_ndcg > validate_max_ndcg:
                    validate_max_ndcg = validate_ndcg
                    model_path = f"{model_folder}/model_epoch_{epoch}.pkl"
                    save_model(model, model_path)
                    print(f"model save as {model_path}")

        scheduler.step(loss_dict['train'])

        writer.add_scalars('Loss', {
            f'{name} loss': loss_dict[name] for name in name_list}, global_step=epoch)

        for metric in metric_dict['train'].keys():
            for name in name_list:
                writer.add_scalars(f'{name} {metric}', {f'{metric}': metric_dict[name][metric]}, global_step=epoch)

    end_time = datetime.datetime.now()
    print("cost %d seconds" % (end_time - start_time).seconds)
