# Predicting Temporal Sets with Deep Neural Networks (DNNTSP)

DNNTSP is a general neural network architecture that could make prediction on temporal sets.

Please refer to our KDD 2020 paper [“**Predicting Temporal Sets with Deep Neural Networks**”](https://dl.acm.org/doi/pdf/10.1145/3394486.3403152) for more details.

## Project Structure
The descriptions of principal files in this project are explained as follows:
- ./model/
    - `weighted_graph_conv.py`: codes for the Element Relationship Learning component (i.e. weighted GCN on dynamic graphs)
    - `masked_self_attention.py` and `aggregate_nodes_temporal_feature.py`: codes for the Attention-based Temporal Dependency Learning component (i.e. masked self-attention and weighted aggregation of temporal information)
    - `global_gated_update.py`: codes for the Gated Information Fusing component (i.e. gated updating mechanism)
- ./train/
  - `train_model.py` and `train_main.py`: codes for training models
- ./test/
  - `testing_model.py`: codes for evaluating models
- ./utils/: containing useful files that are required in the project (e.g. data loader, metrics calculation, loss function, configurations) 
- ./data/: processed datasets are under in this folder. Original datasets could be downloaded as follows:
  - [TaFeng](https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset)  
  - [DC](https://www.dunnhumby.com/careers/engineering/sourcefiles)  
  - [TaoBao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)  
  - [TMS](https://math.stackexchange.com)
- ./save_model_folder/ and ./runs/: folders to save models and outputs of tensorboardX respectively
- ./results/: folders to save the evaluation metrics for models. 

## Parameter Settings
Please refer to our paper for more details of parameter settings. 
Hyperparameters could be found in ./utils/config.json and you can adjust them when running the model.

## How to use
- Training: after setting the parameters, run ```train_main.py``` file to train models. 
- Testing: figure out the path of the specific saved model (i.e. variable ```model_path``` in ./test/testing_model.py) and then run ```testing_model.py``` file to evaluate models.

Principal environmental dependencies as follows:
- [PyTorch 1.5.0](https://pytorch.org/)
- [dgl 0.5.2](https://www.dgl.ai/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [sklearn](https://scikit-learn.org/stable/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)

## Citation
Please consider citing the following paper when using our code.

```bibtex
@inproceedings{DBLP:conf/kdd/YuSDL0L20,
  author    = {Le Yu and
               Leilei Sun and
               Bowen Du and
               Chuanren Liu and
               Hui Xiong and
               Weifeng Lv},
  title     = {Predicting Temporal Sets with Deep Neural Networks},
  booktitle = {{KDD} '20: The 26th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Virtual Event, CA, USA, August 23-27, 2020},
  pages     = {1083--1091},
  year      = {2020}
}
```
