import json
import sys
import os

from utils.metric import evaluate
from utils.data_container import get_data_loader
from utils.load_config import get_attribute
from utils.util import convert_to_gpu
from train.train_main import create_model
from utils.util import load_model


if __name__ == '__main__':
    model_path = f"../save_model_folder/{get_attribute('data')}/{get_attribute('save_model_folder')}" \
        f"/model_epoch_15.pkl"
    print(f'model path -> {model_path}')

    model = create_model()
    model = load_model(model, model_path)
    model = convert_to_gpu(model)
    print(model)

    test_data_loader = get_data_loader(data_path=get_attribute('data_path'),
                                       data_type='test',
                                       batch_size=get_attribute('batch_size'),
                                       item_embedding_matrix=model.item_embedding)

    print('===== Test predict result =====')
    scores = evaluate(model,
                      test_data_loader)

    scores = sorted(scores.items(), key=lambda item: item[0], reverse=False)
    scores = {item[0]: item[1] for item in scores}

    scores_str = json.dumps(scores, indent=4)
    print(f'scores -> {scores_str}')

    model_folder = f"../results/{get_attribute('data')}"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    save_path = f"{model_folder}/{get_attribute('save_model_folder')}_result.json"
    print(f'save path is {save_path}')

    with open(save_path, 'w') as file:
        file.write(scores_str)
    sys.exit()
