import os
import json
import itertools

abs_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(abs_path) as file:
    config = json.load(file)


def get_attribute(name, default_value=None):
    """
    get configs
    :param name:
    :param default_value:
    :return:
    """
    try:
        return config[name]
    except KeyError:
        return default_value


def get_items_total(data_path):
    items_set = set()
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
        for key in ["train", "validate", "test"]:
            data = data_dict[key]
            for user in data:
                items_set = items_set.union(set(itertools.chain.from_iterable(data[user])))
    return len(items_set)


config['data_path'] = \
    f"{os.path.dirname(os.path.dirname(__file__))}/data/{get_attribute('data')}/{get_attribute('data')}.json"
config['items_total'] = get_items_total(get_attribute('data_path'))
