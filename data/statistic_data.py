import json
from tqdm import tqdm
import itertools
import sys


# statistics of preprocessed datasets
# total amount of users, items, sets
# average set size (total elements in sets / total sets )
# average sets per person (total sets / total users)
def statistic_data(data_path):
    totalUsers = 0
    totalSets = 0
    itemList = []
    with open(data_path, 'r') as file:
        # train validate test
        dataDicTotal = json.load(file)
        for key in tqdm(dataDicTotal):
            dataDic = dataDicTotal[key]
            totalUsers += len(dataDic)
            for userKey in tqdm(dataDic):
                user = dataDic[userKey]
                totalSets += len(user)
                itemList.extend(list(itertools.chain.from_iterable(user)))

    totalItems = len(set(itemList))
    totalAppearItems = len(itemList)

    averageSetSize = totalAppearItems / totalSets
    averageSetPerPerson = totalSets / totalUsers
    statisticDic = {
                    "totalSets": totalSets,
                    "totalUsers": totalUsers,
                    "totalItems": totalItems,
                    "averageSetSize": averageSetSize,
                    "averageSetPerPerson": averageSetPerPerson
                    }
    return statisticDic


if __name__ == "__main__":
    statisticDic = statistic_data(data_path='../data/TaFeng/TaFeng.json')
    print("TaFeng.json:")
    print(json.dumps(statisticDic, indent=4))
    statisticDic = statistic_data(data_path='DC/DC.json')
    print("DC.json:")
    print(json.dumps(statisticDic, indent=4))
    statisticDic = statistic_data(data_path='../data/TaoBao/TaoBao.json')
    print("TaoBao.json:")
    print(json.dumps(statisticDic, indent=4))
    statisticDic = statistic_data(data_path='TMS/TMS.json')
    print("TMS.json:")
    print(json.dumps(statisticDic, indent=4))
    sys.exit()
