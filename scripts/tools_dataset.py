from datetime import datetime
from math import floor
from typing import Dict, List

import torch
from torch.utils.data.dataset import Dataset, TensorDataset

from tools_logging import logValue


def splitDateList(dateList: List[str]):
    split = getDataSplitSizes(len(dateList))
    return {
        'train': dateList[0:(split["train"]-1)],
        'test': dateList[split["train"]:(split["train"]+split["test"]-1)],
        'validation':  dateList[(split["train"]+split["test"]):]
    }


def getDataSplitSizes(total: int, verbose: bool = False, dateList: List[str] = None, trainPercentage=0.8, testPercentage=0.15) -> Dict[str, int]:
    train = floor(total*trainPercentage)
    test = floor(total*testPercentage)
    val = total - train - test
    print("Dataset got split: {} Training, {} Testing & {} Validation Sub-dataset of total {}".format(train, test, val, total))
    if verbose and dateList is not None:
        trainStart = 0
        trainEnd = train-1
        print("{} - {} Training dataset (Index {} - {} = #{})".format(
            dateList[trainStart], dateList[trainEnd], trainStart, trainEnd, trainEnd-trainStart+1))
        testStart = train
        testEnd = train+test-1
        print("{} - {} Testing dataset (Index {} - {} = #{})".format(
            dateList[testStart], dateList[testEnd], testStart, testEnd, testEnd-testStart+1))
        valStart = train+test
        valEnd = total-1
        print("{} - {} Validation dataset (Index {} - {} = #{})".format(
            dateList[valStart], dateList[valEnd], valStart, valEnd, valEnd-valStart+1))

    return {
        'train': train,
        'test': test,
        'validation':  val
    }


def getDataSplitIndicesHotOne(dataset: TensorDataset, trainPercentage=0.8, testPercentage=0.15) -> dict:
    lengths = getDataSplitSizes(dataset.__len__(
    ), trainPercentage=trainPercentage, testPercentage=testPercentage)
    return {
        'train': [1]*lengths['train'] + [0]*lengths['test'] + [0]*lengths['validation'],
        'test': [0]*lengths['train'] + [1]*lengths['test'] + [0]*lengths['validation'],
        'validation':  [0]*lengths['train'] + [0]*lengths['test'] + [1]*lengths['validation']
    }


def getDataSplitIndices(dataset: TensorDataset, dates=None, run=None, trainPercentage=0.8, testPercentage=0.15) -> dict:
    lengths = getDataSplitSizes(dataset.__len__(
    ), trainPercentage=trainPercentage, testPercentage=testPercentage)

    split = {
        'train': range(0, lengths['train']),
        'test': range(lengths['train'], lengths['train']+lengths['test']),
        'validation': range(lengths['train']+lengths['test'], dataset.__len__())
    }

    if run is not None:
        logValue(run, "share_train", trainPercentage)
        logValue(run, "share_test", testPercentage)

    if dates is not None:
        for stage in ["train", "test", "validation"]:
            ranges = {
                "start": datetime.strptime(str(dates[split[stage][0]].item()), '%Y%m%d').strftime('%Y.%m.%d'),
                "end": datetime.strptime(str(dates[split[stage][-1]].item()), '%Y%m%d').strftime('%Y.%m.%d')
            }

            if run is not None:
                for event in ["start", "end"]:
                    logValue(run, "date_{}_{}".format(stage, event),
                             ranges[event], verbose=True)
    return split


class BertDataset(Dataset):
    def __init__(self, features, labels, selectIndices=None, transform=None):
        # super().__init__()

        if selectIndices is not None:
            self.input_ids = features["input_ids"][selectIndices]
            self.token_type_ids = features["token_type_ids"][selectIndices]
            self.attention_mask = features["attention_mask"][selectIndices]
            self.labels = labels[selectIndices]
        else:
            self.input_ids = features["input_ids"]
            self.token_type_ids = features["token_type_ids"]
            self.attention_mask = features["attention_mask"]
            self.labels = labels

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return {
            'input_ids': self.input_ids[index],
            'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index],
            'labels': self.labels[index]
        }

    def __len__(self):
        return len(self.input_ids)

    def getRawData(self):
        return self.features, self.labels
