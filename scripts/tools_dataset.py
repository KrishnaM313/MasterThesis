

from torch.utils.data.dataset import Dataset, TensorDataset
from math import floor
from typing import Dict, List


def splitDateList(dateList: List[str]):
    split = getDataSplitSizes(len(dateList))
    return {
        'train': dateList[0:(split["train"]-1)],
        'test': dateList[split["train"]:(split["train"]+split["test"]-1)],
        'validation':  dateList[(split["train"]+split["test"]):]
    }

def getDataSplitSizes(total: int, verbose: bool=False, dateList: List[str]=None) -> Dict[str, int]:
    train = floor(total*0.8)
    test = floor(total*0.15)
    val = total - train - test
    print("Dataset got split: {} Training, {} Testing & {} Validation Sub-dataset of total {}".format(train, test, val, total))
    if verbose and dateList is not None:
        trainStart = 0
        trainEnd = train-1
        print("{} - {} Training dataset (Index {} - {} = #{})".format(dateList[trainStart], dateList[trainEnd], trainStart, trainEnd, trainEnd-trainStart+1))
        testStart = train
        testEnd = train+test-1
        print("{} - {} Testing dataset (Index {} - {} = #{})".format(dateList[testStart], dateList[testEnd], testStart, testEnd, testEnd-testStart+1))
        valStart = train+test
        valEnd = total-1
        print("{} - {} Validation dataset (Index {} - {} = #{})".format(dateList[valStart], dateList[valEnd], valStart, valEnd, valEnd-valStart+1))
    
    return {
        'train': train,
        'test': test,
        'validation':  val
    }
    # tokens['token_type_ids'],
    #dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


def getDataSplitIndices(dataset: TensorDataset) -> dict:
    lengths = getDataSplitSizes(dataset.__len__())
    return {
        'train': [1]*lengths['train'] + [0]*lengths['test'] + [0]*lengths['validation'],
        'test': [0]*lengths['train'] + [1]*lengths['test'] + [0]*lengths['validation'],
        'validation':  [0]*lengths['train'] + [0]*lengths['test'] + [1]*lengths['validation']
    }


class BertDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        # super().__init__()
        self.features = features
        # self.input_ids = features['input_ids']
        # self.token_type_ids = features['token_type_ids']
        # self.attention_mask = features['attention_mask']
        self.labels = labels
        #self.transform = transform

#  def getInputIDs(self):
#    return self.features['input_ids']
#
#  def getTokenTypeIds(self):
#    return self.features['token_type_ids']
#
#  def getAttentionMask(self):
#    return self.features['attention_mask']
#
#  def getLabels(self):
#    return self.labels

    def __getitem__(self, index):
        # Load actual image here
        #x = self.features['input_ids'][index]
        # exit()
        return {
            'input_ids': self.features['input_ids'][index],
            'token_type_ids': self.features['token_type_ids'][index],
            'attention_mask': self.features['attention_mask'][index],
            'labels': self.labels[index]
        }

        #x = self.features[index]
        #x = self.features[index]
        # if self.transform:
        #    x = self.transform(x)
        #y = self.labels[index]
        # return x, y
        # return x,y

    def __len__(self):
        return len(self.features['input_ids'])

    def getRawData(self):
        return self.features, self.labels

    def filterNAs(self):
        return self.features, self.labels
