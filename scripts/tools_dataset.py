


from torch.utils.data.dataset import Dataset, TensorDataset
from math import floor

def getDataSplitSizes(dataset: TensorDataset):
    total = dataset.__len__()
    train = floor(total*0.8)
    test = floor(total*0.15)
    val = total - train - test
    print("Dataset got split: {} Training, {} Testing & {} Validation Sub-dataset".format(train,test,val))
    return [train, test, val]
  #tokens['token_type_ids'],
    #dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


class BertDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        # self.input_ids = features['input_ids']
        # self.token_type_ids = features['token_type_ids']
        # self.attention_mask = features['attention_mask']
        self.labels = labels
        #self.transform = transform
        
    def __getitem__(self, index):
        # Load actual image here
        x = self.features[index]
        #x = self.features[index]
        #if self.transform:
        #    x = self.transform(x)
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return len(self.features)