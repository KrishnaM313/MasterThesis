import torch


from tools_data import getBaseDir, loadJSON
import torch
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch.nn as nn
import torch.nn.functional as F


from class_dataset import DayDataset



if __name__ == '__main__':

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir,"data")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")
    embeddingsDir = os.path.join(baseDir,"embeddings")

    tensors_path = os.path.join(embeddingsDir,"tokens")
    tensors = torch.load(tensors_path)

    dates_path = os.path.join(embeddingsDir,"labels")
    dates = torch.tensor(torch.load(dates_path))

    print(dates)
    exit()

    demo_file_path = os.path.join(JSONEnrichedDir,"2019-03-25.json")
    #demo_file = loadJSON(demo_file_path)

    
    # data = torchtext.data.TabularDataset(
    #     path=demo_file,
    #     format='.json',
    #     fields=[('text', torchtext.data.Field())]
    # )

    # dataset = JsonDataset(['data/1.json', 'data/2.json', ])
    # dataloader = DataLoader(dataset, batch_size=32)
    # data = Dataset.from_dict(demo_file)

    #data_loader = torch.utils.data.DataLoader

    #data = DayDataset(demo_file_path)
    dataset = TensorDataset(tensors, dates )


    #data = torch.utils.data.TensorDataset(tensors)

    #print(list(torch.utils.data.DataLoader(data, num_workers=0)))
    print(dataset.__getitem__(2))
    exit()



    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    exit()
    embeddings = torch.load(os.path.join(embeddingsDir,"tokens"))
    dates = torch.load(os.path.join(embeddingsDir,"dates"))
    labels = torch.load(os.path.join(embeddingsDir,"labels"))
    print(embeddings)
    print(dates)
    print(labels)