import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from tools_data import getBaseDir

if __name__ == '__main__':

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir, "data")
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")
    embeddingsDir = os.path.join(baseDir, "embeddings")

    tensors_path = os.path.join(embeddingsDir, "tokens")
    tensors = torch.load(tensors_path)

    dates_path = os.path.join(embeddingsDir, "labels")
    dates = torch.tensor(torch.load(dates_path))

    print(dates)

    demo_file_path = os.path.join(JSONEnrichedDir, "2019-03-25.json")

    dataset = TensorDataset(tensors, dates)

    print(dataset.__getitem__(2))

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

    embeddings = torch.load(os.path.join(embeddingsDir, "tokens"))
    dates = torch.load(os.path.join(embeddingsDir, "dates"))
    labels = torch.load(os.path.join(embeddingsDir, "labels"))
    print(embeddings)
    print(dates)
    print(labels)
