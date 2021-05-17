import os

import torch
from tqdm import tqdm

from tools_data import getBaseDir

if __name__ == '__main__':
    repoDir = getBaseDir()
    print(repoDir)
    baseDir = os.path.join(repoDir, "data")
    embeddingsDir = os.path.join(baseDir, "embeddings")

    small = False
    postfix = ""
    if small:
        postfix = "_small"

    dates = torch.load(os.path.join(embeddingsDir, "dates"+postfix))
    labels = torch.load(os.path.join(embeddingsDir, "labels"+postfix))

    print(labels)
