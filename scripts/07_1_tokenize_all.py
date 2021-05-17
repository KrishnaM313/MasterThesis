import os

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from tools_data import extractDateValues, getBaseDir, getDateInteger, loadJSON
from tools_parties import getIdeologyID

if __name__ == '__main__':
    small = False

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir, "data")
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")
    embeddingsDir = os.path.join(baseDir, "embeddings")

    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    startYear = 2018
    endYear = 2021

    print("cuda: " + str(torch.cuda.is_available()))

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased')  # bert-large-cased

    tensorHeight = 25
    tensorLength = 20
    tensorSize = tensorHeight*tensorLength

    dates = []
    tokens = []
    labels = []

    texts = []

    for filePath in tqdm(filePaths):

        year, month, day = extractDateValues(filePath)
        if startYear is not None:
            if year < startYear or year > endYear:
                continue

        JSONfile = loadJSON(filePath)

        for speech in JSONfile:
            dates += [getDateInteger(year, month, day)]
            texts += [speech["text"]]
            labels += [getIdeologyID(speech["partyIdeology"])]

        if small:
            break
    tokens = tokenizer.batch_encode_plus(
        texts, verbose=True, padding="max_length", max_length=tensorSize, truncation=True,  return_tensors='pt')
    tokens['labels'] = torch.tensor(labels, dtype=torch.int32)

    postfix = ""
    if small:
        postfix = "_small"

    torch.save(tokens, os.path.join(embeddingsDir, "tokens"+postfix))
    torch.save(torch.tensor(dates), os.path.join(
        embeddingsDir, "dates"+postfix))
    torch.save(torch.tensor(labels), os.path.join(
        embeddingsDir, "labels"+postfix))
