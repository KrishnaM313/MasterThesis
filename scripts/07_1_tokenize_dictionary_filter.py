from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from tools_data import getBaseDir, extractDate, extractDateValues, loadJSON, getDateString, getDateInteger
from tools_parties import getIdeologyID
from tqdm import tqdm
import os
import torch

if __name__ == '__main__':

    small = False

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir, "data")
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")
    embeddingsDir = os.path.join(baseDir, "embeddings")

    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    keywordsFilePath = os.path.join(repoDir, "config", "keywords.json")
    keywords = loadJSON(keywordsFilePath)
    print(keywords)

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    # sort filepaths by filename = date to process most recent last
    filePaths.sort()

    startYear = 2018
    endYear = 2021

    print("cuda: " + str(torch.cuda.is_available()))
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased')  # bert-large-cased

    tensorHeight = 25
    tensorLength = 20
    tensorSize = tensorHeight*tensorLength

    categories = ['health', 'climate']

    # The threshold determines the number of dictionary words that
    # have to be used in a speech for it to be considered a speech
    # about that topic
    thresholds = [1, 2, 3]

    # Setup dictionary that will contain all relevant speeches


    # Trying out different thresholds: how many words have to be in the climate dictionary
    # to put this speech in the climate dataset
    for threshold in thresholds:

        # Setup empty dictionaries
        dates = {}
        labels = {}
        texts = {}
        for keywordCategory in keywords:
            dates[keywordCategory] = []
            labels[keywordCategory] = []
            texts[keywordCategory] = []

        print("Creating tensor for threshold %s" % threshold)
        # going through all the json files
        print("Processing %s JSON files" % len(filePaths))
        for filePath in tqdm(filePaths):
            # extracting date from filename
            year, month, day = extractDateValues(filePath)
            if startYear is not None:
                if year < startYear or year > endYear:
                    continue

            JSONfile = loadJSON(filePath)
            # going through all the speeches in that day
            for speech in JSONfile:
                # Check for each category of dictionary w
                for keywordCategory in keywords:
                    dictionaryWordCount = sum(
                        speech['keywordAnalysis'][keywordCategory].values())
                    if dictionaryWordCount >= threshold:
                        dates[keywordCategory] += [
                            getDateInteger(year, month, day)]
                        texts[keywordCategory] += [speech["text"]]
                        labels[keywordCategory] += [
                            getIdeologyID(speech["partyIdeology"])]

        print("Tokenizing to create tensors")
        for keywordCategory in tqdm(keywords):
            tokens = tokenizer.batch_encode_plus(
                texts[keywordCategory], verbose=True, padding="max_length", max_length=tensorSize, truncation=True,  return_tensors='pt')
            tokens['labels'] = torch.tensor(
                labels[keywordCategory], dtype=torch.int32)

            postfix = "_"+keywordCategory+"_"+str(threshold)+".pt"

            torch.save(tokens, os.path.join(embeddingsDir,
                                            "tokens"+postfix))
            torch.save(torch.tensor(dates[keywordCategory]), os.path.join(
                embeddingsDir, "dates"+postfix))
            torch.save(torch.tensor(labels[keywordCategory]), os.path.join(
                embeddingsDir, "labels"+postfix))
