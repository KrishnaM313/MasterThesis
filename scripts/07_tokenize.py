from transformers import BertTokenizer, BertModel
from tools_data import getBaseDir, extractDate, extractDateValues, loadJSON, getDateString, getDateInteger
from tools_parties import getIdeologyID
from tqdm import tqdm
import os
import torch

if __name__ == '__main__':

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir,"data")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")
    embeddingsDir = os.path.join(baseDir,"embeddings")

    files = os.listdir(JSONEnrichedDir)
    filePaths = []



    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)


    startYear = 2018
    endYear = 2021


    #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #bert-large-cased
    model = BertModel.from_pretrained("bert-base-uncased")


    dates = []
    texts = []
    labels = []
    for filePath in tqdm(filePaths):
        year, month, day = extractDateValues(filePath)
        if startYear is not None:
            if year < startYear or year > endYear:
                continue
        
        file = loadJSON(filePath)

        count = 0
        for speech in file:
            count += 1
            if (count > 10):
                break
            dates += [getDateInteger(year,month,day)]
            texts += [speech["text"]]
            labels += [getIdeologyID(speech["partyIdeology"])]
        break

    encoded_input = tokenizer(texts, padding="max_length", max_length=512, truncation=True,  return_tensors='pt')
    output = model(**encoded_input)
    print(output)
    print(labels)
    
    print(torch.tensor(labels))
    
    print(dates)
    print(torch.tensor(dates))
    torch.save(encoded_input, os.path.join(embeddingsDir,"tokens"))
    torch.save(torch.tensor(dates), os.path.join(embeddingsDir,"dates"))
    torch.save(torch.tensor(labels), os.path.join(embeddingsDir,"labels"))




#text = "Replace me by any text you'd like."
