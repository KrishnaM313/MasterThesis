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

    print("cuda: "+ str(torch.cuda.is_available()))

    #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #bert-large-cased
    model = BertModel.from_pretrained("bert-base-uncased")


    tensorHeight = 25
    tensorLength = 20
    tensorSize = tensorHeight*tensorLength

    dates = []
    texts = []
    labels = []
    for filePath in tqdm(filePaths):

        year, month, day = extractDateValues(filePath)
        #print(year)
        if startYear is not None:
            if year < startYear or year > endYear:
                continue
        
        JSONfile = loadJSON(filePath)

        #count = 0
        for speech in JSONfile:
            #count += 1
            #if (count > 10):
            #    break
            dates += [getDateInteger(year,month,day)]
            encodedInput = tokenizer.encode(speech["text"], verbose=True, padding="max_length", max_length=tensorSize, truncation=True,  return_tensors='pt')
            encodedMatrix = encodedInput.reshape(tensorHeight,tensorLength)
            texts += [torch.unsqueeze(encodedMatrix, 0)]
            #labels += [getIdeologyID(speech["partyIdeology"])]

            #print(encoded_input)
            #exit()
            #labels += [getIdeologyID(speech["partyIdeology"])]
            #labels = torch.tensor([getIdeologyID(speech["partyIdeology"])]).unsqueeze(0)
            #output = model(**encoded_input, labels=labels)
            #print(type(output))
            #print(output)

            #texts += [speech["text"]]
            
        #break
        # print(texts)
        # print(labels)
        # print(dates)

        #encoded_input = tokenizer(texts, verbose=True, padding="max_length", max_length=512, truncation=True,  return_tensors='pt')
        #output = model(**encoded_input)
    tensor = torch.Tensor(len(texts), tensorHeight,tensorLength)
    torch.cat(texts, out=tensor)
    print(tensor)
    print(type(tensor))
    print(len(texts))
    print(tensor.size())
    torch.save(tensor, os.path.join(embeddingsDir,"texts_"+str(year)+str(month)+str(day)))
    torch.save(torch.tensor(dates), os.path.join(embeddingsDir,"dates_"+str(year)+str(month)+str(day)))
    torch.save(torch.tensor(labels), os.path.join(embeddingsDir,"labels_"+str(year)+str(month)+str(day)))
    #print(output)
    #print(labels)
    
    #print(torch.tensor(labels))
    
    #print(dates)
    #print(torch.tensor(dates))





#text = "Replace me by any text you'd like."
