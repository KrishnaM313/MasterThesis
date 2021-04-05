
from tools_analysis import countKeywords
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tools_data import getBaseDir, loadJSON
from model_finetuned import FintunedModel
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader, TensorDataset, random_split
from torch import Tensor
import math
from tools_dataset import getDataSplitSizes, BertDataset, getDataSplitIndices, splitDateList
from tqdm import tqdm
from datasets import Dataset
import mlflow
from azureml.core import Run
import argparse
import torch.nn as nn
from sklearn.metrics import classification_report
import json
from tools_logging import logValues, logConfusionMatrix, printModelParameters
from tools_data import getFileList, convertFileListToDateList
# from pytorch_lightning.loggers import MLFlowLogger
from pandas import DataFrame
from icecream import ic
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from tools_parties import getIdeologyID

if __name__ == '__main__':

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the training data"
    )
    # parser.add_argument(
    #     "--pretrained-model",
    #     type=str,
    #     help="Path to the pretrained model to avoid download"
    # )
    parser.add_argument(
        "--category",
        type=str,
        help="Which dictionary should be used to determine which speeches are included. just determines the filename picked. No processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for randomization",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to train"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./outputs",
        help='output directory'
    )

    args = parser.parse_args()



    run = Run.get_context()

    fileList = getFileList(args.data_path)
    fileList.sort()
    dateList = convertFileListToDateList(fileList)

    split = getDataSplitSizes(len(fileList), dateList=dateList, verbose=True)
    splitDateList = splitDateList(dateList)
    #print(split)
    #print(splitDateList)

    trainingSet = splitDateList["train"]
    
    sc = StandardScaler()

    data = {}

    for stage in tqdm(["train", "test"], position=0): #validate
        data[stage] = {}
        for dictionary in tqdm(["health", "climate"], position=1):
            data[stage][dictionary] = {}
            attribute_labels = None
            attributes = []
            labels = []

            for date in tqdm(trainingSet, position=2):
                dateJSON = loadJSON(os.path.join(args.data_path, date + ".json"))
                for speech in dateJSON:
                    consider = sum(speech["keywordAnalysis"][dictionary].values()) > 0
                    if consider:
                        if attribute_labels == None:
                            attribute_labels = speech["keywordAnalysis"][dictionary].keys()
                        attributes += [speech["keywordAnalysis"][dictionary].values()]
                        labels += [getIdeologyID(speech["partyIdeology"])]
                
            #attributesDf = DataFrame(attributes,columns=attribute_labels)
            #labelsDf = DataFrame(labels,columns=["label"])


            data[stage][dictionary]["attributes"] = DataFrame(attributes,columns=attribute_labels)
            data[stage][dictionary]["labels"] = DataFrame(labels,columns=["label"])
            

            #print(attributesDf.head())
            #print(ic(labelsDf.head()))


            #attributesDf = sc.fit_transform(attributesDf)
            #X_test = sc.transform(X_test)

            #exit()


    dictionary = "health"

    #print(data[stage][dictionary]["attributes"])
    #print(data[stage][dictionary]["labels"])

    data["train"][dictionary]["attributes"] = sc.fit_transform(data["train"][dictionary]["attributes"])
    data["test"][dictionary]["attributes"] = sc.transform(data["test"][dictionary]["attributes"])

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(data["train"][dictionary]["attributes"], data["train"][dictionary]["labels"])
    y_pred = regressor.predict(data["test"][dictionary]["attributes"])

    print('Mean Absolute Error:', metrics.mean_absolute_error(data["test"][dictionary]["labels"], y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(data["test"][dictionary]["labels"], y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data["test"][dictionary]["labels"], y_pred)))

    #print(dateList)

    #print()
    

    #postfix = "_"+args.category+"_"+str(args.threshold)+".pt"

    #tokens_path = os.path.join(args.data_path, "tokens"+postfix)
    #tokens = torch.load(tokens_path)

    #labels_path = os.path.join(args.data_path, "labels"+postfix)
    #labels = torch.load(labels_path)
