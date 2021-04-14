
from tools_analysis import countKeywords
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tools_data import getBaseDir, loadJSON
from model_finetuned import FintunedModel
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader, TensorDataset, random_split
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
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from tools_parties import getIdeologyID
from tools_nn import evaluateResult
from icecream import ic
import joblib

if __name__ == '__main__':

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the training data"
    )
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
        "--threshold",
        type=int,
        default=1,
        help="How many words have to be included from the dictionary for the speech to be included. just determines the filename picked. No processing.",
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
    parser.add_argument(
        '--labels',
        type=str,
        help="which tensor data set to use as labels: can be leftRightPosition or partyGroupIdeology",
    )

    #default=0.80,
    parser.add_argument(
        "--train_share",
        type=float
    )
    
    #default=0.15,
    parser.add_argument(
        "--test_share",
        type=float
    )

    args = parser.parse_args()

    run = Run.get_context()

    postfix = "_"+args.category+"_"+str(args.threshold)+".pt"
    # Load Tensors
    tokens: torch.Tensor = torch.load(os.path.join(args.data_path, "tokens"+postfix))

    if args.labels == "leftRightPosition":
        labels: torch.Tensor = torch.load(os.path.join(args.data_path, "leftRightLabels"+postfix))
    elif args.labels == "partyGroupIdeology":
        labels: torch.Tensor = torch.load(os.path.join(args.data_path, "labels"+postfix))
    

    dates: torch.Tensor = torch.load(os.path.join(args.data_path, "dates"+postfix))
    keywordAnalysis_list: torch.Tensor = torch.load(os.path.join(args.data_path, "keywordAnalysis"+postfix))

    n = len(keywordAnalysis_list)
    m = len(keywordAnalysis_list[0])

    keywordAnalysis = np.empty([n, m], dtype=int)

    for i, entry in enumerate(keywordAnalysis_list):
        keywordAnalysis[i,:] = np.fromiter(entry.values(), dtype=int)

    # Create Dataset
    dataset = BertDataset(tokens, labels)

    # trainPercentage=0.9 testPercentage=0.05
    # train: 2018.01.15 - 2020.10.21
    # test: 2020.10.21 - 2020.11.25
    # validation: 2020.12.14 - 2021.02.08

    # trainPercentage=0.6 testPercentage=0.1
    # train: 2018.01.15 - 2019.11.25
    # test: 2019.11.25 - 2020.01.13
    # validation: 2020.01.13 - 2021.02.08

    splitIndices = getDataSplitIndices(
        dataset, 
        dates=dates, 
        run=run, 
        trainPercentage=args.train_share, 
        testPercentage=args.test_share)

    data = {
        "train" : {
            "attributes" : keywordAnalysis[splitIndices['train']],
            "labels" : labels[splitIndices['train']]
        },
        "test" : {
            "attributes" : keywordAnalysis[splitIndices['test']],
            "labels" : labels[splitIndices['test']]
        }
    }
    
    sc = StandardScaler()

    dictionary = args.category

    data["train"]["attributes"] = sc.fit_transform(data["train"]["attributes"])
    data["test"]["attributes"] = sc.transform(data["test"]["attributes"])

    classifier = RandomForestClassifier(n_estimators=20, random_state=0)
    classifier.fit(data["train"]["attributes"], data["train"]["labels"])

    print("#### TRAIN SET")
    stage = "train"
    train_evaluation = {
        "labels" : data[stage]["labels"],
        "predicted" : classifier.predict(data[stage]["attributes"])
    }
    train_result = evaluateResult(**train_evaluation, prefix="final_train_")
    logValues(run, train_result, verbose=True)

    print("#### TESTING SET")
    stage = "test"
    test_evaluation = {
        "labels" : data[stage]["labels"],
        "predicted" : classifier.predict(data[stage]["attributes"])
        
    }
    test_result = evaluateResult(**test_evaluation, prefix="final_test_")
    logValues(run, test_result, verbose=True)

    modelName = "forest_"+args.category+"_"+args.labels+"_train"+str(round(args.train_share*100))+"_test"+str(round(args.test_share*100))
    
    modelPath = "./"+modelName+".pkl" #args.output_dir

    joblib.dump(classifier, modelPath)
    run.upload_file("outputs/"+modelName+".pkl", modelPath)

    run.register_model(
        model_name=modelName,
        model_path="outputs/"+modelName+".pkl"
    )