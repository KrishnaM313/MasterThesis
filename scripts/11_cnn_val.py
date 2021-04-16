from azureml.core.workspace import Workspace
import torch
from tools_data import getBaseDir
import os
from azureml.core import Run
import argparse
from tools_dataset import getDataSplitSizes, BertDataset, getDataSplitIndices
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader, TensorDataset, random_split,SubsetRandomSampler
from tools_logging import logValues, logConfusionMatrix, printModelParameters, logValue
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tools_nn import evaluateModel
from azureml.core.model import Model
from azureml.core import (
    Experiment,
    Environment,
    ScriptRunConfig,
    Dataset,
    Run
)
import joblib
from icecream import ic
from tools_nn import evaluateResult

def getModelFilename(model_type, category, labels, train_share, test_share, verbose=False):
    modelFilename = model_type+'_'+category+'_'+labels+'_train'+str(round(train_share*100))+'_test'+str(round(test_share*100))+".pkl"
    if verbose:
        ic(modelFilename)
    return modelFilename

def loadModel(models_path, model_type, category, labels, train_share, test_share, verbose=False):
    modelFilename = getModelFilename(model_type, category, labels, train_share, test_share, verbose=verbose)
    modelPath = os.path.join(models_path, modelFilename)
    model = joblib.load(modelPath)
    if verbose:
        ic(model)
    return model

if __name__ == '__main__':

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data-path",
    #     type=str,
    #     help="Path to the training data"
    # )
    parser.add_argument(
        "--models-path",
        type=str,
        help="Path to the pretrained model file (joblib)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Path to the pretrained model file (joblib)",
        default="bert"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Which dictionary should be used to determine which speeches are included. just determines the filename picked. No processing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for learning",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for randomization",
    )
    parser.add_argument(
        '--labels',
        type=str,
        help="which tensor data set to use as labels: can be leftRightPosition or partyGroupIdeology",
    )

    #default=0.80,
    parser.add_argument(
        "--train-share",
        type=float
    )
    
    #default=0.15,
    parser.add_argument(
        "--test-share",
        type=float
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.seed)


    #args.category = "climate"
    #labels = "partyGroupIdeology"

    repoDir = getBaseDir()
    dataDirectory = os.path.join(repoDir, "data")
    #bertModelDirectory = os.path.join(dataDirectory, "models", "bert-base-uncased")
    modelsTrained = os.path.join(dataDirectory,"models_trained")
    modelPath = os.path.join(modelsTrained, args.category+"_"+args.labels+".pt")
    tokensDirectory = os.path.join(dataDirectory, "embeddings")

    postfix = "_"+args.category+"_1.pt"
    # Load Tensors
    tokens: torch.Tensor = torch.load(os.path.join(tokensDirectory, "tokens"+postfix))
    if args.labels == "leftRightPosition":
        labels: torch.Tensor = torch.load(os.path.join(tokensDirectory, "leftRightLabels"+postfix))
    elif args.labels == "partyGroupIdeology":
        labels: torch.Tensor = torch.load(os.path.join(tokensDirectory, "labels"+postfix))

    dates: torch.Tensor = torch.load(os.path.join(tokensDirectory, "dates"+postfix))

    # Create Dataset
    dataset = BertDataset(tokens, labels)

    run = None
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

    #train_dataset = BertDataset(tokens, labels, splitIndices['train'])
    test_dataset = BertDataset(tokens, labels, splitIndices['test'])
    val_dataset = BertDataset(tokens, labels, splitIndices['validation'])

    testDataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False)

    valDataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False
    )


    model = loadModel(args.models_path, args.model_type, args.category, args.labels, args.train_share, args.test_share, verbose=True)


    #logValue(run,"batch_size",args.batch_size)


    model.to(device)
    model.eval()
    result = evaluateModel(model, valDataloader, device, run, demoLimit=0, verbose=True, prefix="" )
    #model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
    
    
    print(result)