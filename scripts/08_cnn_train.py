
from urllib.parse import ParseResultBytes
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tools_data import getBaseDir, getFileList
from model_finetuned import FintunedModel
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader, TensorDataset, random_split,SubsetRandomSampler
from torch import Tensor
import math
from tools_dataset import getDataSplitSizes, BertDataset, getDataSplitIndices
from tqdm import tqdm
from datasets import Dataset
import mlflow
from azureml.core import Run
import argparse
import torch.nn as nn
from sklearn.metrics import classification_report
import json
from tools_logging import logValues, logConfusionMatrix, printModelParameters, logValue
from tools_nn import evaluateResult, evaluateModel
from transformers import get_linear_schedule_with_warmup
# from pytorch_lightning.loggers import MLFlowLogger
# from torch.nn import functional as F
from icecream import ic
import joblib
from azureml.core.model import Model

if __name__ == '__main__':

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the training data"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to the pretrained model to avoid download"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Which dictionary should be used to determine which speeches are included. just determines the filename picked. No processing.",
    ) 
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for learning",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="How many words have to be included from the dictionary for the speech to be included. just determines the filename picked. No processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for randomization",
    )
    parser.add_argument(
        '--big',
        default=False,
        action='store_true',
        help="should small or big dataset be used for training",
    )
    parser.add_argument(
        '--labels',
        type=str,
        default="leftRightPosition",
        help="which tensor data set to use as labels: can be leftRightPosition or partyGroupIdeology",
    )
#    parser.add_argument(
#      "--momentum",
#      type=float,
#      default=0.9,
#      help="Momentum for SGD"
#    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--demo-limit",
        type=int,
        default=0,
        help="Limit for loops"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./outputs",
        help='output directory'
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

    print(getFileList(args.data_path))

    run = Run.get_context()


    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.seed)


    postfix = "_"+args.category+"_"+str(args.threshold)+".pt"
    # Load Tensors
    tokens: torch.Tensor = torch.load(os.path.join(args.data_path, "tokens"+postfix))
    if args.labels == "leftRightPosition":
        labels: torch.Tensor = torch.load(os.path.join(args.data_path, "leftRightLabels"+postfix))
    elif args.labels == "partyGroupIdeology":
        labels: torch.Tensor = torch.load(os.path.join(args.data_path, "labels"+postfix))
    
    dates: torch.Tensor = torch.load(os.path.join(args.data_path, "dates"+postfix))

    # Create Dataset
    dataset = BertDataset(tokens, labels)

    batchSize = args.batch_size

    # lengths = getDataSplitSizes(dataset)

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


    

    #trainData = torch.utils.data.Subset(dataset, splitIndices['train'])
    #testData = torch.utils.data.Subset(dataset, splitIndices['test'])
    #valData = torch.utils.data.Subset(dataset, splitIndices['validation'])

    #train_indices, valid_indices = indices[split:], indices[:split]
    #trainSampler = SubsetRandomSampler(splitIndices['train'], generator=g_cpu)
    #testSampler = SubsetRandomSampler(splitIndices['test'], generator=g_cpu)
    #valSampler = SubsetRandomSampler(splitIndices['validation'], generator=g_cpu)

    train_dataset = BertDataset(tokens, labels, splitIndices['train'])
    test_dataset = BertDataset(tokens, labels, splitIndices['test'])
    val_dataset = BertDataset(tokens, labels, splitIndices['validation'])



    #SubsetRandomSampler

    #dataset_train = BertDataset(tokens[splitIndices['train']], labels[splitIndices['train']])


    #trainSampler = SequentialSampler(trainData) #TODO: RAndomSampler
    trainDataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size
    )

    testDataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size
    )

    valDataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size
    )

    logValue(run,"batch_size",args.batch_size)
    valDataloader = DataLoader(
            val_dataset, batch_size=args.batch_size)

    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model,
        num_labels=9,
        output_attentions=False,
        output_hidden_states=False)
 
    print(torch.cuda.memory_allocated()/1024**2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.memory_allocated()/1024**2)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=args.learning_rate,
        eps=args.epsilon
    )
    logValue(run,"learning_rate",args.learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss()

    demoLimit=args.demo_limit

    train_criterion = nn.CrossEntropyLoss()

    total_step = len(trainDataloader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = args.epochs * len(trainDataloader)
    )

    model.to(device)

    for epoch in tqdm(range(args.epochs)):
        total_train_loss = 0

        model.train()

        train_epoch_labels = []
        train_epoch_predictions = []

        for i, batch in enumerate(tqdm(trainDataloader)):
            if (demoLimit>0) and (i>demoLimit):
                break
            
            labels = batch["labels"].to(device)

            model.zero_grad() # https://mccormickml.com/2019/07/22/BERT-fine-tuning/

            output, logits = model(
                input_ids=batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels
            )

            total_train_loss += output.item()

            train_batch_predicted = torch.argmax(logits, 1)

            loss = train_criterion(logits, labels)

            # appending the overall predicted and target tensor for the whole epoch to calculate the metrics as lists
            train_epoch_labels.append(torch.flatten(labels.cpu()))
            train_epoch_predictions.append(torch.flatten(train_batch_predicted.cpu()))
    
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            #if (i+1) % 20 == 0:
            #    print(evaluateResult(train_epoch_labels,train_epoch_predictions))
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
            #         .format(epoch+1, args.epochs, i+1, total_step, loss.item()))


        avg_train_loss = total_train_loss / len(trainDataloader) 
        logValue(run,"epoch_train_avg_loss", avg_train_loss)

        train_epoch_labels = torch.cat(train_epoch_labels)
        train_epoch_predictions = torch.cat(train_epoch_predictions)

        result = evaluateResult(train_epoch_labels,train_epoch_predictions, run, prefix="epoch_train_", demoLimit=demoLimit)
        print(result)
        logValues(run, result)

        # Testing
        test_result = evaluateModel(model, testDataloader, device, run, verbose=True, prefix="epoch_test_", demoLimit=demoLimit)
        logValues(run, test_result)
    
    result = evaluateModel(model, trainDataloader, device, run, verbose=True, demoLimit=demoLimit, prefix="final_train_")
    logValues(run, result)
    result = evaluateModel(model, testDataloader, device, run, verbose=True, demoLimit=demoLimit, prefix="final_test_")
    logValues(run, result)
    #result = evaluateModel(model, valDataLoader, device, run, verbose=True, demoLimit=demoLimit, prefix="final_validation_")
    #logValues(run, result)

    print("Finished Training")

    
    
    modelName = "bert_"+args.category+"_"+args.labels+"_train"+str(round(args.train_share*100))+"_test"+str(round(args.test_share*100))


    
    modelPath = "./"+modelName+".pkl" #args.output_dir

    joblib.dump(model, modelPath)
    run.upload_file("outputs/"+modelName+".pkl", modelPath)

    run.register_model(
        model_name=modelName,
        model_path="outputs/"+modelName+".pkl",
        model_framework=Model.Framework.PYTORCH
    )