
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
from tools_logging import logValues, logConfusionMatrix, printModelParameters
from tools_nn import evaluateResult, evaluateModel
# from pytorch_lightning.loggers import MLFlowLogger
# from torch.nn import functional as F
from icecream import ic


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
        "--learning-rate",
        type=float,
        default=0.0029,
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
#    parser.add_argument(
#      "--momentum",
#      type=float,
#      default=0.9,
#      help="Momentum for SGD"
#    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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
    labels: torch.Tensor = torch.load(os.path.join(args.data_path, "labels"+postfix))

    # Create Dataset
    dataset = BertDataset(tokens, labels)

    batchSize = args.batch_size

    # lengths = getDataSplitSizes(dataset)

    splitIndices = getDataSplitIndices(dataset)
    
    

    #trainData = torch.utils.data.Subset(dataset, splitIndices['train'])
    #testData = torch.utils.data.Subset(dataset, splitIndices['test'])
    #valData = torch.utils.data.Subset(dataset, splitIndices['validation'])

    #train_indices, valid_indices = indices[split:], indices[:split]
    #trainSampler = SubsetRandomSampler(splitIndices['train'], generator=g_cpu)
    #testSampler = SubsetRandomSampler(splitIndices['test'], generator=g_cpu)
    #valSampler = SubsetRandomSampler(splitIndices['validation'], generator=g_cpu)

    train_dataset = BertDataset(tokens, labels,splitIndices['train'])
    test_dataset = BertDataset(tokens, labels,splitIndices['test'])
    val_dataset = BertDataset(tokens, labels,splitIndices['validation'])

    #SubsetRandomSampler

    #dataset_train = BertDataset(tokens[splitIndices['train']], labels[splitIndices['train']])


    #trainSampler = SequentialSampler(trainData) #TODO: RAndomSampler
    trainDataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # for i, batch in enumerate(trainDataloader):
    #     ic(torch.sum(batch['labels']))
    #     #(torch.sum(batch["input_ids"]))
    #     if i > 5:
    #         exit()
        


#    testSampler = RandomSampler(testData)
#    testDataloader = DataLoader(
#        testData, sampler=testSampler, batch_size=batchSize, shuffle=False)
    testDataloader = DataLoader(
            test_dataset, batch_size=args.batch_size)

    #ic(torch.sum(train_dataset.__getitem__(0)["input_ids"]))
    
    #ic(splitIndices['train'])
    #for number in [0,1,2,3,4,5]:
    #    ic(torch.sum(train_dataset.__getitem__(number)["input_ids"]))
        #ic(torch.sum(test_dataset.__getitem__(number)["input_ids"]))
    #ic(sum(splitIndices['train']))
    

    # valSampler = RandomSampler(valData)
    # valDataloader = DataLoader(
    #     valData, sampler=valSampler, batch_size=batchSize)
    valDataloader = DataLoader(
            val_dataset, batch_size=args.batch_size)

    # 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model,
        num_labels=9,
        output_attentions=False,
        output_hidden_states=False)


    #printModelParameters(model)
 
    print(torch.cuda.memory_allocated()/1024**2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.memory_allocated()/1024**2)

    print(torch.cuda.memory_allocated()/1024**2)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.learning_rate)
    print(torch.cuda.memory_allocated()/1024**2)

    loss_fn = torch.nn.CrossEntropyLoss()

    demoLimit=args.demo_limit

    train_criterion = nn.CrossEntropyLoss()

    total_step = len(trainDataloader)

    model.to(device)

    for epoch in range(args.epochs):
        model.train()

        train_epoch_labels = []
        train_epoch_predictions = []

        for i, batch in enumerate(tqdm(trainDataloader)):
            if (demoLimit>0) and (i>demoLimit):
                break
            
            labels = batch["labels"].to(device)

            outputs, aux_outputs = model(
                input_ids=batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels
            )

            _, train_batch_predicted = torch.max(aux_outputs, 1)

            loss = train_criterion(aux_outputs, labels)

            # appending the overall predicted and target tensor for the whole epoch to calculate the metrics as lists
            train_epoch_labels.append(torch.flatten(labels.cpu()))
            train_epoch_predictions.append(torch.flatten(train_batch_predicted.cpu()))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if (i+1) % 20 == 0:
            #    print(evaluateResult(train_epoch_labels,train_epoch_predictions))
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
            #         .format(epoch+1, args.epochs, i+1, total_step, loss.item()))

        train_epoch_labels = torch.cat(train_epoch_labels)
        train_epoch_predictions = torch.cat(train_epoch_predictions)

        result = evaluateResult(train_epoch_labels,train_epoch_predictions, prefix="epoch_train_")
        print(result)
        logValues(run, result)

        # Testing
        test_result = evaluateModel(model, trainDataloader, device, demoLimit, verbose=True, prefix="epoch_test_")
        logValues(run, test_result)
    result = evaluateModel(model, testDataloader, device, demoLimit, verbose=True)
    logValues(run, result)

    print("Finished Training")

    torch.save(model.state_dict(), os.path.join(
        args.output_dir, "model_epoch"+str(epoch)+postfix))