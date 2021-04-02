
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tools_data import getBaseDir
from model_finetuned import FintunedModel
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader, TensorDataset, random_split
from torch import Tensor
import math
from tools_dataset import getDataSplitSizes, BertDataset, getDataSplitIndices
from tqdm import tqdm
from datasets import Dataset
import mlflow
from azureml.core import Run
import argparse
# from pytorch_lightning.loggers import MLFlowLogger
# from torch.nn import functional as F

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
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    run = Run.get_context()

    postfix = "_"+args.category+"_"+str(args.threshold)+".pt"

    tokens_path = os.path.join(args.data_path, "tokens"+postfix)
    tokens = torch.load(tokens_path)

    labels_path = os.path.join(args.data_path, "labels"+postfix)
    labels = torch.load(labels_path)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = BertDataset(tokens, labels)

    batchSize = args.batch_size

    # lengths = getDataSplitSizes(dataset)

    splitIndices = getDataSplitIndices(dataset)

    trainData = torch.utils.data.Subset(dataset, splitIndices['train'])
    testData = torch.utils.data.Subset(dataset, splitIndices['test'])
    valData = torch.utils.data.Subset(dataset, splitIndices['validation'])

    trainSampler = RandomSampler(trainData)
    trainDataloader = DataLoader(
        trainData, sampler=trainSampler, batch_size=batchSize)

    testSampler = RandomSampler(testData)
    testDataloader = DataLoader(
        testData, sampler=testSampler, batch_size=batchSize)

    valSampler = RandomSampler(valData)
    valDataloader = DataLoader(
        valData, sampler=valSampler, batch_size=batchSize)

    # 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model,
        num_labels=9,
        output_attentions=False,
        output_hidden_states=False)

    print(torch.cuda.memory_allocated()/1024**2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("push model")
    print(torch.cuda.memory_allocated()/1024**2)
    model.train().to(device)
    print(torch.cuda.memory_allocated()/1024**2)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.learning_rate)
    print(torch.cuda.memory_allocated()/1024**2)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_loss = 0.0
    train_count = 0
    test_loss = 0.0
    test_count = 0

    for epoch in range(args.epochs):

                    correct = 0
            total = 0
            outputs_epoch = []
            labels_epoch = []
            predicted_epoch = []
            running_acc = 0.
            running_loss = 0.

        for i, batch in enumerate(tqdm(trainDataloader)):
            print(torch.cuda.memory_allocated()/1024**2)
            input_ids = batch["input_ids"]
            input_ids = input_ids.to(device)

            token_type_ids = batch["token_type_ids"]
            token_type_ids = token_type_ids.to(device)

            attention_mask = batch["attention_mask"]
            attention_mask = attention_mask.to(device)

            labels = batch["labels"]
            labels = labels.to(device)

            # del batch
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )  # **batch

            # loss = loss_fn(outputs, labels)

            del input_ids
            del token_type_ids
            del attention_mask
            del labels

            loss = outputs[0]
            # loss = F.cross_entropy(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                # mlf_logger.log_metric("loss", loss)
                run.log('loss', outputs[0].item())
                print(
                    f"epoch={epoch + 1}, batch={i + 1:5}: loss {outputs[0].item():.2f}")
            train_loss += outputs[0]
            train_count += 1
        torch.save(model.state_dict(), os.path.join(
            args.output_dir, "model_epoch"+str(epoch)+postfix))
    model.eval()  # prep model for evaluation
    for i, batch in enumerate(tqdm(testDataloader)):
        # forward pass: compute predicted outputs by passing inputs to the model

        input_ids = batch["input_ids"]
        input_ids = input_ids.to(device)

        token_type_ids = batch["token_type_ids"]
        token_type_ids = token_type_ids.to(device)

        attention_mask = batch["attention_mask"]
        attention_mask = attention_mask.to(device)

        labels = batch["labels"]
        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels)

        # calculate the loss
        # print(outputs)
        # loss =
        # loss = loss_fn(output, labels)

        del input_ids
        del token_type_ids
        del attention_mask
        del labels

        # update running validation loss
        test_loss += outputs[0]
        test_count += 1

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss/train_count  # len(trainDataloader)
    run.log('avg_train_loss', train_loss.item())
    test_loss = test_loss/test_count  # len(testDataloader)
    run.log('avg_test_loss', test_loss.item())
    # model.eval()
    # y_pred = model(testX)
    # test_loss = criterion(y_pred, testY)
    # print('test loss is {}'.format(test_loss))
    # model_file_name = 'model.pkl'.format(alpha)
    # with open(model, "wb") as file:
    #    joblib.dump(value=reg, filename=os.path.join('./outputs/',
    # model_file_name))
    print("Finished Training")
