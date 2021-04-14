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

if __name__ == '__main__':

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data-path",
    #     type=str,
    #     help="Path to the training data"
    # )
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
#    parser.add_argument(
#      "--momentum",
#      type=float,
#      default=0.9,
#      help="Momentum for SGD"
#    )
    parser.add_argument(
        "--demo-limit",
        type=int,
        default=0,
        help="Limit for loops"
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run = Run.get_context()


    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(args.seed)


    #args.category = "climate"
    #labels = "partyGroupIdeology"

    repoDir = getBaseDir()
    dataDirectory = os.path.join(repoDir, "data")
    bertModelDirectory = os.path.join(dataDirectory, "models", "bert-base-uncased")
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
        trainPercentage=0.8, 
        testPercentage=0.15)

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
    logValue(run,"batch_size",args.batch_size)

    model = BertForSequenceClassification.from_pretrained(
        bertModelDirectory,
        num_labels=9,
        output_attentions=False,
        output_hidden_states=False)

    print(model)
    model.to(device)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
    model.eval()
    result = evaluateModel(model, valDataloader, device, run, demoLimit=0, verbose=True, prefix="" )
    print(result)