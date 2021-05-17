import os
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm

from tools_data import getBaseDir, loadJSON
from tools_latex import writeTable
from tools_parties import (getPartyIdeologyAssociations,
                           getPartyIdeologyAssociationsList)

healthAnalysis = pd.DataFrame()
climateAnalysis = pd.DataFrame()

healthAnalysisNA = 0
climateAnalysisNA = 0


def addToFileAnalysis(date, keyword, analysisArray, healthAnalysisFile, keywords):
    columns = len(keywords[keyword])
    if healthAnalysisFile is None:
        healthAnalysisFile = pd.DataFrame()
    return healthAnalysisFile


count = 0


if __name__ == '__main__':

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir, "data")
    JSONDir = os.path.join(baseDir, "json")
    embeddingsDir = os.path.join(baseDir, "embeddings")
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")
    statsDir = os.path.join(baseDir, "stats")

    assoc = getPartyIdeologyAssociations()
    assocList = getPartyIdeologyAssociationsList()

    tokens = {}
    labels = {}

    categories = ["climate", "health"]

    header = ["Ideology", "LabelID", "Occurance #"]

    for category in categories:
        table = []
        labels[category] = torch.load(os.path.join(
            embeddingsDir, "labels_{}_1.pt".format(category)))
        count = dict(Counter(labels[category].tolist()))
        countSorted = {}
        for i in range(len(assoc)):
            if i in count:
                row = [assocList[i], i, count[i]]
                table.append(row)

        outputFile = os.path.join(
            statsDir, "ideology_distribution_{}.tex".format(category))
        writeTable("Ideology distribution in dataset for speeches related to {}".format(
            category.capitalize()), header, table, outputFile)

    occurance = {}
    ideologies = {}
    for category in categories:
        occurance[category] = 0
        ideologies[category] = []

    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)


    for filePath in tqdm(filePaths):
        speeches = loadJSON(filePath)
        for speech in speeches:
            ideologies[category].append(speech["partyIdeology"])
            keywordAnalysis = speech["keywordAnalysis"]
            for category in categories:
                if(sum(keywordAnalysis[category].values()) > 0):
                    occurance[category] += 1

    for category in categories:
        print(Counter(ideologies[category]))

    header = [categories[0].capitalize(), categories[1].capitalize()]
    value_matrix = [[
        str(occurance[categories[0]]),
        str(occurance[categories[1]])
    ]]

    print(header)
    print(value_matrix)

    statsDir = os.path.join(baseDir, "stats")
    outputFile = os.path.join(statsDir, "data_size.tex")
    table = writeTable("Amount of speeches for topics",
                       header, value_matrix, outputFile)
    print(occurance)
