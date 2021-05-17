import os

import pandas as pd

from tools_analysis import countInSpeech
from tools_data import getBaseDir, loadJSON
from tools_latex import writeTable

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
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")

    plotsDir = os.path.join(baseDir, "plots")

    analysisDir = os.path.join(baseDir, "analysis")
    if not os.path.exists(analysisDir):
        os.makedirs(analysisDir)

    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    count = 0
    result = {}

    categories = ["climate", "health"]

    occurance = {
        categories[0]: 0,
        categories[1]: 0
    }

    for filePath in filePaths:
        speeches = loadJSON(filePath)
        for speech in speeches:

            keywordAnalysis = speech["keywordAnalysis"]

            for category in categories:
                if(sum(keywordAnalysis[category].values()) > 0):
                    occurance[category] += 1

    header = [categories[0].capitalize(), categories[1].capitalize()]
    value_matrix = [
        occurance[categories[0]],
        occurance[categories[1]]
    ]

    outputFile = os.path.join(analysisDir,"text_length")

    table = writeTable("Amount of speeches for topics",
                       header, value_matrix, outputFile)
    print(occurance)
    countInSpeech("word", filePaths, plotsDir, 2018,
                  2021, small=True, showPlot=False)
