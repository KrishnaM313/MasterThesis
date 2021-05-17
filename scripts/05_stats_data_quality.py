import os
import pandas as pd
from tqdm import tqdm
from tools_analysis import addDictionaries, analyseFileQuality
from tools_data import getBaseDir, saveJSON
from tools_latex import writeTable
from tools_logging import prettyPrint

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
    statsDir = os.path.join(baseDir, "stats")

    analysisDir = os.path.join(baseDir, "analysis")
    if not os.path.exists(analysisDir):
        os.makedirs(analysisDir)

    healthAnalysisCSVPath = os.path.join(analysisDir, "health.csv")
    climateAnalysisCSVPath = os.path.join(analysisDir, "climate.csv")

    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    count = 0
    result = {}
    total = 0

    for file in tqdm(filePaths):
        response, amount = analyseFileQuality(file)
        total += amount
        result = addDictionaries(response, result)

    header = ["attribute", "exists (abs.)", "exists (rel.)"]

    value_matrix = [
        [
            "MEP ID",
            result["mepidExists"],
            result["mepidExists"]/total*100
        ],
        [
            "Name exists",
            result["nameExists"],
            result["nameExists"]/total*100
        ],
        [
            "Political Group",
            result["politicalGroupExists"],
            result["politicalGroupExists"]/total*100
        ]
    ]

    print(value_matrix)
    providerStatsFile = os.path.join(statsDir, "data_quality_stats.tex")
    table = writeTable("Data quality statistics", header,
                       value_matrix, providerStatsFile)

    prettyPrint(result)

    statsDir = os.path.join(baseDir, "stats")
    statsFile = os.path.join(statsDir, "data_quality.json")
    if not os.path.isdir(statsDir):
        os.mkdir(statsDir)
    saveJSON(result, statsFile, verbose=True)
