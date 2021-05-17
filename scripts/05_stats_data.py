import os

import pandas as pd
from tqdm import tqdm

from tools_analysis import analyseFile
from tools_data import extractDate

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

    baseDir = "/home/user/workspaces/MasterThesis/data"

    JSONDir = os.path.join(baseDir, "json")
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")

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

    parties = {}

    healthAnalysisFile = None
    climateAnalysisFile = None

    healthAnalysisFiles = []
    climateAnalysisFiles = []

    fileAnalysisList = []

    fileAnalysis = None

    for i, file in enumerate(tqdm(filePaths)):
        analyseFile(file)

    analysisDir = os.path.join(baseDir, "analysis")
    analysisSummaryDir = os.path.join(baseDir, "analysisSummary")
    if not os.path.isdir(analysisSummaryDir):
        os.mkdir(analysisSummaryDir)

    for keyword in ["climate", "health"]:
        keywordAnalysis = None
        for i, file in enumerate(tqdm(files)):
            date = extractDate(file)
            filePath = os.path.join(
                analysisDir, "{date}-{keyword}.csv".format(date=date, keyword=keyword))
            fileAnalysis = pd.read_csv(filePath)
            keywordAnalysis = pd.concat([keywordAnalysis, fileAnalysis])

        fileAnalysisPath = os.path.join(
            analysisSummaryDir, "{keyword}.csv".format(keyword=keyword))
        keywordAnalysis.to_csv(fileAnalysisPath, index=False)
