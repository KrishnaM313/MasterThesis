import os
import json
from fuzzy_match import algorithims
import time


import re
from tools_data import loadJSON, saveJSON, extractDate, getBaseDir
from tools_parties import extractFromName, getParty
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty
from tools_analysis import getDataFrames, appendAnalysis, doKeywordAnalysis, doFileAnalysis, analyseFile
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from joblib import Parallel, delayed

healthAnalysis = pd.DataFrame()
climateAnalysis = pd.DataFrame()

healthAnalysisNA = 0
climateAnalysisNA = 0



def addToFileAnalysis(date, keyword, analysisArray, healthAnalysisFile,keywords):
    columns = len(keywords[keyword])
    if healthAnalysisFile is None:
        healthAnalysisFile = pd.DataFrame()
    return healthAnalysisFile

count = 0


if __name__ == '__main__':

    baseDir = "/home/user/workspaces/MasterThesis/data"


    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")


    analysisDir = os.path.join(baseDir,"analysis")
    if not os.path.exists(analysisDir):
        os.makedirs(analysisDir)

    healthAnalysisCSVPath = os.path.join(analysisDir,"health.csv")
    climateAnalysisCSVPath = os.path.join(analysisDir,"climate.csv")


    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)
    #files.sort()


    parties = {}

    #climate = {}
    



    healthAnalysisFile = None
    climateAnalysisFile = None

    healthAnalysisFiles = []
    climateAnalysisFiles = []

    fileAnalysisList = []

    fileAnalysis = None

    for i,file in enumerate(tqdm(filePaths)):
        analyseFile(file)

    # Parallel(n_jobs=-1)(
    #     delayed(analyseFile)(file) for file in tqdm(filePaths)
    # )

    
    analysisDir = os.path.join(baseDir,"analysis")
    analysisSummaryDir = os.path.join(baseDir,"analysisSummary")
    if not os.path.isdir(analysisSummaryDir):
        os.mkdir(analysisSummaryDir)
    # climate

    for keyword in ["climate", "health"]:
        keywordAnalysis = None
        for i,file in enumerate(tqdm(files)):
            date = extractDate(file)
            filePath = os.path.join(analysisDir, "{date}-{keyword}.csv".format(date=date,keyword=keyword))
            fileAnalysis = pd.read_csv(filePath)
            keywordAnalysis = pd.concat([keywordAnalysis, fileAnalysis])

        fileAnalysisPath = os.path.join(analysisSummaryDir, "{keyword}.csv".format(keyword=keyword))
        keywordAnalysis.to_csv(fileAnalysisPath, index=False)  