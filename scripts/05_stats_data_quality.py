import os
import json
from fuzzy_match import algorithims
import time


import re
from tools_data import loadJSON, saveJSON, extractDate
from tools_parties import extractFromName, getParty
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty
from tools_analysis import getDataFrames, appendAnalysis, doKeywordAnalysis, doFileAnalysis, analyseFile, analyseFileQuality, addDictionaries
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

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir,"data")
    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")
    plotsDir = os.path.join(baseDir,"plots")


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

    count = 0
    result = {}
    for file in tqdm(filePaths):

        response = analyseFileQuality(file)
        result = addDictionaries(response, result)

    print(result)


    # Parallel(n_jobs=-1)(
    #     delayed(analyseFile)(file) for file in tqdm(filePaths)
    # )

    
    analysisDir = os.path.join(baseDir,"analysis")
    analysisSummaryDir = os.path.join(baseDir,"analysisSummary")
    if not os.path.isdir(analysisSummaryDir):
        os.mkdir(analysisSummaryDir)
    # climate

