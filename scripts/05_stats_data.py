import os
import json
from fuzzy_match import algorithims
import time


import re
from tools_data import loadJSON, saveJSON, extractDate
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
        print(file)
        file = "/home/user/workspaces/MasterThesis/data/json_enriched/2014-07-01.json"
        # if i > 5:
        #     exit()
        analyseFile(file)

        exit()

        # filePath = os.path.join(JSONEnrichedDir,file)
        # data = loadJSON(filePath)

        # parties, politicalGroup, fileAnalysis = doFileAnalysis(data, date, parties, fileAnalysis)


        

    # for category in fileAnalysis:
    #     fileAnalysisPath = os.path.join(analysisDir,category + ".csv")
    #     fileAnalysis[category].to_csv(fileAnalysisPath, index=False)  

    # if healthAnalysisFiles:
    #     healthAnalysis = pd.concat(healthAnalysisFiles)
    #     healthAnalysis.to_csv(healthAnalysisCSVPath, index=False)  

    # if climateAnalysisFile:
    #     climateAnalysis = pd.concat(climateAnalysisFiles)
    #     climateAnalysis.to_csv(climateAnalysisCSVPath, index=False)  
    #print(climateAnalysis)
    #print(healthAnalysis)
        
    

    #print(parties)

