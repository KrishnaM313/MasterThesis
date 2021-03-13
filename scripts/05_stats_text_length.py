import os
import json
from fuzzy_match import algorithims
import time
from collections import Counter
import matplotlib.pyplot as plt

import re
from tools_data import loadJSON, saveJSON, extractDate, getBaseDir
from tools_parties import extractFromName, getParty
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty
from tools_analysis import getDataFrames, appendAnalysis, doKeywordAnalysis, doFileAnalysis, analyseFile, analyseFileQuality, addDictionaries
from tools_plot import plotCounterHistogram
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


    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    count = 0
    result = {}
    cnt = Counter()
    for filePath in tqdm(filePaths):
        file = loadJSON(filePath)
        for speech in file:
            cnt[len(speech["text"].split())] += 1

    plot = plotCounterHistogram(cnt, plotsDir)

