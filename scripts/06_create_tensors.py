import os
import re
from typing import Counter

import pandas as pd
from tqdm import tqdm

from tools_analysis import countInSpeech
from tools_data import getBaseDir, loadJSON, extractDateValues

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
    JSONEnrichedDir = os.path.join(baseDir, "json_enriched")
    plotsDir = os.path.join(baseDir, "plots")

    startYear = 2018
    endYear = 2020
    category = "word"

    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    cnt = Counter()
    startYear = 2018
    endYear = 2021
    count = 0
    result = {}

    for filePath in tqdm(filePaths):
        if startYear is not None:
            year, month, day = extractDateValues(filePath)
            if year < startYear or year > endYear:
                continue

        file = loadJSON(filePath)
        for speech in file:
            if category == "word":
                cnt[len(speech["text"].split())] += 1
            elif category == "character":
                cnt[len(speech["text"])] += 1

    countInSpeech("word", filePaths, plotsDir)
    countInSpeech("character", filePaths, plotsDir)
    countInSpeech("word", filePaths, plotsDir, 2018, 2021)
    countInSpeech("character", filePaths, plotsDir, 2018, 2021)
