
import json
import os
import re
from collections import Counter

import pandas as pd

baseDir = "/home/user/workspaces/MasterThesis/data"
txtDir = os.path.join(baseDir, "txt")

dictResultPath = os.path.join(baseDir, "dictResult")
if not os.path.exists(dictResultPath):
    os.makedirs(dictResultPath)

keywordsFile = "/home/user/workspaces/MasterThesis/scripts/02_ingest_transcripts/keywords.json"


# load JSON file with categories
with open(keywordsFile) as keywordsJSON:
    keywordCategories = json.load(keywordsJSON)


def countKeywords(txtFilePath, keyWordDict):
    # Open txt file
    with open(txtFilePath, "r") as f:
        txtString = f.read()
    txtSplit = txtString.split()
    counter = Counter()

    result = {}

    for category in keywordCategories:
        result[category] = {}
        for word in txtSplit:
            counter[word] += 1
        f.close()
        keywords = keywordCategories[category]
        for keyword in keywords:
            result[category][keyword] = counter[keyword]
    return result


# create dataframes
categoryResults = {}
for category in keywordCategories:
    columns = keywordCategories[category]
    print(type(columns))
    columns.append("sum")
    columns.append("date")

    df = pd.DataFrame(columns=columns)
    categoryResults[category] = df

years = os.listdir(txtDir)
for year in years:
    yearPath = os.path.join(dictResultPath, year)
    if not os.path.exists(yearPath):
        os.makedirs(yearPath)
    yearDir = os.path.join(txtDir, year)
    txtFiles = os.listdir(yearDir)
    for day in txtFiles:
        search = re.search(r"([\d]{2}-[\d]{2})", day)
        monthDay = search.group()
        txtFile = os.path.join(txtDir, year, day)
        result = countKeywords(txtFile, keywordCategories)
        resultFile = os.path.join(dictResultPath, year, monthDay + ".json")

        for category in keywordCategories:

            df = pd.DataFrame([result[category]])
            df['sum'] = sum(result[category].values())
            df['date'] = year + "-" + monthDay
            if categoryResults[category].empty:
                categoryResults[category] = df
            else:
                categoryResults[category] = pd.concat(
                    [categoryResults[category], df], axis=0)

        print(year + "-" + monthDay)


for category in keywordCategories:
    resultPath = os.path.join(dictResultPath, category+".csv")
    categoryResults[category].to_csv(resultPath)
