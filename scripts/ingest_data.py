
from collections import Counter
import os
import json
import re
import pandas as pd



baseDir = "/home/user/workspaces/MasterThesis/data"
txtDir = os.path.join(baseDir,"txt")

dictResultPath = os.path.join(baseDir,"dictResult")
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
        #print("Checking keywords for category {}".format(category))
        for word in txtSplit:
            counter[word] += 1
        f.close() 
        keywords = keywordCategories[category]
        for keyword in keywords:
            #print("{}: {}".format(keyword,counter[keyword]))
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
    #df = df.fillna(0) # with 0s rather than NaNs
    categoryResults[category] = df
#print(categoryResults)

years = os.listdir(txtDir)
for year in years:
    yearPath = os.path.join(dictResultPath,year)
    if not os.path.exists(yearPath):
        os.makedirs(yearPath)
    yearDir = os.path.join(txtDir,year)
    txtFiles = os.listdir(yearDir)
    for day in txtFiles:
        search = re.search(r"([\d]{2}-[\d]{2})", day)
        monthDay = search.group()
        txtFile = os.path.join(txtDir,year,day)
        result = countKeywords(txtFile,keywordCategories)
        resultFile = os.path.join(dictResultPath, year, monthDay + ".json")


        for category in keywordCategories:

            df = pd.DataFrame([result[category]])
            df['sum'] = sum(result[category].values())
            df['date'] = year + "-" + monthDay
            if categoryResults[category].empty:
                categoryResults[category] = df
            else:
                categoryResults[category] = pd.concat([categoryResults[category],df], axis=0)
        
        #print(df)
        #categoryResults[category].append(df)
        #print(df)
        #print(pd.concat([categoryResults[category]  , df], axis=0))


        
        #for category in keywordCategories:
        #
        # exit()
        # if os.path.isfile(resultFile):
        #     print ("analysis exists", end='')
        # else:
        #     print ("analysis writing", end='')
        #     with open(resultFile, 'w') as f:
        #         f.write(json.dumps(result))

        print(year + "-" + monthDay)


for category in keywordCategories:
    resultPath = os.path.join(dictResultPath,category+".csv")
    categoryResults[category].to_csv(resultPath)
#print(countKeywords(txtFile,keywordCategories))