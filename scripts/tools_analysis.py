from collections import Counter
from numpy import array
import pandas as pd
from tools_parties import getParty
from tools_data import extractDate, extractDateValues, loadJSON, saveJSON, findDictKeyValue, addDictionaries
from tools_plot import plotCounterHistogram
import os
from tqdm import tqdm
from tools_stats import addPercentage

def countKeywords(text, keywordCategories):
    txtSplit = text.split()
    counter = Counter()

    result = {}

    for category in keywordCategories:
        result[category] = {}
        #print("Checking keywords for category {}".format(category))
        for word in txtSplit:
            counter[word] += 1
        keywords = keywordCategories[category]
        for keyword in keywords:
            #print("{}: {}".format(keyword,counter[keyword]))
            result[category][keyword] = counter[keyword]
    return result

# def getKeywordsCountArray(speech,keywords):
#     keywordAnalysis = countKeywords(speech["text"],keywords)
#     health = list(keywordAnalysis["health"].values())
#     climate = list(keywordAnalysis["climate"].values())
#     return health, climate

def getDataFrames(analysisDictionary: dict, date: str, party: str):
    dataFrames = {}
    for category in analysisDictionary:
        analysisDictionary[category]["politicalParty"] = party
        df = pd.Series(analysisDictionary[category]).to_frame().T
        df["date"] = date
        dataFrames[category] = df
    return dataFrames

def appendAnalysis(fileAnalysis: pd.DataFrame,speechAnalysis: pd.DataFrame):
    for category in fileAnalysis:
        fileAnalysis[category] = updateAnalysisRow(category, fileAnalysis,speechAnalysis)
    return fileAnalysis

def updateAnalysisRow(category: str,fileAnalysis: pd.DataFrame,speechAnalysis: pd.DataFrame, verbose=False):
 

    party = speechAnalysis[category]["politicalParty"][0]
    date = speechAnalysis[category]["date"][0]

    if party == "":
        party = "na"

    if verbose:
        print("Party: {party}, date: {date}".format(party=party,date=date))

    if fileAnalysis[category] is not None:
        #print((fileAnalysis[category]["date"] == date).any())
        #print((fileAnalysis[category]["politicalParty"] == party).any())
        exists = (fileAnalysis[category]["date"] == date).any() and  (fileAnalysis[category]["politicalParty"] == party).any()
        if exists:
            #print(fileAnalysis[category])
            originalRow = fileAnalysis[category].loc[(fileAnalysis[category]['politicalParty'] == party) & (fileAnalysis[category]["date"] == date)]
            addition = speechAnalysis[category]
            

            updateColumns = originalRow.columns.tolist()
            #print(fileAnalysis[category].loc[(fileAnalysis[category]['politicalParty'] == party) & (fileAnalysis[category]["date"] == date)][updateColumns])
            updateColumns.remove("date")
            updateColumns.remove("politicalParty")

            fileAnalysis[category].loc[(fileAnalysis[category]['politicalParty'] == party) & (fileAnalysis[category]["date"] == date)][updateColumns] += addition[updateColumns]
        else:
            concatList = []
            concatList.append(fileAnalysis[category])
            concatList.append(speechAnalysis[category])
            fileAnalysis[category] = pd.concat(concatList)
    else:
        fileAnalysis[category] = speechAnalysis[category]
    #print(fileAnalysis)
    return fileAnalysis[category]

def doKeywordAnalysis(speech, category: str):
    if "keywordAnalysis" in speech:
        #speechAnalysis = getDataFrames(speech["keywordAnalysis"], date, politicalGroup)

        df = pd.Series(speech["keywordAnalysis"][category]).to_frame().T
        return None, df
        # df["date"] = date
        # dataFrames[category] = df
        #return dataFrames

        # if fileAnalysis is None:
        #     fileAnalysis = speechAnalysis
        # else:
        #     fileAnalysis = appendAnalysis(fileAnalysis,speechAnalysis)
        #for category in fileAnalysis:
            #print(tabulate(fileAnalysis[category], headers='keys', tablefmt='psql'))

        # if count > 3:
        #     exit()
        # count += 1
    else:
        return "keyword Analysis missing", None

def doFileAnalysis(data, date, parties, fileAnalysis):
    for n,speech in enumerate(data):
        parties, politicalGroup = getParty(speech, parties)
        fileAnalysis = doKeywordAnalysis(speech, date, politicalGroup,  fileAnalysis)

    return parties, politicalGroup, fileAnalysis

def analyseFile(filePath,verbose=False):
    date = extractDate(filePath)
    data = loadJSON(filePath)

    keywordsFilePath = "/home/user/workspaces/MasterThesis/scripts/02_ingest_transcripts/keywords.json"
    keywords = loadJSON(keywordsFilePath)


    baseDir = "/home/user/workspaces/MasterThesis/data"
    analysisDir = os.path.join(baseDir,"analysis")

    for category in keywords:
        fileAnalysis = None
        categoryAnalysis = None
        for n,speech in enumerate(data):
            politicalGroup = getParty(speech)
            err, speechAnalysis = doKeywordAnalysis(speech, category)
            if err:
                print(err)
            speechAnalysis.loc[0,"politicalGroup"] = politicalGroup
            speechAnalysis.loc[0,"date"] = date

            if fileAnalysis is not None:
                if (fileAnalysis['politicalGroup'] == politicalGroup).any():
                    oldRow = fileAnalysis.loc[fileAnalysis['politicalGroup'] == politicalGroup]
                    newRow = speechAnalysis
                    sumRows = oldRow.add(newRow)
                    sumRows.loc[0,"politicalGroup"] = politicalGroup
                    sumRows.loc[0,"date"] = date
                    fileAnalysis.loc[fileAnalysis['politicalGroup'] == politicalGroup] = sumRows
                else:
                    fileAnalysis = pd.concat([fileAnalysis, speechAnalysis])
            else:
                fileAnalysis = pd.concat([fileAnalysis, speechAnalysis])

    #for category in keywords:
        fileAnalysisPath = os.path.join(analysisDir,date + "-" +category + ".csv")
        if verbose:
            print("Save {category} {date} to {file}".format(category=category,date=date,file=fileAnalysisPath))
        fileAnalysis.to_csv(fileAnalysisPath, index=False)

def analyseFileQuality(filePath,verbose=False):
    #date = extractDate(filePath)
    
    data = loadJSON(filePath)

    result = {}

    result

    for n,speech in enumerate(data):
        response = {}
        response = updateDataQualityResponse(speech,response,"politicalGroup")
        response = updateDataQualityResponse(speech,response,"mepid")
        response = updateDataQualityResponse(speech,response,"name")
        response["total"] = 1
        result = addDictionaries(response, result)
    return result, len(data)

def updateDataQualityResponse(data: dict, response: dict, key: str) -> dict:
    err, politicalGroup = findDictKeyValue(data,key)
    if err is None:
        response[key + "Exists"] = 1
    else:
        response[key + "Missing"] = 1
    return response

def countInSpeech(category: str, filePaths: list, plotsDir: str, startYear=None, endYear=None, small=False, showPlot=False, title=True):
    cnt = Counter()
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

    plot = plotCounterHistogram(cnt,category, plotsDir, startYear=startYear, endYear=endYear, small=small, title=title)

