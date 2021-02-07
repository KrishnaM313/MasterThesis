from collections import Counter
import pandas as pd
from tools_parties import getParty
from tools_data import extractDate, loadJSON, saveJSON
import os

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

def doKeywordAnalysis(speech, date: str, politicalGroup: str,  fileAnalysis: dict):
    if "keywordAnalysis" in speech:
        speechAnalysis = getDataFrames(speech["keywordAnalysis"], date, politicalGroup)

        if fileAnalysis is None:
            fileAnalysis = speechAnalysis
        else:
            fileAnalysis = appendAnalysis(fileAnalysis,speechAnalysis)
        #for category in fileAnalysis:
            #print(tabulate(fileAnalysis[category], headers='keys', tablefmt='psql'))

        # if count > 3:
        #     exit()
        # count += 1
    else:
        print("keyword Analysis missing")
    return fileAnalysis

def doFileAnalysis(data, date, parties, fileAnalysis):
    for n,speech in enumerate(data):
        parties, politicalGroup = getParty(speech, parties)
        fileAnalysis = doKeywordAnalysis(speech, date, politicalGroup,  fileAnalysis)
    return parties, politicalGroup, fileAnalysis

def analyseFile(filePath):
    date = extractDate(filePath)
    data = loadJSON(filePath)

    keywordsFilePath = "/home/user/workspaces/MasterThesis/scripts/02_ingest_transcripts/keywords.json"
    keywords = loadJSON(keywordsFilePath)


    baseDir = "/home/user/workspaces/MasterThesis/data"
    analysisDir = os.path.join(baseDir,"analysis")

    fileAnalysis = None
    for n,speech in enumerate(data):
        politicalGroup = getParty(speech)
        fileAnalysis = doKeywordAnalysis(speech, date, politicalGroup,  fileAnalysis) 

    for category in fileAnalysis:
        fileAnalysisPath = os.path.join(analysisDir,date + "-" +category + ".csv")
        fileAnalysis[category].to_csv(fileAnalysisPath, index=False)  