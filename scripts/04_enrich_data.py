import os
import json
from fuzzy_match import algorithims
from fuzzywuzzy import fuzz 
import time


import re
from tools_data import loadJSON, saveJSON, extractDate, findDictKeyValue, speechHasOnlyNameInfo
from tools_parties import extractFromName, fixParty
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty
from tools_analysis import countKeywords
import numpy as np



if __name__ == '__main__':

    rootDir = "/home/user/workspaces/MasterThesis"

    baseDir = os.path.join(rootDir,"data")

    mepInfoDir = os.path.join(baseDir,"meps")


    manualFixDBFilepath = os.path.join(rootDir,"scripts","settings","manual-mep-association.json")
    print(manualFixDBFilepath)
    manualFixDB = loadJSON(manualFixDBFilepath)
    

    filePathMepById = os.path.join(mepInfoDir,"mep_list_by_id.json")
    mepsByID = loadJSON(filePathMepById)

    filePathMepByName = os.path.join(mepInfoDir,"mep_list_by_name.json")
    mepsByName = loadJSON(filePathMepByName)

    

    mepInfoDBFilename = "database.json"
    mepInfoDBFilepath = os.path.join(mepInfoDir,mepInfoDBFilename)

    mepDB = loadJSON(mepInfoDBFilepath, create="dictionary")

    ### Keyword Analysis

    keywordsFilePath = "/home/user/workspaces/MasterThesis/scripts/02_ingest_transcripts/keywords.json"
    keywords = loadJSON(keywordsFilePath)



    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")

    if not os.path.exists(JSONEnrichedDir):
        os.makedirs(JSONEnrichedDir)


    files = os.listdir(JSONDir)

    filesProcessed = os.listdir(JSONEnrichedDir)

    verbose = False

    for i,file in enumerate(files):
        date = extractDate(file)
        # loads processed file to save time on double processing
        if file in filesProcessed:
            filePath = os.path.join(JSONEnrichedDir,file)
        else:
            filePath = os.path.join(JSONDir,file)
        data = loadJSON(filePath)

        for n,speech in enumerate(data):
            # Search for name if mepID is empty

            # 1st try: automatic fuzzy matching of name
            if speechHasOnlyNameInfo(speech):
                # Just the Name of the speaker exists, mepID and politicalGroup are missing
                print("No MEP ID available for name {name}".format(name=name))
                print("{i}/{itotal} - {n}/{ntotal} - {file} - mepID is empty for {mepName}".format(i=i,n=n,itotal=len(files),ntotal=len(data),file=date,mepName=data[n]["name"]))
                #name = speech["name"]
                #print("{i}/{itotal} - {n}/{ntotal} - {file} - No MEP ID available for name {name}".format(i=i,n=n,itotal=len(files),ntotal=len(data),file=date,name=name))
                
                best = 0
                for mep in mepsByName:
                    match = fuzz.partial_ratio(name.lower(),mep.lower())
                    #exit()
                    if match > 90 and match > best:
                        best = match
                        bestName = mep
                        bestID = mepsByName[mep]
                        speech['mepid'] = bestID
                        print("Found Name '{mep}' for given name '{name}'. Will now use id '{bestID}'".format(mep=mep,name=name, bestID=bestID))
                if best == 0:
                    print("No mepID could be found for name: {name}".format(name=name))

            # 2nd try with manual matching
            if speechHasOnlyNameInfo(speech):
                # Just the Name of the speaker exists, mepID and politicalGroup are missing
                err, fix = findDictKeyValue(manualFixDB,data[n]["name"])
                if err is not None:
                    print(err)
                    print(speech)
                    exit()
                else:
                    if "politicalGroup" in fix:
                        speech["politicalGroup"] = fix["politicalGroup"]
                    if "mepID" in fix:
                        speech["mepid"] = fix["mepid"]

                name = speech["name"]

                
                ## TODO: search for ID based on name
 
            
            # Search for political Group based on mepID
            if "politicalGroup" not in data[n] or data[n]["politicalGroup"] == "":
                if verbose:
                    print("{i}.{n} political group is empty for {mepID}: {mepName}".format(i=i,n=n,mepID=data[n]["mepid"],mepName=data[n]["name"]))
                err, politicalGroup = extractFromName(speech['name'])
                politicalGroup = fixParty(politicalGroup)
                if err is None:
                    if verbose:
                        print("{i}/{itotal} - {n}/{ntotal} - {file} - political group found as reference in name: '{politicalGroup}'".format(i=i,n=n,itotal=len(files),ntotal=len(data),file=date,politicalGroup=politicalGroup))
                    data[n]["politicalGroup"] = politicalGroup
                else:
                    print(err)
                    print("{i}/{itotal} - {n}/{ntotal} - {file} - political group not referenced in Name".format(i=i,n=n,itotal=len(files),ntotal=len(data),file=date))
                    err, mep = findMEP(mepsByID, speech['mepid'], mepInfoDir, mepInfoDBFilepath)
                    if err is not None:
                        print(err)
                    else:
                        # err, name = findMEPName(mep)
                        # if err is not None:
                        #     print(err)
                        # else:
                        #     data[n]["name"] = name


                        err, party = findMEPParty(mep)
                        if err is None:
                            if verbose:
                                print("{i}/{itotal} - {n}/{ntotal} - {file} - political group found in Database: '{politicalGroup}'".format(i=i,n=n,itotal=len(files),ntotal=len(data),file=date,politicalGroup=party))
                            data[n]["politicalGroup"] = party
                        else:
                            print(err)


           

            if not (speech['mepid'] == "" or speech['mepid'] == "n/a"):
                err, mep = findMEP(mepsByID, speech['mepid'], mepInfoDir, mepInfoDBFilepath)
                if err is not None:
                    print(err)
                else:
                    if "politicalGroup" not in data[n] or data[n]["politicalGroup"] == "":
                        if "politicalGroup" in mep:
                            data[n]["politicalGroup"] = mep["politicalGroup"]

                    err, name = findMEPName(mep)
                    if err is not None:
                        print(err)
                    else:
                        if verbose:
                            print("{i}/{itotal} - {n}/{ntotal} - {file} - name found in Database for mepID: '{name}'".format(i=i,n=n,itotal=len(files),ntotal=len(data),file=date,name=name,mepID=mepId))
                        data[n]["name"] = name

            keywordAnalysis = countKeywords(speech["text"],keywords)
            data[n]["keywordAnalysis"] = keywordAnalysis
            #data[n]["health"] = health
            #data[n]["climate"] = climate

            #test = np.sum(np.array(health)) + np.sum(np.array(climate))
            #if test != 0:
            #    print("found speech with keywords: {test} words matched".format(test=test))
                #print(health)
                #print(climate)
                #time.sleep(2)

            if n % 50 == 0:
                filePathOut = os.path.join(JSONEnrichedDir,file)
                saveJSON(data, filePathOut)

        #print(file)
        #exit()
        filePathOut = os.path.join(JSONEnrichedDir,file)
        saveJSON(data, filePathOut)