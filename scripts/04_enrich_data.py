import os
import json
from fuzzy_match import algorithims
import time


import re
from tools_data import loadJSON, saveJSON
from tools_parties import extractFromName
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty
from tools_analysis import countKeywords, getKeywordsCountArray
import numpy as np



if __name__ == '__main__':

    baseDir = "/home/user/workspaces/MasterThesis/data"

    mepInfoDir = os.path.join(baseDir,"meps")


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


    for i,file in enumerate(files):
        # loads processed file to save time on double processing
        if file in filesProcessed:
            filePath = os.path.join(JSONEnrichedDir,file)
        else:
            filePath = os.path.join(JSONDir,file)
        data = loadJSON(filePath)

        for n,speech in enumerate(data):
            if "politicalGroup" not in data[n] or data[n]["politicalGroup"] == "":
                print("{i}.{n} political group is empty for {mepID}: {mepName}".format(i=i,n=n,mepID=data[n]["mepid"],mepName=data[n]["name"]))
                politicalGroup = extractFromName(speech['name'])
                if politicalGroup != -1:
                    print("{i}/{itotal} - {n}/{ntotal} political group found as reference in name: '{politicalGroup}'".format(i=i,n=n,itotal=len(files),ntotal=len(data),politicalGroup=politicalGroup))
                    data[n]["politicalGroup"] = politicalGroup
                else:
                    print("{i}/{itotal} - {n}/{ntotal} political group not referenced in Name".format(i=i,n=n,itotal=len(files),ntotal=len(data)))
                    mepId = speech['mepid']
                    err, mep = findMEP(mepsByID,mepId)
                    if err is not None:
                        print(err)
                    else:
                        # err, name = findMEPName(mep)
                        # if err is not None:
                        #     print(err)
                        # else:
                        #     data[n]["name"] = name


                        err, party = findMEPParty(mep)
                        if err is not None:
                            print(err)
                        else:
                            print("{i}/{itotal} - {n}/{ntotal} political group found in Database: '{politicalGroup}'".format(i=i,n=n,itotal=len(files),ntotal=len(data),politicalGroup=party))
                            data[n]["politicalGroup"] = party


           
            mepId = speech['mepid']
            if mepId == "" or mepId == "n/a":

                print("{i}/{itotal} - {n}/{ntotal} mepID is empty for {mepName}".format(i=i,n=n,itotal=len(files),ntotal=len(data),mepName=data[n]["name"]))
                name = speech["name"]
                print("{i}/{itotal} - {n}/{ntotal} No MEP ID available for name {name}".format(i=i,n=n,itotal=len(files),ntotal=len(data),name=name))
                
                best = 0
                for mep in mepsByName:
                    match = algorithims.jaro_winkler(name,mep)
                    if match > 0.80 and match > best:
                        best = match
                        bestName = mep
                        bestID = mepsByName[mep]
                        print(match)

                if best == 0:
                    continue
                    

                # print(name)
                # print("result")
                # print(bestName)
                # print(bestID)
                # exit()

                # print(match)

                # exit()
            else:
                err, mep = findMEP(mepsByID,mepId)
                if err is not None:
                    print(err)
                else:
                    err, name = findMEPName(mep)
                    if err is not None:
                        print(err)
                    else:
                        print("{i}/{itotal} - {n}/{ntotal} name found in Database for mepID: '{name}'".format(i=i,n=n,itotal=len(files),ntotal=len(data),name=name,mepID=mepId))
                        data[n]["name"] = name

                #name = mepsByID[mepId]["name"]
                #politicalGroup = mepsByID[mepId]["politicalGroup"]
                #country = mepsByID[mepId]["country"]
                #nationalPoliticalGroup = mepsByID[mepId]["nationalPoliticalGroup"]

                #data[n]["name"] = name
                #data[n]["politicalGroup"] = politicalGroup
                #data[n]["country"] = country
                #data[n]["nationalPoliticalGroup"] = nationalPoliticalGroup
        #print(data[0])

            # if mepId != "" and mepId != "n/a":
            #     infos, mepDB = downloadMEPInfos(mepId, mepInfoDir, mepInfoDBFilepath, mepDB)
            #     data[n] = dict(data[n].items() + infos.items())
            #     #data[n] = {**data[n] , **infos}
            health, climate = getKeywordsCountArray(speech,keywords)
            data[n]["health"] = health
            data[n]["climate"] = climate

            test = np.sum(np.array(health)) + np.sum(np.array(climate))
            if test != 0:
                print("found speech with keywords: {test} words matched".format(test=test))
                #print(health)
                #print(climate)
                #time.sleep(2)

            if n % 50 == 0:
                filePathOut = os.path.join(JSONEnrichedDir,file)
                saveJSON(data, filePathOut)

            

        filePathOut = os.path.join(JSONEnrichedDir,file)
        saveJSON(data, filePathOut)