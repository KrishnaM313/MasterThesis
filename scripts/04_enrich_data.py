import os
import json
from fuzzy_match import algorithims
import time


import re
from tools_data import loadJSON, saveJSON
from tools_parties import extractFromName
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty




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


    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")

    if not os.path.exists(JSONEnrichedDir):
        os.makedirs(JSONEnrichedDir)


    files = os.listdir(JSONDir)
    files.sort()

    filesProcessed = os.listdir(JSONEnrichedDir)


    for file in files:
        # loads processed file to save time on double processing
        if file in filesProcessed:
            filePath = os.path.join(JSONEnrichedDir,file)
        else:
            filePath = os.path.join(JSONDir,file)
        data = loadJSON(filePath)

        for n,speech in enumerate(data):
            if "politicalGroup" not in data[n] or data[n]["politicalGroup"] == "":
                print("political group is empty for {mepID}: {mepName}".format(mepID=data[n]["mepid"],mepName=data[n]["name"]))
                politicalGroup = extractFromName(speech['name'])
                if politicalGroup != -1:
                    print("political group found as reference in name: '{politicalGroup}'".format(politicalGroup=politicalGroup))
                    data[n]["politicalGroup"] = politicalGroup
                else:
                    print("political group not referenced in Name")
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
                            print("political group found in Database: '{politicalGroup}'".format(politicalGroup=party))
                            data[n]["politicalGroup"] = party


           
            mepId = speech['mepid']
            if mepId == "" or mepId == "n/a":

                print("mepID is empty for {mepName}".format(mepName=data[n]["name"]))
                name = speech["name"]
                print("No MEP ID available for name {name}".format(name=name))
                
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
                        print("name found in Database for mepID: '{name}'".format(name=name,mepID=mepId))
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
            if n % 50 == 0:
                filePathOut = os.path.join(JSONEnrichedDir,file)
                saveJSON(data, filePathOut)

        filePathOut = os.path.join(JSONEnrichedDir,file)
        saveJSON(data, filePathOut)