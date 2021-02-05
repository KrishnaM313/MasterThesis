import os
import json
from fuzzy_match import algorithims



import re
from tools_data import loadJSON, saveJSON
from tools_parties import extractFromName
from tools_meps import downloadMEPInfos




if __name__ == '__main__':

    baseDir = "/home/user/workspaces/MasterThesis/data"

    mepInfoDir = os.path.join(baseDir,"meps")


    filePathMepById = os.path.join(mepInfoDir,"mep_list_by_id.json")
    mepsByID = loadJSON(filePathMepById)

    filePathMepByName = os.path.join(mepInfoDir,"mep_list_by_name.json")
    mepsByName = loadJSON(filePathMepByName)


    mepInfoDBFilename = "database.json"
    mepInfoDBFilepath = os.path.join(mepInfoDir,mepInfoDBFilename)

    if os.path.exists(mepInfoDBFilepath):
        with open(mepInfoDBFilepath, "rb") as fp:   # Unpickling
            mepDB = json.load(fp) 
    else:
        mepDB = {}


    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")

    if not os.path.exists(JSONEnrichedDir):
        os.makedirs(JSONEnrichedDir)


    files = os.listdir(JSONDir)
    files.sort()


    for file in files:
        filePath = os.path.join(JSONDir,file)   
        data = loadJSON(filePath)

        
        for n,speech in enumerate(data):
            mepId = speech['mepid']
            
            if data[n]["politicalGroup"] == "":
                politicalGroup = extractFromName(speech['name'])
                if politicalGroup != -1:
                    data[n]["politicalGroup"] = politicalGroup
                

            if mepId == "" or mepId == "n/a":
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
                name = mepsByID[mepId]["name"]
                #politicalGroup = mepsByID[mepId]["politicalGroup"]
                #country = mepsByID[mepId]["country"]
                #nationalPoliticalGroup = mepsByID[mepId]["nationalPoliticalGroup"]

                data[n]["name"] = name
                #data[n]["politicalGroup"] = politicalGroup
                #data[n]["country"] = country
                #data[n]["nationalPoliticalGroup"] = nationalPoliticalGroup
        #print(data[0])

            if mepId != "" and mepId != "n/a":
                infos, mepDB = downloadMEPInfos(mepId, mepInfoDir, mepInfoDBFilepath, mepDB)
                data[n] = dict(data[n].items() + infos.items())
                #data[n] = {**data[n] , **infos}
            



        
        filePathOut = os.path.join(JSONEnrichedDir,file)
        saveJSON(data, filePathOut)



