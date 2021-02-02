import os
import json
from fuzzy_match import algorithims
import urllib.request
import time
from urllib.error import HTTPError
import re
from data_tools import loadJSON, saveJSON

pPoliticalGroup = re.compile(r'<h4 class="erpl_title-h4">Political groups</h4>.+?<li><strong>.+?</strong> : (.+?) - .+?</li>', flags=re.DOTALL)
pNationalPoliticalGroup = re.compile(r'<h4 class="erpl_title-h4">National parties</h4>.+?<li><strong>.+?</strong> ?:? ?([\w\s\(\)]*)</li>', flags=re.DOTALL)
pCountry = re.compile(r'<div class="erpl_title-h3 mt-1 mb-1">[\s\n]+(.*)\n[\s\n]+')


def downloadMEPInfos(mepID, database):
    if mepID in database:
        print("MEP #{mepid} found in DB".format(mepid=mepID))
        return database[mepID], database
    url = "https://www.europarl.europa.eu/meps/en/" + mepID
    filePath = os.path.join(mepInfoDir,mepID + ".html")

    print("Downloading infos for MEP #{mepid}".format(mepid=mepID))
    remaining_download_tries = 2
    while remaining_download_tries > 0 :
        try:
            urllib.request.urlretrieve(url, filePath)
            time.sleep(0.1)
        except HTTPError as err:
            if err.code == 404:
                print("Not available")
                #return 1
            else:
                print("error downloading pdf on trial no: " + str(2 - remaining_download_tries))
                remaining_download_tries = remaining_download_tries - 1
                continue
        except:
            print("error downloading pdf on trial no: " + str(2 - remaining_download_tries))
            remaining_download_tries = remaining_download_tries - 1
            continue
        else:
            break
    
    with open(filePath, "r") as htmlFile:
        htmlText = htmlFile.read()

    politicalGroupSearch = pPoliticalGroup.search(htmlText)
    if politicalGroupSearch is None:
        politicalGroup = ""
    else:
        politicalGroup = politicalGroupSearch.group(1)
    #print(politicalGroup)

    
    nationalPoliticalGroupSearch = pNationalPoliticalGroup.search(htmlText)
    if politicalGroupSearch is None:
        nationalPoliticalGroup = ""
    else:
        nationalPoliticalGroup = nationalPoliticalGroupSearch.group(1)
    #print(nationalPoliticalGroup)

    countrySearch = pCountry.search(htmlText)
    if countrySearch is None:
        country = ""
    else:
        country = countrySearch.group(1)
    #print(country)

    infos = {
        "politicalGroup" : politicalGroup,
        "nationalPoliticalGroup" : nationalPoliticalGroup,
        "country" : country
    }

    database[mepID] = infos
    #print(infos)

    print("Write MEP DB file to {filepath}".format(filepath=mepInfoDBFilepath)) 
    with open(mepInfoDBFilepath, 'w') as outfile:
        dataEncoded = json.dumps(mepDB, ensure_ascii=False)
        outfile.write(str(dataEncoded))

    return infos, database
    




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
            infos, mepDB = downloadMEPInfos(mepId,mepDB)
            data[n] = {**data[n] , **infos}


    
    filePathOut = os.path.join(JSONEnrichedDir,file)
    saveJSON(data, filePathOut)