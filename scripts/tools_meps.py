import os
import re
import time
import urllib.request
from urllib.error import HTTPError
from tools_data import loadJSON, saveJSON
import json





pPoliticalGroup = re.compile(r'<h4 class="erpl_title-h4">Political groups</h4>.+?<li><strong>.+?</strong> : (.+?) - .+?</li>', flags=re.DOTALL)
pNationalPoliticalGroup = re.compile(r'<h4 class="erpl_title-h4">National parties</h4>.+?<li><strong>.+?</strong> ?:? ?([\w\s\(\)]*)</li>', flags=re.DOTALL)
pCountry = re.compile(r'<div class="erpl_title-h3 mt-1 mb-1">[\s\n]+(.*)\n[\s\n]+')

def downloadMEPInfos(mepID, mepInfoDir, databasePath, verbose=False): #,database
    # Errorcodes
    # 3 mepID is n/a
    if mepID == "n/a":
        if verbose:
            print("mepID is n/a")
        return 3, None

    database = loadJSON(databasePath)
    # if mepID in database:
    #     #print("MEP #{mepid} found in DB".format(mepid=mepID))
    #     return database[mepID], database
    # else:
    #     print("MEP #{mepid} not found in DB".format(mepid=mepID))
    url = "https://www.europarl.europa.eu/meps/en/" + mepID
    filePath = os.path.join(mepInfoDir,mepID + ".html")

    if os.path.exists(filePath):
        if verbose:
            print("using cached version of website")
    else:
        print("Downloading infos for MEP #{mepid}".format(mepid=mepID))
        # remaining_download_tries = 2
        # while remaining_download_tries > 0 :
        try:
            urllib.request.urlretrieve(url, filePath)
            time.sleep(0.1)
        except HTTPError as err:
            if err.code == 404:
                return print("Not available"), None
            else:
                return print("other problem"), None
                
            # else:
            #     print("error downloading pdf on trial no: " + str(2 - remaining_download_tries))
            #     remaining_download_tries = remaining_download_tries - 1
            #     continue
        # except:
        #     print("error downloading pdf on trial no: " + str(2 - remaining_download_tries))
        #     remaining_download_tries = remaining_download_tries - 1
        #     continue
        # else:
        #     break
    
    with open(filePath, "r") as htmlFile:
        htmlText = htmlFile.read()

    politicalGroupSearch = pPoliticalGroup.search(htmlText)
    if politicalGroupSearch is None:
        politicalGroup = ""
        print("No political group found")
    else:
        politicalGroup = politicalGroupSearch.group(1)
    #print(politicalGroup)
    
    
    nationalPoliticalGroupSearch = pNationalPoliticalGroup.search(htmlText)
    if politicalGroupSearch is None:
        nationalPoliticalGroup = ""
        print("No national political Group found")
    else:
        nationalPoliticalGroup = nationalPoliticalGroupSearch.group(1)
    #print(nationalPoliticalGroup)

    countrySearch = pCountry.search(htmlText)
    if countrySearch is None:
        country = ""
        print("No country found")
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

    if verbose:
        print("Write MEP DB file to {filepath}".format(filepath=databasePath)) 
    saveJSON(database,databasePath)
    # with open(mepInfoDBFilepath, 'w') as outfile:
    #     dataEncoded = json.dumps(database, ensure_ascii=False)
    #     outfile.write(str(dataEncoded))

    print(infos)

    return None, infos

def findMEP(mepsByID,mepID, mepInfoDir, databasePath):
    if mepID in mepsByID:
        return None, mepsByID[mepID]
    else:
        err, infos = downloadMEPInfos(mepID, mepInfoDir, databasePath)
        if err is not None:
            return err, None
        return infos

def findMEPName(mep):
    return findDictKeyValue(mep,"name")

def findMEPParty(mep):
    return findDictKeyValue(mep,"politicalGroup")

def findDictKeyValue(dictionary,key):
    if key not in dictionary:
        return "Key '{key}' does not exist in dictionary".format(key=key), None
    elif dictionary[key] == "":
        return "Key '{key}' has empty value".format(key=key), None
    else:
        return None, dictionary[key]