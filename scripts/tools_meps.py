import os
import re
import time
import urllib.request
from urllib.error import HTTPError

from tools_data import findDictKeyValue, loadJSON, saveJSON

pPoliticalGroup = re.compile(
    r'<h4 class="erpl_title-h4">Political groups</h4>.+?<li><strong>.+?</strong> : (.+?) - .+?</li>', flags=re.DOTALL)
pNationalPoliticalGroup = re.compile(
    r'<h4 class="erpl_title-h4">National parties</h4>.+?<li><strong>.+?</strong> ?:? ?([\w\s\(\)]*)</li>', flags=re.DOTALL)
pCountry = re.compile(
    r'<div class="erpl_title-h3 mt-1 mb-1">[\s\n]+(.*)\n[\s\n]+')


def downloadMEPInfos(mepID, mepInfoDir, databasePath, verbose=False):  # ,database
    # Errorcodes
    # 3 mepID is n/a
    if mepID == "n/a":
        if verbose:
            print("mepID is n/a")
        return 3, None

    database = loadJSON(databasePath)
    url = "https://www.europarl.europa.eu/meps/en/" + mepID
    filePath = os.path.join(mepInfoDir, mepID + ".html")

    if os.path.exists(filePath):
        if verbose:
            print("using cached version of website")
    else:
        print("Downloading infos for MEP #{mepid}".format(mepid=mepID))
        try:
            urllib.request.urlretrieve(url, filePath)
            time.sleep(0.1)
        except HTTPError as err:
            if err.code == 404:
                return print("Not available"), None
            else:
                return print("other problem"), None

    with open(filePath, "r") as htmlFile:
        htmlText = htmlFile.read()

    politicalGroupSearch = pPoliticalGroup.search(htmlText)
    if politicalGroupSearch is None:
        politicalGroup = ""
        print("No political group found")
    else:
        politicalGroup = politicalGroupSearch.group(1)

    nationalPoliticalGroupSearch = pNationalPoliticalGroup.search(htmlText)
    if politicalGroupSearch is None:
        nationalPoliticalGroup = ""
        print("No national political Group found")
    else:
        nationalPoliticalGroup = nationalPoliticalGroupSearch.group(1)

    countrySearch = pCountry.search(htmlText)
    if countrySearch is None:
        country = ""
        print("No country found")
    else:
        country = countrySearch.group(1)

    infos = {
        "politicalGroup": politicalGroup,
        "nationalPoliticalGroup": nationalPoliticalGroup,
        "country": country
    }

    database[mepID] = infos

    if verbose:
        print("Write MEP DB file to {filepath}".format(filepath=databasePath))
    saveJSON(database, databasePath)

    print(infos)

    return None, infos


def findMEP(mepsByID, mepID, mepInfoDir, databasePath):
    if mepID in mepsByID:
        return None, mepsByID[mepID]
    else:
        err, infos = downloadMEPInfos(mepID, mepInfoDir, databasePath)
        if err is not None:
            return err, None
        return infos


def findMEPName(mep):
    return findDictKeyValue(mep, "name")


def findMEPParty(mep):
    return findDictKeyValue(mep, "politicalGroup")

# Function changes "LASTNAME firstName" to "firstName LASTNAME"


def reorderNames(database, nameColumn="NOM", reorderdNameColumn="name"):
    for index, _ in database.iterrows():
        name = database.loc[index, nameColumn]
        if name != name:
            continue
        for i in range(0, len(name)-3):
            if name[i].isupper() and name[i+1].isspace() and name[i+2].isupper() and name[i+3].islower():

                lastName = name[0:i+1]
                firstName = name[i+2:len(name)]
                reorderedName = firstName + " " + lastName
                database.loc[index, reorderdNameColumn] = reorderedName
                break
    return database
