import re

def extractFromName(name):
    # extracts party name from brackets in name
    partySearch = re.compile('\((.*)\)')
    result = partySearch.search(name)
    if result is not None:
        name = result.group(1)
        if name == "S&amp;D":
            name = "S&D"
        return name
    else:
        return -1

def getParty(speech):
    if "politicalGroup" in speech:
        politicalGroup = speech["politicalGroup"]
    else:
        politicalGroup = "None"
    return politicalGroup

def getPartyStats(speech, parties):
    if "politicalGroup" in speech:
        if speech["politicalGroup"] in parties:
            parties[speech["politicalGroup"]] += 1
            politicalGroup = speech["politicalGroup"]
        else:
            parties[speech["politicalGroup"]] = 1
            politicalGroup = speech["politicalGroup"]
    else:
        if "none" in parties:
            parties["none"] += 1
            politicalGroup = "None"
        else:
            parties["none"] = 1
            politicalGroup = "None"
    return parties, politicalGroup