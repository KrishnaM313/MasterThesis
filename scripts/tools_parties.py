import re

def extractFromName(name):
    # extracts party name from brackets in name
    partySearch = re.compile('\((.*)\)')
    result = partySearch.search(name)
    if result is not None:
        name = result.group(1)
        if name == "S&amp;D":
            name = "S&D"
        return None, name
    else:
        return "No name found", None

def getParty(speech):
    if "politicalGroup" in speech:
        politicalGroup = speech["politicalGroup"]
        if politicalGroup == "":
            politicalGroup = "na"
        elif politicalGroup == "None" or politicalGroup is None:
            politicalGroup = "na"
    else:
        politicalGroup = "na"
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

def fixParty(party):
    if party is not None:
        party = party.strip()
    if party == "PPE-DE":
        return "PPE"
    elif party == "EFD":
        return "EFDD"
    elif party == "ENL":
        return "ENF"
    elif party == "S&amp;D" or party == "S" or party == "PSE":
        return "S&D"
    elif party == "Verts.ALE":
        return "Verts/ALE"
    elif party == "The Earl of) Dartmouth (EFDD" or party == "The Earl of) Dartmouth Mike Hookem Diane James Margot Parker and Julia Reid (EFDD" or party == "The Earl of":
        return "EFDD"

    return party