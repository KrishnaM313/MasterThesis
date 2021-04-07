import re
from tools_data import saveJSON

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
    if isinstance(party, str):
        party = party.strip()

    if party == "PPE-DE":
        return "PPE"
    elif party == "EFD":
        return "EFDD"
    elif party == "ENL":
        return "ENF"
    elif party == "S&amp;D" or party == "S" or party == "PSE" or party == " S&amp;D":
        return "S&D"
    elif party == "Verts.ALE":
        return "Verts/ALE"
    elif party == "The Earl of) Dartmouth (EFDD" or party == "The Earl of) Dartmouth Mike Hookem Diane James Margot Parker and Julia Reid (EFDD" or party == "The Earl of":
        return "EFDD"
    elif party is None or party == "" or party == "None":
        return "na"
    
    return party

def getPartyIdeologyAssociations():
    return {
        "Left-wing" : ["GUE/NGL", "The Left"],
        "Social democrats" : ["S&D"],
        "Greens and regionalists": ["Verts/ALE"],
        "Liberals and centrists": ["ALDE", "ELDR", "Renew", "LDR"],
        "Christian democrats and conservatives" : ["PPE", "ECR", "RDE"],
        "Eurosceptic conservatives": ["EFDD","IND/DEM"],
        "Far-right nationalists": ["ID", "ENF"],
        "Non-Inscrits": ["NI"]
    }

def savePartyIdeologyAssociations(filepath):
    saveJSON(getPartyIdeologyAssociations(),filepath)

savePartyIdeologyAssociations('/Users/michael/workspaces/MasterThesis/data/stats/party_associations.json')


def getPartyIdeology(party):
    if isinstance(party, str):
        party = party.strip()

    ideologies =  getPartyIdeologyAssociations()
    associatedIdeology = None
    for ideology in ideologies:
        if party in ideologies[ideology]:
            associatedIdeology = ideology
    if associatedIdeology is None:
        associatedIdeology = "na"
    return associatedIdeology

    # https://en.wikipedia.org/wiki/European_Parliament
    # https://www.europe-politique.eu/parlement-europeen.htm

    return party

def getIdeologyID(ideology) -> int:
    if ideology == "Left-wing":
        return 1
    elif ideology == "Social democrats":
        return 2
    elif ideology == "Greens and regionalists":
        return 3
    elif ideology == "Liberals and centrists":
        return 4
    elif ideology == "Christian democrats and conservatives":
        return 5
    elif ideology == "Eurosceptic conservatives":
        return 6
    elif ideology == "Far-right nationalists":
        return 7
    elif ideology == "Non-Inscrits":
        return 8
    else:
        return 0