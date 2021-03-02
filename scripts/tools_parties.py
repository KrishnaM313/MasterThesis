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

def getPartyIdeology(party):
    if isinstance(party, str):
        party = party.strip()

    if party in ["GUE/NGL", "The Left"]:
        return "Left-wing"
    elif party in ["S&D"]:
        return "Social democrats"
    elif party in ["Verts/ALE"]:
        return "Greens and regionalists"
    elif party in ["ALDE", "ELDR", "Renew", "LDR"]:
        return "Liberals and centrists"
    elif party in ["PPE", "ECR", "RDE"]:
        return "Christian democrats and conservatives"
    elif party in ["EFDD","IND/DEM"]:
        return "Eurosceptic conservatives"
    elif party in ["ID", "ENF"]:
        return "Far-right nationalists"
    elif party in ["NI"]:
        return "Non-Inscrits"
    elif party is None or party == "" or party == "None":
        return "na"
    
    return party

    # What is IND/DEM?

    # https://en.wikipedia.org/wiki/European_Parliament
    # https://www.europe-politique.eu/parlement-europeen.htm

    return party