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