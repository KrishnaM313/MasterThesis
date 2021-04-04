
def addPercentage(dictionary: dict):
    percentageDictionary = {}
    for key in dictionary:
        percentageDictionary[key] = dictionary[key]
        if key == "total":
            continue
        percentageDictionary[key+"_percentage"] = dictionary[key]/dictionary["total"]*100
    return percentageDictionary

