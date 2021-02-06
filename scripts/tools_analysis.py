from collections import Counter

def countKeywords(text, keywordCategories):
    txtSplit = text.split()
    counter = Counter()

    result = {}

    for category in keywordCategories:
        result[category] = {}
        #print("Checking keywords for category {}".format(category))
        for word in txtSplit:
            counter[word] += 1
        keywords = keywordCategories[category]
        for keyword in keywords:
            #print("{}: {}".format(keyword,counter[keyword]))
            result[category][keyword] = counter[keyword]
    return result

def getKeywordsCountArray(speech,keywords):
    keywordAnalysis = countKeywords(speech["text"],keywords)
    health = list(keywordAnalysis["health"].values())
    climate = list(keywordAnalysis["climate"].values())
    return health, climate
