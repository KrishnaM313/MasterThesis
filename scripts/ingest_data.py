
from collections import Counter
import os
import json



baseDir = "/home/user/workspaces/MasterThesis/data"

keywordsFile = "/home/user/workspaces/MasterThesis/scripts/02_ingest_transcripts/keywords.json"

txtFile = os.path.join(baseDir,"txt","2005","01-11.txt")


test = '''Herman Van Rompuy, President of the European Council. − Mr President, President of the
European Commission, honourable Members, within the space of 49 days, I have chaired
3 meetings of the European Council and a Summit of Heads of State and Government of
the eurozone. These facts illustrate the great and urgent challenges our Union is facing,
both on the economic and on the diplomatic front. It also neatly illustrates that meetings
of the European Council are not just an event: they are part of a process. Indeed, in the'''

test2 = test.split()
print(test2)

# load JSON file with categories
with open(keywordsFile) as keywordsJSON:
    keywordCategories = json.load(keywordsJSON)


def countKeywords(txtFilePath, keyWordDict):
    # Open txt file
    with open(txtFilePath, "r") as f:
        txtString = f.read()
    txtSplit = txtString.split()
    counter = Counter()

    result = {}

    for category in keywordCategories:
        result[category] = {}
        print("Checking keywords for category {}".format(category))
        for word in txtSplit:
            counter[word] += 1
        f.close() 
        keywords = keywordCategories[category]
        for keyword in keywords:
            #print("{}: {}".format(keyword,counter[keyword]))
            result[category][keyword] = counter[keyword]
    print(result)


print(countKeywords(txtFile,keywordCategories))