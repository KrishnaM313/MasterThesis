from datetime import datetime
import numpy as np
import requests
import os
import json

import argparse

parser = argparse.ArgumentParser("download")
parser.add_argument("--output", type=str, help="output directory")

args = parser.parse_args()

print("Argument 2: %s" % args.output)

if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    print("%s created" % args.output)


#baseFolder = args.output_extract

#metaFolderPath = args.output_extract
apiResponseFolderPath = args.output #os.path.join(metaFolderPath, "apiResponse")
if not os.path.exists(apiResponseFolderPath):
    os.makedirs(apiResponseFolderPath)

url2 = "https://www.europarl.europa.eu/RegistreWeb/services/search"
data2 = '''{
    "references": [],
    "authors": [],
    "typesDoc": ["PCRE"],
    "eurovoc": null,
    "codeAuthor": null,
    "fulltext": null,
    "searchLanguages": ["EN"],
    "relations": [],
    "allAuthorities": [],
    "year": %s,
    "leg": null,
    "currentPage": 1,
    "nbRows": 400,
    "dateCriteria": {
        "field": "DATE_DOCU",
        "startDate": null,
        "endDate": null
    },
    "sortAndOrder": null
  }'''

years = np.arange(2003, datetime.now().year, 1).tolist()
print("Download years: {}".format(years))

for year in years:
  print("Year {}".format(year),end=' ')
  response2 = requests.post(url2, data=(data2 % (str(year))),headers={"Content-Type": "application/json"})
  print(response2)
  documents=response2.json()["documents"]   

  filename = os.path.join(apiResponseFolderPath, str(year) + ".json")

  with open(filename, 'w') as json_file:
    json.dump(response2.json(), json_file)