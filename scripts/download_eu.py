import requests
import re
import urllib.request
import os
import time


downloadBaseFolder = "/home/user/workspaces/MasterThesis/data" 


url1 = 'https://data.europa.eu/euodp/data/apiodp/action/package_show'
data1 = '''{
  "id": "european-parliament-finalised-minutes-in-xml-2017"
}'''
response1 = requests.post(url1, data=data1,headers={"Content-Type": "application/json"})
print(response1)
package=response1.json()
datasets=package["result"]["dataset"]["distribution_dcat"]
print(len(datasets))
print(type(datasets))

url2 = "https://www.europarl.europa.eu/RegistreWeb/services/search"
data2 = '''{
    "references": [],
    "authors": [],
    "typesDoc": ["PPVD"],
    "eurovoc": null,
    "codeAuthor": null,
    "fulltext": "Plenary Sitting",
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

for dataset in datasets:
  title = dataset["title_dcterms"][0]["value_or_uri"]
  language = dataset["title_dcterms"][0]["lang"]
  print(title + " - " + language) 
  search = re.search("[\d]{4}", title)
  year = search.group()
  response2 = requests.post(url2, data=(data2 % (year)),headers={"Content-Type": "application/json"})
  print(response2)
  documents=response2.json()["documents"]

  for document in documents:
    subfiles = document["formatDocs"]
    for subfile in subfiles:
      if subfile["typeDoc"] == "text/xml":
        search2 = re.search("([\d]{2}-[\d]{2})", document["reference"])
        monthDay = search2.group()

        fileUrl = subfile["url"]
        print("File {}: {}, ".format(year,document["reference"]), end='')

        folderPath = os.path.join(downloadBaseFolder, year)

        # Check if folder exists or create it
        if not os.path.exists(folderPath):
          os.makedirs(folderPath)
        
        filePath = os.path.join(folderPath, monthDay + ".xml")

        # Check if file already exists
        if os.path.isfile(filePath):
          print ("File exists")
        else:
          print ("Downloading")
          remaining_download_tries = 15
          while remaining_download_tries > 0 :
            try:
              urllib.request.urlretrieve(fileUrl, filePath)
              time.sleep(0.1)
            except:
              print("error downloading " + document["reference"]   +" on trial no: " + str(16 - remaining_download_tries))
              remaining_download_tries = remaining_download_tries - 1
              continue
            else:
              break
          

  


