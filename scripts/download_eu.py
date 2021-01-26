import requests
import re
import urllib.request
import os
import time
import textract
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

downloadBaseFolder = "/home/user/workspaces/MasterThesis/data" 




#url1 = 'https://data.europa.eu/euodp/data/apiodp/action/package_show'
#data1 = '''{
#  "id": "european-parliament-finalised-minutes-in-xml-2017"
#}'''
#response1 = requests.post(url1, data=data1,headers={"Content-Type": "application/json"})
#print(response1)
#package=response1.json()
#datasets=package["result"]["dataset"]["distribution_dcat"]

def downloadDocument(document, downloadBaseFolder, downloadedFiles):
  subfiles = document["formatDocs"]

  # folderPath = os.path.join(downloadBaseFolder, "odt" , str(year))
  
  

  # Check if folder exists or create it
  # if not os.path.exists(folderPath):
  #   os.makedirs(folderPath)


  search2 = re.search(r"([\d]{2}-[\d]{2})", document["reference"])
  monthDay = search2.group()

  print("{}-{}: {}".format(str(year),monthDay,document["reference"]))

  #filetypes = []
  #for subfile in subfiles:
    
    
  #  if (subfile["typeDoc"] == "text/xml"):
  #    filetypes.append(subfile["typeDoc"])
      #print(subfile["typeDoc"])
  #print(filetypes)
  
  for subfile in subfiles:
    if (subfile["typeDoc"] == "text/xml"):
      #break
      fileUrl = subfile["url"]
      #print(fileUrl)
      
      #print(downloadedFiles)
      
      #if subfile["typeDoc"] == "text/xml":
      extension = ".xml"
      fileName = monthDay + extension
      print("{}-{}".format(str(year),monthDay))
      filePath = os.path.join(originalDir, fileName)
      
      if fileName not in downloadedFiles:
        downlodFile(fileUrl,filePath)

      
      # txtFilePath = os.path.join(txtDir, monthDay + ".txt")
      # if os.path.isfile(txtFilePath):
      #   print (", txt exists")
      # else:
      #   print (", txt decoding")
      #   #with open(txtFilePath, 'w') as f:
      #   #  f.write(text.decode())
      # break


    # else if subfile["typeDoc"] == "application/pdf" or subfile["typeDoc"] == "application/msword":
    # # and not ("application/vnd.oasis.opendocument.text" in filetypes):
    #   if subfile["typeDoc"] == "application/pdf":
    #     extension = ".pdf"
    #   else:
    #     extension = ".doc"
      
    #   filePath = os.path.join(originalDir, monthDay + extension)
    #   # Check if file already exists
    #   if os.path.isfile(filePath):
    #     print (", file exists", end='')
    #   else:
    #     print (", downloading file", end='')
    #     remaining_download_tries = 15
    #     while remaining_download_tries > 0 :
    #       try:
    #         urllib.request.urlretrieve(fileUrl, filePath)
    #         time.sleep(0.1)
    #       except:
    #         print(", error downloading pdf " + document["reference"]   +" on trial no: " + str(16 - remaining_download_tries))
    #         remaining_download_tries = remaining_download_tries - 1
    #         continue
    #       else:
    #         break
    #   text = textract.process(filePath)

    #   txtFilePath = os.path.join(txtDir, monthDay + ".txt")
    #   if os.path.isfile(txtFilePath):
    #     print (", txt exists")
    #   else:
    #     print (", txt decoding")
    #     with open(txtFilePath, 'w') as f:
    #       f.write(text.decode())
    #   break

          
def downlodFile(fileUrl,filePath):
  if os.path.isfile(filePath):
    print ("file exists")
  else:
    print ("downloading file")
    remaining_download_tries = 15
    while remaining_download_tries > 0 :
      try:
        urllib.request.urlretrieve(fileUrl, filePath)
        time.sleep(0.1)
      except:
        print("error downloading pdf " + document["reference"]   +" on trial no: " + str(16 - remaining_download_tries))
        remaining_download_tries = remaining_download_tries - 1
        continue
      else:
        break





url2 = "https://www.europarl.europa.eu/RegistreWeb/services/search"
data2 = '''{
    "references": [],
    "authors": [],
    "typesDoc": ["PPVD"],
    "eurovoc": null,
    "codeAuthor": null,
    "fulltext": "Minutes - Plenary sitting",
		"fragDocu":"FULL",
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
# #for dataset in datasets:
#   # title = dataset["title_dcterms"][0]["value_or_uri"]
#   # language = dataset["title_dcterms"][0]["lang"]
#   # print(title + " - " + language) 
#   # search = re.search("[\d]{4}", title)
#   # year = search.group()
  response2 = requests.post(url2, data=(data2 % (str(year))),headers={"Content-Type": "application/json"})
  #print(response2)
  documents=response2.json()["documents"]

  txtDir = os.path.join(downloadBaseFolder, "txt" , str(year))
  if not os.path.exists(txtDir):
    os.makedirs(txtDir)

  originalDir = os.path.join(downloadBaseFolder, "original" , str(year))
  if not os.path.exists(originalDir):
    os.makedirs(originalDir)

  downloadedFiles = os.listdir(originalDir)

  #print(response2.json())
  for document in documents:
    downloadDocument(document, downloadBaseFolder, downloadedFiles)


  # results = Parallel(n_jobs=-1, verbose=50)(
  #   map(delayed(downloadDocument), documents)
  # )




