import requests
import re
import urllib.request
import os
import time
import textract
import numpy as np
import datetime
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


from urllib.error import HTTPError
          
def downlodFile(fileUrl,filePath):
  if os.path.isfile(filePath):
    print ("file exists")
    return 2
  else:
    print ("downloading file")
    remaining_download_tries = 2
    while remaining_download_tries > 0 :
      try:
        urllib.request.urlretrieve(fileUrl, filePath)
        time.sleep(0.1)
      except HTTPError as err:
        if err.code == 404:
          print("Not available")
          return 1
        else:
          print("error downloading pdf on trial no: " + str(2 - remaining_download_tries))
          remaining_download_tries = remaining_download_tries - 1
          continue
      except:
        print("error downloading pdf on trial no: " + str(2 - remaining_download_tries))
        remaining_download_tries = remaining_download_tries - 1
        continue
      else:
        break
  return 0





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

years = np.arange(2003, datetime.datetime.now().year, 1).tolist() 
print("Download years: {}".format(years))





terms = [
  dict(
    number = 8,
    start = "2014-01-01",
    end = "2019-12-31"
  ),

]

import pickle

downloadDir = os.path.join(downloadBaseFolder, "html")
if not os.path.exists(downloadDir):
  os.makedirs(downloadDir)

for term in terms:
  print(term["number"])
  #base = datetime.datetime.today()
  start = datetime.datetime.strptime(term["start"], '%Y-%m-%d')
  end = min(datetime.datetime.today(),datetime.datetime.strptime(term["end"], '%Y-%m-%d'))
  delta = end - start 

  xmlDownloadStateFilename = "notAvailableDocuments.txt"
  xmlDownloadState = os.path.join(downloadDir,xmlDownloadStateFilename)

  downloadedFiles = os.listdir(downloadDir)
  #notAvailableDocuments = []

  # with open(xmlDownloadState, "wb") as fp:   #Pickling
  #   pickle.dump(notAvailableDocuments, fp)
  

  if os.path.exists(xmlDownloadState):
    with open(xmlDownloadState, "rb") as fp:   # Unpickling
      notAvailableDocuments = pickle.load(fp)
  else:
    notAvailableDocuments = []

  

  


  for day in range(delta.days + 1):
    date = start + datetime.timedelta(days=day)
    
    # https://www.europarl.europa.eu/doceo/document/CRE-8-2019-04-18_EN.html
    fileUrl = "https://www.europarl.europa.eu/doceo/document/CRE-%s-%s-%02d-%02d_EN.html" % (term["number"], date.year, date.month, date.day)
    localFilename = "%s-%02d-%02d.html" % (date.year, date.month, date.day)
    localPath = os.path.join(downloadDir,localFilename)

    if localFilename not in notAvailableDocuments:
      try:
        downloadedFiles = os.listdir(downloadDir)
      except:
        print("Error downloading")
      print(fileUrl)
      result = downlodFile(fileUrl,localPath)
      if(result == 1):
        notAvailableDocuments.append(localFilename)
      
  

  with open(xmlDownloadState, "wb") as fp:   #Pickling
    pickle.dump(notAvailableDocuments, fp)
  exit()


    #print(date)
  #date_list = [start - datetime.timedelta(days=x) for x in range(delta.days + 1)]


exit()





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




