import os
import argparse

parser = argparse.ArgumentParser("extract")
parser.add_argument("--input_data", type=str, help="input_data diectory")
parser.add_argument("--output_data", type=str, help="output_data directory")
parser.add_argument("--datastore_name", type=str, help="name of datastore")

args = parser.parse_args()

print("Argument 1: %s" % args.input_data)
print("Argument 2: %s" % args.output_data)

if not (args.output_data is None):
    os.makedirs(args.output_data, exist_ok=True)
    print("%s created" % args.output_data)


import requests
import re
import urllib.request
import time
import textract
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

downloadBaseFolder = "/home/user/workspaces/MasterThesis/data" 


from azureml.core import Workspace, Datastore, Dataset

datastore_name = args.datastore_name

# get existing workspace
workspace = Workspace.from_config()
    
# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)


#url1 = 'https://data.europa.eu/euodp/data/apiodp/action/package_show'
#data1 = '''{
#  "id": "european-parliament-finalised-minutes-in-xml-2017"
#}'''
#response1 = requests.post(url1, data=data1,headers={"Content-Type": "application/json"})
#print(response1)
#package=response1.json()
#datasets=package["result"]["dataset"]["distribution_dcat"]

tmpFolder = "/home/user/workspaces/MasterThesis/tmp"

def downloadDocument(document):
  subfiles = document["formatDocs"]

  # folderPath = os.path.join(downloadBaseFolder, "odt" , str(year))
  
    
  

  # Check if folder exists or create it
  # if not os.path.exists(folderPath):
  #   os.makedirs(folderPath)
#   txtFolderPath = os.path.join(downloadBaseFolder, "txt" , str(year))
#   if not os.path.exists(txtFolderPath):
#     os.makedirs(txtFolderPath)

#   pdfFolderPath = os.path.join(downloadBaseFolder, "pdf" , str(year))
#   if not os.path.exists(pdfFolderPath):
#     os.makedirs(pdfFolderPath)

#   metaFolderPath = os.path.join(downloadBaseFolder, "meta" , str(year))
#   if not os.path.exists(metaFolderPath):
#     os.makedirs(metaFolderPath)

  search2 = re.search(r"([\d]{2}-[\d]{2})", document["reference"])
  monthDay = search2.group()

  print("{}-{}: {}".format(str(year),monthDay,document["reference"]))


  filetypes = []
  for subfile in subfiles:
    filetypes.append(subfile["typeDoc"])

  for subfile in subfiles:
    fileUrl = subfile["url"]
    if subfile["typeDoc"] == "application/pdf":
      
      
      dataset_name = str(year)+"-"+monthDay

      # Get a dataset by name
      try:
        titanic_ds = Dataset.get_by_name(workspace=workspace, name=dataset_name)
      except:
          print("No Dataset exists yet for {}".format(dataset_name))
          pdfFilePath = os.path.join(tmpFolder, monthDay + ".pdf")
          print ("Downloading PDF")
          print(pdfFilePath)
          remaining_download_tries = 15
          while remaining_download_tries > 0 :
            try:
                urllib.request.urlretrieve(fileUrl, pdfFilePath)
                time.sleep(0.1)
            except:
                print(", error downloading pdf " + document["reference"]   +" on trial no: " + str(16 - remaining_download_tries))
                remaining_download_tries = remaining_download_tries - 1
                continue
            else:
                break
          text = textract.process(pdfFilePath)
          os.remove(pdfFilePath) 
          txtFilePath = os.path.join(tmpFolder, monthDay + ".txt")
          with open(txtFilePath, 'w') as f:
            f.write(text.decode())
          txtDataset = Dataset.File.(txtFilePath)
          txtDataset.register(workspace=workspace,
                                 name=dataset_name,
                                 description='titanic training data')
          os.remove(txtFilePath) 
          exit()
    

                                 
      else:
          print("hha")
      exit()
      
      # Check if file already exists
        
      
      
    print(" ")
          

  




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

years = [x[:-5] for x in os.listdir(args.input_data)]
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
  print(response2)
  documents=response2.json()["documents"]

  #results = Parallel(n_jobs=-1, verbose=50)(
  #           map(delayed(downloadDocument), documents))
  downloadDocument(documents[0])
# for document in documents:
#     subfiles = document["formatDocs"]

#     folderPath = os.path.join(downloadBaseFolder, "odt" , str(year))
#     txtFolderPath = os.path.join(downloadBaseFolder, "txt" , str(year))
#     pdfFolderPath = os.path.join(downloadBaseFolder, "pdf" , str(year))

#     # Check if folder exists or create it
#     if not os.path.exists(folderPath):
#       os.makedirs(folderPath)
#     if not os.path.exists(txtFolderPath):
#       os.makedirs(txtFolderPath)
#     if not os.path.exists(pdfFolderPath):
#       os.makedirs(pdfFolderPath)

#     search2 = re.search(r"([\d]{2}-[\d]{2})", document["reference"])
#     monthDay = search2.group()

#     print("{}-{}: {}".format(str(year),monthDay,document["reference"]), end='')

#     filetypes = []
#     for subfile in subfiles:
#       filetypes.append(subfile["typeDoc"])

#     for subfile in subfiles:
#       fileUrl = subfile["url"]
#       # if subfile["typeDoc"] == "application/vnd.oasis.opendocument.text":

#       #   filePath = os.path.join(folderPath, monthDay + ".odt")
#       #   # Check if file already exists
#       #   if os.path.isfile(filePath):
#       #     print (", odt exists", end='')
#       #   else:
#       #     print (", downloading odt", end='')
#       #     remaining_download_tries = 15
#       #     while remaining_download_tries > 0 :
#       #       try:
#       #         urllib.request.urlretrieve(fileUrl, filePath)
#       #         time.sleep(0.1)
#       #       except:
#       #         print(", error downloading odt " + document["reference"]   +" on trial no: " + str(16 - remaining_download_tries))
#       #         remaining_download_tries = remaining_download_tries - 1
#       #         continue
#       #       else:
#       #         break
#       #     print (".", end='')
#       #   text = textract.process(filePath)
#       #   txtFilePath = os.path.join(txtFolderPath, monthDay + ".txt")
#       #   if os.path.isfile(txtFilePath):
#       #     print (", txt exists", end='')
#       #   else:
#       #     print (", txt decoding", end='')
#       #     with open(txtFilePath, 'w') as f:
#       #       f.write(text.decode())
#       #     print (".", end='')
        
#       # el
#       if subfile["typeDoc"] == "application/pdf":
#       # and not ("application/vnd.oasis.opendocument.text" in filetypes):
        
#         filePath = os.path.join(pdfFolderPath, monthDay + ".pdf")
#         # Check if file already exists
#         if os.path.isfile(filePath):
#           print (", pdf exists", end='')
#         else:
#           print (", downloading pdf", end='')
#           remaining_download_tries = 15
#           while remaining_download_tries > 0 :
#             try:
#               urllib.request.urlretrieve(fileUrl, filePath)
#               time.sleep(0.1)
#             except:
#               print(", error downloading pdf " + document["reference"]   +" on trial no: " + str(16 - remaining_download_tries))
#               remaining_download_tries = remaining_download_tries - 1
#               continue
#             else:
#               break
#           print (".", end='')
#         text = textract.process(filePath)
#         txtFilePath = os.path.join(txtFolderPath, monthDay + ".txt")
#         if os.path.isfile(txtFilePath):
#           print (", txt exists", end='')
#         else:
#           print (", txt decoding", end='')
#           with open(txtFilePath, 'w') as f:
#             f.write(text.decode())
#           print (".", end='')
#       print(" ")





