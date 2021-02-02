import os
import time
import urllib.request
from urllib.error import HTTPError
import json
import pandas as pd
from xml.etree import ElementTree as ET

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

def saveJSON(data, filepath):
    print("Write JSON file to {filepath}".format(filepath=filepath))
    with open(filepath, 'w') as outfile:
        dataEncoded = json.dumps(data, ensure_ascii=False)
        outfile.write(dataEncoded)

def loadJSON(filepath, create=None):
    print("Read JSON file from {filepath}".format(filepath=filepath))
    if os.path.exists(filepath):
      with open(filepath, "r") as jsonFile:
        return json.load(jsonFile)
    else:
      if create == "list":
        return []
      elif create == "dictionary":
        return {}
      elif create is not None:
        raise Exception("creation type \""+create+"\" in loadJSON is not valid") 


def loadFile(filepath):
  if os.path.exists(filepath):
    with open(filepath, "r") as fileObject:
      data = fileObject.read()
    return data
  else:
    raise Exception("File {filepath} does not exist".format(filepath=filepath)) 

def download(type,filepath,fileurl,header=1):
    if os.path.isfile(filepath):
        print(type + " already exists: {filepath}".format(filepath=filepath))
        print("using cached version")
    else:  
        downlodFile(fileurl,filepath)
    if type == "xls":
        return pd.read_excel(filepath, header=header)
    elif type == "xml":
        with open(filepath, "r") as xmlFile:
            xmlData = xmlFile.read()
        return ET.XML(xmlData)
    else:
        return "error"