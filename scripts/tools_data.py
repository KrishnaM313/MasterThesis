import json
import os
import re
import time
import urllib.request
from urllib.error import HTTPError
from xml.etree import ElementTree as ET
import pandas as pd
from collections import Counter
import time
from datetime import datetime

def getBaseDir():
  environmentVariableName = "REPODIR"
  if os.environ.get(environmentVariableName, False):
    return os.environ.get(environmentVariableName)
  else:
    currentWorkingDirectory = os.path.abspath(os.getcwd())
    print("To avoid this message please set environment variable REPODIR with the repo path")
    path = str(input("Enter repository path or press Enter to use current working directory ("+currentWorkingDirectory+") : ") or currentWorkingDirectory)
    os.environ[environmentVariableName] = path
    return path

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

def saveJSON(data, filepath, verbose=False):
    if verbose:
      print("Write JSON file to {filepath}".format(filepath=filepath))
    with open(filepath, 'w') as outfile:
        dataEncoded = json.dumps(data, ensure_ascii=False)
        outfile.write(dataEncoded)

def loadJSON(filepath, create=None, verbose=False):
    if verbose:
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

def getDateInteger(year: int,month: int,day: int):
  return year*10000 + month*100 + day

def getDateString(year: int,month: int,day: int):
  return "{year}-{month}-{day}".format(year=str(year),month=str(month),day=str(day))

def extractDate(filename):
  year, month, day = extractDateValues(filename)
  return getDateString(year,month,day)

def extractDateValues(filename):
  pFileName = re.compile(r'.*(\d{4})-(\d{2})-(\d{2})', flags=re.DOTALL)
  FilenameExtraction = pFileName.match(filename)
  year = int(FilenameExtraction.group(1))
  month = int(FilenameExtraction.group(2))
  day = int(FilenameExtraction.group(3))
  return year, month, day


def findDictKeyValue(dictionary,key):
    if dictionary is None:
        return "Dictionary is None", None
    elif key not in dictionary:
        return "Key '{key}' does not exist in dictionary".format(key=key), None
    elif dictionary[key] == "":
        return "Key '{key}' has empty value".format(key=key), None
    else:
        return None, dictionary[key]

def speechHasOnlyNameInfo(speech):
  if ("politicalGroup" not in speech or speech["politicalGroup"] is None) \
    and (speech['mepid'] == "" or speech['mepid'] == "n/a") \
    and speech['name']:
    return True
  return False

def addDictionaries(x: dict,y: dict) -> dict:
    return dict(Counter(x)+Counter(y))