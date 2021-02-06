import os
import re
import json
import time
from langdetect import detect
from tools_data import loadJSON, saveJSON, loadFile
from tools_language import translateText

baseDir = "/home/user/workspaces/MasterThesis/data" 

htmlDir = os.path.join(baseDir, "html")
JSONDir = os.path.join(baseDir, "json")

availableJSONFiles = os.listdir(JSONDir)

if not os.path.exists(JSONDir):
    os.makedirs(JSONDir)

files = os.listdir(htmlDir)
files.sort(reverse=True)


globalTranslationProvider = "google"


for file in files:
    if not file.endswith(".html"):
        continue

    pFileName = re.compile(r'(\d{4})-(\d{2})-(\d{2})', flags=re.DOTALL)
    FilenameExtraction = pFileName.match(file)
    year = FilenameExtraction.group(1)
    month = FilenameExtraction.group(2)
    day = FilenameExtraction.group(3)

    print(availableJSONFiles)


    JSONFilename = "{year}-{month}-{day}.json".format(year=year,month=month,day=day)
    print(JSONFilename)
  
    reuse = False
    if JSONFilename in availableJSONFiles:
        reuse = True
        JSONfilePath = os.path.join(JSONDir,JSONFilename)
        existingData = loadJSON(JSONfilePath)
        if len(existingData) == 0:
            reuse = False
    else:
        existingData = ""

    print(file)

    filePath = os.path.join(htmlDir,file)
    data = loadFile(filePath)

    #p = re.compile('<table width="100%" border="0" cellpadding="5" cellspacing="0">(.+?)<\/table><\/td><\/tr><\/table>/mgs', re.DOTALL)
    # Regex search for finding speeches
    pSpeech = re.compile(r'<img alt="MPphoto".+?doc_subtitle_level1_bis">(.+?)</table></td></tr></table>', flags=re.DOTALL)

    # Regex search for finding Name of speaker
    pName = re.compile(r'<span class="doc_subtitle_level1_bis"><span class="bold">(.+?)</span></span>', flags=re.DOTALL)

    # Regex for finding MEP id of speaker
    pMEPId = re.compile(r'/mepphoto/(\d+?).jpg')

    pSpeechText = re.compile(r'<span class="bold">.+?</span>(.+?)</p></td><td width="16">',flags=re.DOTALL)

    # Regex to remove HTML tags from speech
    pRemoveHTML = re.compile('<.*?>', re.MULTILINE)
    pRemoveLinebreak = re.compile(r'\n', re.MULTILINE)
    
    speeches = pSpeech.finditer(data)

    count = 0

    speechObjects = []

    texts = 0
    textstranslated = 0

    for match in speeches:

        speech = data[match.start():match.end()]
        
        nameSearch = pName.search(speech)
        if nameSearch is None:
            name = "n/a"
        else:
            name = nameSearch.group(1)
            name = name.replace(",","")
            name = name.replace("  "," ")
            name = name.strip()
            #print(name)


        MEPIdSearch = pMEPId.search(speech)
        if MEPIdSearch is None:
            MEPId = "n/a"
        else:
            MEPId = MEPIdSearch.group(1)
        #print(MEPId)


        speechTextSearch = pSpeechText.search(speech)
        if speechTextSearch is None:
            continue
        speechTextHTML = speechTextSearch.group(1)
        speechText = str(re.sub(pRemoveHTML, '',speechTextHTML))

        # Remove line break command "\n" in text
        speechText = speechText.replace("\n", " ")

        try:
            language = detect(speechText)
        except:
            continue

    
        if (count+1) > len(existingData):
            reuse = False

        print("{year}-{month}-{day} #{count}: {mepid} - {language} - {name}".format(year=year,month=month,day=day,count=count,mepid=MEPId,language=language,name=name))

        texts += 1
        if reuse:
            #print("reuse existing JSON")
            # JSON file already exists
            if existingData[count]["text"] == "":
                print("text is empty. translate again")
                if language == "en":
                    text = speechText
                    translationProvider = "none"
                else:
                    textstranslated += 1
                    text = translateText(speechText,language,"en", globalTranslationProvider)
                    translationProvider = globalTranslationProvider
            else:
                #print("translation already exists")
                text = existingData[count]["text"]
                if "translation_provider" not in existingData[count] or existingData[count]["translation_provider"] == "":
                    translationProvider = "azure"
                else:
                    translationProvider = existingData[count]["translation_provider"]
        else:
            print("translating ...")
            # no JSON file exists for this day
            if language == "en":
                text = speechText
                translationProvider = "none"
            if language == "sq":
                # was not available in certain translation providers
                text = ""
                translationProvider = ""
            else:
                text = translateText(speechText,language,"en", globalTranslationProvider)
                translationProvider = globalTranslationProvider
            
        
        
        speechObject = {
            "id" : count,
            "date" : "{year}-{month}-{day}.json".format(year=year,month=month,day=day),
            "name" : name,
            "mepid" : MEPId,
            "text" : text,
            "language" : "en",
            "original_language" : language,
            "translation_provider" : translationProvider
        }

        speechObjects.append(speechObject)

        if count % 50 == 0:
            JSONFilepath = os.path.join(JSONDir, JSONFilename)
            print("Write JSON file to {filepath}".format(filepath=JSONFilepath))
            with open(JSONFilepath, 'w') as outfile:
                dataEncoded = json.dumps(speechObjects, ensure_ascii=False)
                outfile.write(str(dataEncoded))
        count += 1

    JSONFilepath = os.path.join(JSONDir, JSONFilename)
    print("Write JSON file to {filepath}".format(filepath=JSONFilepath))
    with open(JSONFilepath, 'w') as outfile:
        dataEncoded = json.dumps(speechObjects, ensure_ascii=False)
        outfile.write(str(dataEncoded))
    print(count)

print("done")