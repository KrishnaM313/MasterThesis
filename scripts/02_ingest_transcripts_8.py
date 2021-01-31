import os
import re
from googletrans import Translator
import json

baseDir = "/home/user/workspaces/MasterThesis/data" 

htmlDir = os.path.join(baseDir, "html")
JSONDir = os.path.join(baseDir, "json")

if not os.path.exists(JSONDir):
    os.makedirs(JSONDir)


files = os.listdir(htmlDir)
print(files)

#### demofile
filePath = os.path.join(htmlDir,"2019-03-14.html")
fileObject = open(filePath, "r")
datademo = fileObject.read()
fileObject.close()


### demofile


for file in files:

    pFileName = re.compile(r'(\d{4})-(\d{2})-(\d{2})', flags=re.DOTALL)
    FilenameExtraction = pFileName.match(file)
    year = FilenameExtraction.group(1)
    month = FilenameExtraction.group(2)
    day = FilenameExtraction.group(3)


    print(file)

    filePath = os.path.join(htmlDir,file)
    fileObject = open(filePath, "r")
    data = fileObject.read()
    fileObject.close()

    # TODO: remove demofile
    #data = datademo

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

    for match in speeches:
        speech = data[match.start():match.end()]
        
        #print('String match "%s" at %d:%d' % (data[s:e], s, e))
        #print(count)
        count += 1

        #print('String match "%s" at %d:%d' % (data[s:e], s, e))
        #print(speech)
        #exit()
        #name = pName.findall(speech)[0]
        nameSearch = pName.search(speech)
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
        speechTextHTML = speechTextSearch.group(1)
        speechText = str(re.sub(pRemoveHTML, '',speechTextHTML))

        # Remove line break command "\n" in text
        speechText = speechText.replace("\n", " ")


        
        #exit()

        translator = Translator()
        language = translator.detect(speechText).lang
        #print(language)
        if language == "en":
            text = speechText
        else:
            text = translator.translate(speechText, dest='en').text
            #print(text)            

        print("#{count}: {language} - {mepid} - {name}".format(count=count,language=language,mepid=MEPId,name=name))
        
        speechObject = {
            "name" : name,
            "mepid" : MEPId,
            "text" : text,
            "originalLanguage" : language,
            "language" : "en"
        }
        #print(speechObject)
        #exit()
        speechObjects.append(speechObject)
        if count == 5:
            break

    
    JSONFilepath = os.path.join(JSONDir, "{year}-{month}-{day}.json".format(year=year,month=month,day=day))
    print("Write JSON file to {filepath}".format(filepath=JSONFilepath))
    with open(JSONFilepath, 'w') as outfile:
        dataEncoded = json.dumps(speechObjects, ensure_ascii=False)
        outfile.write(str(dataEncoded))
    print(count)


    

    #print(speeches.next())

    #print(m.groups())
    #print(m.group(0))
    # if m:
    #     print('Match found: ', m.group())
    # else:
    #     print('No match')
    exit()