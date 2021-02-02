import os
import re
#from googletrans import Translator
from langdetect import detect
from translate import Translator
import json
import uuid
import requests
import textwrap
import googletrans
import translate
from data_tools import loadJSONListOrCreate, saveJSON
import boto3
import time

baseDir = "/home/user/workspaces/MasterThesis/data" 

htmlDir = os.path.join(baseDir, "html")
JSONDir = os.path.join(baseDir, "json")

availableJSONFiles = os.listdir(JSONDir)


progressTrackerFilepath = os.path.join(JSONDir,"progressTracker.json")
progressTracker = loadJSONListOrCreate(progressTrackerFilepath)

if not os.path.exists(JSONDir):
    os.makedirs(JSONDir)


files = os.listdir(htmlDir)
files.sort(reverse=True)





def translateText(text,from_lang,to_lang,provider):
    if provider == "azure":
        return translateAzure(text,from_lang,to_lang)
    elif provider == "google":
        return translateGoogle(text,from_lang,to_lang)
    elif provider == "mymemory":
        return translateMyMemory(text,from_lang,to_lang)
    elif provider == "aws":
        return translateAWS(text,from_lang,to_lang)

def translateAWS(text,from_lang,to_lang):
    translate = boto3.client(service_name='translate', region_name='us-east-1', use_ssl=True)

    parts = textwrap.wrap(text, 2000, break_long_words=False)

    translations = []

    for part in parts:
        result = translate.translate_text(Text=part, 
            SourceLanguageCode=from_lang, TargetLanguageCode=to_lang)
        #print(result)
        #print(response)
        translations.append(result.get('TranslatedText'))
        time.sleep(0.1)
        
    #print(' '.join(translations))
    return ' '.join(translations)




def translateMyMemory(text,from_lang,to_lang):
    envVariable = "MY_MEMORY_EMAIL"
    if os.environ.get(envVariable) is not None:
        translator = Translator(from_lang=from_lang,to_lang=to_lang,email=os.environ.get(envVariable))
        return translator.translate(text)
    else:
        raise SystemExit("Please define environment variable for MyMemory translation provider: {envVariable}".format(envVariable=envVariable))


def translateGoogle(text,from_lang,to_lang):
    translator = googletrans.Translator()

    

    parts = textwrap.wrap(text, 2000, break_long_words=False)

    translations = []

    for part in parts:
        translation = translator.translate(part, src=from_lang, dest=to_lang)
        #print(translation)
        #print(translation._response)
        #print(result)
        #print(response)
        translations.append(translation.text)
        #time.sleep(0.1)
        
    #print(' '.join(translations))
    return ' '.join(translations)


def translateAzure(text,from_lang,to_lang):
    key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
    if not key_var_name in os.environ:
        raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
    subscription_key = os.environ[key_var_name]

    endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
    if not endpoint_var_name in os.environ:
        raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
    endpoint = os.environ[endpoint_var_name]
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = "westeurope"

    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': from_lang,
        'to': [to_lang]
    }
    constructed_url = endpoint + path

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }


    parts = textwrap.wrap(text, 5000, break_long_words=False)

    translations = []

    for part in parts:
        body = [{
            'text': part
        }]

        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()
        print(response)
        translations.append(response[0]["translations"][0]["text"])
        
    #print(' '.join(translations))
    return ' '.join(translations)


    # You can pass more than one object in body.
    

    

    #print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))
    #print(response)
    

globalTranslationProvider = "google"


for file in files:
    if not file.endswith(".html"):
        continue

    #if file in progressTracker:
    #    continue

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
        with open(JSONfilePath, "r") as jsonFile:
            existingData = json.load(jsonFile) 
        if len(existingData) == 0:
            reuse = False
    else:
        existingData = ""
        #print("skipped because already translated")
        #continue
        #continue

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

    texts = 0
    textstranslated = 0

    for match in speeches:

        speech = data[match.start():match.end()]
        
        #print('String match "%s" at %d:%d' % (data[s:e], s, e))
        #print(count)
        

        #if count<110:
        #    continue

        #print('String match "%s" at %d:%d' % (data[s:e], s, e))
        #print(speech)
        #exit()
        #name = pName.findall(speech)[0]
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

        texts += 1
        if reuse:
            print("reuse existing JSON")
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
                print("translation already exists")
                text = existingData[count]["text"]
                if "translation_provider" not in existingData[count] or existingData[count]["translation_provider"] == "":
                    translationProvider = "azure"
                else:
                    translationProvider = existingData[count]["translation_provider"]
        else:
            print("no translation for reuse found, translate again")
            # no JSON file exists for this day
            if language == "en":
                text = speechText
                translationProvider = "none"
            else:        
                #try:
                text = translateText(speechText,language,"en", globalTranslationProvider)
                #except:
                #    print("An exception occurred while translating")
                #    text = ""
                translationProvider = globalTranslationProvider
        #print(speechText)
        #print(MEPId)
        
        #print(existingData[count]['name'])


        #if count>5:
        #    exit()
        
        #print(type(speechText))
        #print(speechText)
        
        #exit()

        #translator = Translator()


        

        #exit()
            
        print("{year}-{month}-{day} #{count}: {mepid} - {language} - {name}".format(year=year,month=month,day=day,count=count,mepid=MEPId,language=language,name=name))
        
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
        #print(speechObject)

        speechObjects.append(speechObject)

        if count % 50 == 0:
            JSONFilepath = os.path.join(JSONDir, JSONFilename)
            print("Write JSON file to {filepath}".format(filepath=JSONFilepath))
            with open(JSONFilepath, 'w') as outfile:
                dataEncoded = json.dumps(speechObjects, ensure_ascii=False)
                outfile.write(str(dataEncoded))

        #exit()
        #print(speechObject)
        #exit()
        
        # if count == 5:
        #     break
        count += 1

    #progressTracker.append(file)
    
    #saveJSON(texts,progressTrackerFilepath)

    JSONFilepath = os.path.join(JSONDir, JSONFilename)
    print("Write JSON file to {filepath}".format(filepath=JSONFilepath))
    with open(JSONFilepath, 'w') as outfile:
        dataEncoded = json.dumps(speechObjects, ensure_ascii=False)
        outfile.write(str(dataEncoded))
    print(count)

print("done")
    
    

    #print(speeches.next())

    #print(m.groups())
    #print(m.group(0))
    # if m:
    #     print('Match found: ', m.group())
    # else:
    #     print('No match')
    

