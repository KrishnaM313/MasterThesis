import os
import re
#from googletrans import Translator
from langdetect import detect
from translate import Translator
import json
import uuid
import requests
import textwrap


baseDir = "/home/user/workspaces/MasterThesis/data" 

htmlDir = os.path.join(baseDir, "html")
JSONDir = os.path.join(baseDir, "json")

availableJSONFiles = os.listdir(JSONDir)


if not os.path.exists(JSONDir):
    os.makedirs(JSONDir)


files = os.listdir(htmlDir)


key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
if not key_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
subscription_key = os.environ[key_var_name]

endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
if not endpoint_var_name in os.environ:
    raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
endpoint = os.environ[endpoint_var_name]


def translate(text,from_lang,to_lang):
    
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
        translations.append(response[0]["translations"][0]["text"])
        
    #print(' '.join(translations))
    return ' '.join(translations)


    # You can pass more than one object in body.
    

    

    #print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))
    #print(response)
    



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
  
    if JSONFilename in availableJSONFiles:
        
        print("skipped because already translated")
        continue
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


    for match in speeches:
        

        speech = data[match.start():match.end()]
        
        #print('String match "%s" at %d:%d' % (data[s:e], s, e))
        #print(count)
        count += 1

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
        
        #print(type(speechText))
        #print(speechText)
        
        #exit()

        #translator = Translator()
        try:
            language = detect(speechText)
        except:
            continue

        

        if language == "en":
            text = speechText
        else:
            #text = translator.translate(speechText, dest='en').text
            #translator2 = Translator(provider='microsoft', to_lang=language, secret_access_key=os.environ.get('TRANSLATION_API_KEY'))
            #print(translator.provider)
            #text = translator.translate(speechText)
            #print(text)            
            #try:
            text = translate(speechText,language,"en")
            #except:
            #    print("An exception occurred while translating")
             #   text = ""
            
        print("#{count}: {mepid} - {language} - {name}".format(count=count,mepid=MEPId,language=language,name=name))
        
        speechObject = {
            "name" : name,
            "mepid" : MEPId,
            "text" : text,
            "language" : "en"
        }
        #print(speechObject)
        #exit()
        speechObjects.append(speechObject)
        # if count == 5:
        #     break

    
    JSONFilepath = os.path.join(JSONDir, JSONFilename)
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

