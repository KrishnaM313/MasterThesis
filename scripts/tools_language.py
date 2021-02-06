from translate import Translator
import requests
import textwrap
import googletrans
import translate
import boto3
import os
import time
import uuid


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
    translate = boto3.client(service_name='translate', region_name='us-east-2', use_ssl=True)
    parts = textwrap.wrap(text, 2000, break_long_words=False)
    translations = []
    for part in parts:
        result = translate.translate_text(Text=part, 
            SourceLanguageCode=from_lang, TargetLanguageCode=to_lang)
        translations.append(result.get('TranslatedText'))
        time.sleep(0.1)
    return ' '.join(translations)


def translateMyMemory(text,from_lang,to_lang):
    envVariable = "MY_MEMORY_EMAIL"
    if os.environ.get(envVariable) is not None:
        translator = Translator(from_lang=from_lang,to_lang=to_lang,email=os.environ.get(envVariable))
        return translator.translate(text)
    else:
        raise SystemExit("Please define environment variable for MyMemory translation provider: {envVariable}".format(envVariable=envVariable))


def translateGoogle(text,from_lang,to_lang):
    #proxy_selector = ProxySelector("109.107.200.151:8080") 

    translator = googletrans.Translator(raise_exception=True,proxies={'http': '109.107.200.151:8080'}, use_fallback=True)
    parts = textwrap.wrap(text, 4000, break_long_words=False)
    translations = []
    for part in parts:
        translation = translator.translate(part, src=from_lang, dest=to_lang)
        translations.append(translation.text)
    return ' '.join(translations)


def translateAzure(text,from_lang,to_lang):
    key_var_name = 'TRANSLATOR_TEXT_SUBSCRIPTION_KEY'
    if not key_var_name in os.environ:
        raise Exception('Please set/export the environment variable: {}'.format(key_var_name))
    subscription_key = os.environ[key_var_name]

    # endpoint_var_name = 'TRANSLATOR_TEXT_ENDPOINT'
    # if not endpoint_var_name in os.environ:
    #     raise Exception('Please set/export the environment variable: {}'.format(endpoint_var_name))
    # endpoint = os.environ[endpoint_var_name]
    endpoint = "https://api.cognitive.microsofttranslator.com/"

    location_name = 'TRANSLATOR_LOCATION'
    if not location_name in os.environ:
        raise Exception('Please set/export the environment variable: {}'.format(location_name))
    location = os.environ[location_name]
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.

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
        if request.status_code != 200:
            print(response)
        translations.append(response[0]["translations"][0]["text"])
    return ' '.join(translations)