import pickle
import datetime
import os
import re
import time

import numpy as np
import requests

from tools_data import downlodFile

downloadBaseFolder = "/home/user/workspaces/MasterThesis/data"


def downloadDocument(document, downloadBaseFolder, downloadedFiles):
    subfiles = document["formatDocs"]

    search2 = re.search(r"([\d]{2}-[\d]{2})", document["reference"])
    monthDay = search2.group()

    print("{}-{}: {}".format(str(year), monthDay, document["reference"]))

    for subfile in subfiles:
        if (subfile["typeDoc"] == "text/xml"):
            fileUrl = subfile["url"]
            extension = ".xml"
            fileName = monthDay + extension
            print("{}-{}".format(str(year), monthDay))
            filePath = os.path.join(originalDir, fileName)

            if fileName not in downloadedFiles:
                downlodFile(fileUrl, filePath)


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
        number=8,
        start="2014-01-01",
        end="2019-12-31"
    ),
    dict(
        number=9,
        start="2019-01-01",
        end="2024-12-31"
    )
]


downloadDir = os.path.join(downloadBaseFolder, "html")
if not os.path.exists(downloadDir):
    os.makedirs(downloadDir)

for term in terms:
    print(term["number"])
    start = datetime.datetime.strptime(term["start"], '%Y-%m-%d')
    end = min(datetime.datetime.today(),
              datetime.datetime.strptime(term["end"], '%Y-%m-%d'))
    delta = end - start

    xmlDownloadStateFilename = "notAvailableDocuments.txt"
    xmlDownloadState = os.path.join(downloadDir, xmlDownloadStateFilename)

    downloadedFiles = os.listdir(downloadDir)

    if os.path.exists(xmlDownloadState):
        with open(xmlDownloadState, "rb") as fp:   # Unpickling
            notAvailableDocuments = pickle.load(fp)
    else:
        notAvailableDocuments = []

    for day in range(delta.days + 1):
        date = start + datetime.timedelta(days=day)

        # https://www.europarl.europa.eu/doceo/document/CRE-8-2019-04-18_EN.html
        fileUrl = "https://www.europarl.europa.eu/doceo/document/CRE-%s-%s-%02d-%02d_EN.html" % (
            term["number"], date.year, date.month, date.day)
        localFilename = "%s-%02d-%02d.html" % (date.year, date.month, date.day)
        localPath = os.path.join(downloadDir, localFilename)

        # if localFilename not in notAvailableDocuments:
        try:
            downloadedFiles = os.listdir(downloadDir)
        except:
            print("Error downloading")
        print(fileUrl)
        result = downlodFile(fileUrl, localPath)
        if(result == 1):
            notAvailableDocuments.append(localFilename)
        time.sleep(0.1)

        with open(xmlDownloadState, "wb") as fp:  # Pickling
            pickle.dump(notAvailableDocuments, fp)


for year in years:
    print("Year {}".format(year), end=' ')
    response2 = requests.post(url2, data=(data2 % (str(year))), headers={
                              "Content-Type": "application/json"})
    documents = response2.json()["documents"]

    txtDir = os.path.join(downloadBaseFolder, "txt", str(year))
    if not os.path.exists(txtDir):
        os.makedirs(txtDir)

    originalDir = os.path.join(downloadBaseFolder, "html", str(year))
    if not os.path.exists(originalDir):
        os.makedirs(originalDir)

    downloadedFiles = os.listdir(originalDir)
    print(downloadedFiles)
    exit()
    for document in documents:
        downloadDocument(document, downloadBaseFolder, downloadedFiles)
