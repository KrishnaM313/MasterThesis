import os
import json
from fuzzy_match import algorithims
import time


import re
from tools_data import loadJSON, saveJSON
from tools_parties import extractFromName
from tools_meps import downloadMEPInfos, findMEP, findMEPName, findMEPParty
import numpy as np

healthAnalysis = None
healthAnalysisNA = 0
climateAnalysis = None
climateAnalysisNA = 0

if __name__ == '__main__':

    baseDir = "/home/user/workspaces/MasterThesis/data"


    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")


    files = os.listdir(JSONEnrichedDir)
    files.sort()


    parties = {}

    climate = {}

    for i,file in enumerate(files):

        pFileName = re.compile(r'(\d{4})-(\d{2})-(\d{2})', flags=re.DOTALL)
        FilenameExtraction = pFileName.match(file)
        year = FilenameExtraction.group(1)
        month = FilenameExtraction.group(2)
        day = FilenameExtraction.group(3)

        date = "{year}-{month}-{day}".format(year=year,month=month,day=day)


        filePath = os.path.join(JSONEnrichedDir,file)
        data = loadJSON(filePath)

        for n,speech in enumerate(data):
            ### Parties
            if "politicalGroup" in speech:
                if speech["politicalGroup"] in parties:
                    parties[speech["politicalGroup"]] += 1
                else:
                    parties[speech["politicalGroup"]] = 1
            else:
                if "none" in parties:
                    parties["none"] += 1
                else:
                    parties["none"] = 1
        
        if healthAnalysis is None:
            if "health" in speech:
                healthAnalysis = np.array(speech["health"])
            else:
                healthAnalysisNA += 1
        else:
            if "health" in speech:
                healthAnalysis = healthAnalysis + np.array(speech["health"])
            else:
                healthAnalysisNA += 1
        
        if climateAnalysis is None:
            if "climate" in speech:
                climateAnalysis = np.array(speech["climate"])
            else:
                climateAnalysisNA += 1
        else:
            if "climate" in speech:
                climateAnalysis = climateAnalysis + np.array(speech["climate"])       
            else:
                climateAnalysisNA += 1

        

        
    

    print(parties)
    print(healthAnalysis)
    print(climateAnalysis)

