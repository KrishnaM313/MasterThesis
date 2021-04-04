
import pandas as pd
import matplotlib.pyplot as plt
import os
from tools_data import getBaseDir, loadJSON
from tools_parties import fixParty, getPartyIdeology
from tqdm import tqdm
from tools_plot import plotGraph
from collections import Counter, OrderedDict
from tools_stats import addPercentage
from tools_latex import writeTable, toValueMatrix


if __name__ == '__main__':

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir,"data")
    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")
    plotsDir = os.path.join(baseDir,"plots")
    statsDir = os.path.join(baseDir,"stats")

    analysisSummaryDir = os.path.join(baseDir,"analysisSummary")


    files = os.listdir(JSONEnrichedDir)
    filePaths = []

    for file in files:
        filePath = os.path.join(JSONEnrichedDir, file)
        if os.path.isfile(filePath):
            filePaths.append(filePath)

    used_providers = []

    for file in tqdm(filePaths):
        speeches = loadJSON(file)
        for speech in speeches:
            provider = speech["translation_provider"]
            if provider == "":
                provider = "none"
            used_providers += [provider]

    provider_stats = Counter(used_providers)

    provider_stats = OrderedDict(sorted(provider_stats.items()))

    header = ["provider", "used (abs.)", "used (rel.)"]

    value_matrix = []
    for key in provider_stats:
        provider = key

        row = [
            provider, 
            provider_stats[key],  
            provider_stats[key]/len(used_providers)*100
            ]
        value_matrix += [row]

    print(value_matrix)
    providerStatsFile = os.path.join(statsDir,"provider_stats.tex")
    table = writeTable("Translation Providers Used",header, value_matrix,providerStatsFile)


    exit()
    for category in ["climate", "health"]:
        
        fileAnalysisPath = os.path.join(analysisSummaryDir, "{category}.csv".format(category=category))
        df = pd.read_csv(fileAnalysisPath)
        df["sum"] = df.drop(["politicalGroup","date"],axis=1).sum(axis=1)
        data = df[["politicalGroup","sum","date"]]

        for i, entry in enumerate(tqdm(df["politicalGroup"])):
            update = fixParty(entry)
            df.loc[i, ["fixedPoliticalGroup"]] = update
            df.loc[i, ["politicalGroupIdeology"]] = getPartyIdeology(update)
        
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        for key in ["fixedPoliticalGroup","politicalGroupIdeology"]:
            plotGraph(df.copy(), category, key, plotsDir, 
            startYear=2018, dropNA=False, verbose=True)
        
        

