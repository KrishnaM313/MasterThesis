
import pandas as pd
import matplotlib.pyplot as plt
import os
from tools_data import getBaseDir
from tools_parties import fixParty, getPartyIdeology
from tqdm import tqdm
from tools_plot import plotGraph

if __name__ == '__main__':

    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir,"data")
    JSONDir = os.path.join(baseDir,"json")
    JSONEnrichedDir = os.path.join(baseDir,"json_enriched")
    plotsDir = os.path.join(baseDir,"plots")

    analysisSummaryDir = os.path.join(baseDir,"analysisSummary")

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

        # Shorten Conservative democrats and conservatives category to fit plot
        df['politicalGroupIdeology'] = df['politicalGroupIdeology'].str.replace('Christian democrats and conservatives','Christian democ. and cons.')

        # Drop NA rows
        df = df.loc[df['politicalGroupIdeology'] != "na"]

        for key in ["fixedPoliticalGroup","politicalGroupIdeology"]:
            plotGraph(df.copy(), category, key, plotsDir,
            startYear=2018, dropNA=False, verbose=True, title=False)