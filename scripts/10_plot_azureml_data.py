import os
from icecream import ic
from tools_plot import plotGraph
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
# Azure Connection Details
setup = False
azure = "private"
#modelName = "bert-base-uncased" # "openai-gpt" # 'bert-base-uncased'

if __name__ == '__main__':


    # if azure == "edu":
    #     subscription_id = "b0cfdf73-f3dd-4fd0-891a-c54130a3d181"
    #     resource_group = "master"
    #     workspace_name = "master1"
    #     computeSize = "small"
    # if computeSize == "small":
    #     computeResource = "master-gpu-12"
    # elif computeSize == "big":
    #     computeResource = "master-gpu-32-1" 

    # elif azure == "private":
    subscription_id = "93986b83-8c58-4738-abfd-f7d1cbcce9f8"
    resource_group = "master-privat"
    workspace_name = "master-privat"
    computeResource = "cluster-nd6"

    repoDir = "/Users/michael/workspaces/MasterThesis"

    dataPath = os.path.join(repoDir,"data")
    plotsPath = os.path.join(dataPath,"plots")



    from azureml.core.authentication import InteractiveLoginAuthentication

    interactive_auth = InteractiveLoginAuthentication(tenant_id="b232d827-1e67-4b06-b634-a6a6785fc4bf")

    # Establish Connection to Workspace
    from azureml.core import Dataset, Workspace, Experiment, Run

    ws = Workspace(subscription_id, resource_group, workspace_name, auth=interactive_auth)
    ws

    def getHDExperiment(workspace, category, labels, verbose=False):
        runName = "HyperParameter_"+category+"_"+labels
        if verbose:
            print("Load "+runName)
        experiment = Experiment(ws, name=runName)
        return experiment

    def getbestRunFromHD(experiment: Experiment, hDRunName: str) -> Run:
        run = Run(experiment,hDRunName)
        metrics = run.get_metrics(name="best_child_by_primary_metric")
        loss = metrics["best_child_by_primary_metric"]["metric_value"]
        bestRunNumber = loss.index(min(loss))
        bestRunID = metrics["best_child_by_primary_metric"]["run_id"][bestRunNumber]
        bestRun = Run(experiment,bestRunID)
        return bestRun


    #categories = ["climate", "health"]
    #labelsSelection = ["leftRightPosition", "partyGroupIdeology"]








    def addPlot(prefix,stage,metric, run, ax):
        xlabel="epochs"
        bestRunMetrics = run.get_metrics(name=prefix+stage+"_"+metric)[prefix+stage+"_"+metric]
        df = pd.DataFrame({
            prefix+stage+"_"+metric : bestRunMetrics,
            xlabel : range(len(bestRunMetrics))
        })
        ax.plot(df[xlabel], df[prefix+stage+"_"+metric], label=stage+"_"+metric)
        ax.legend()
        return ax

    def addStages(metric,bestRun,ax,prefix="epoch_"):
        for stage in ["train","test"]:
            ax = addPlot(prefix, stage, metric,bestRun,ax)
        return ax

    def getMetricPlot(metric,run,title=None,subplotID=0, show=False, plot=None):
        if plot is None:
            f, ax = plt.subplots(1)
        else:
            f, ax = plot
        ax[subplotID] = addStages(metric,run,ax[subplotID])
        if title is not None:
            plt.title(title)
        if show:
            plt.show()
        return f, ax

    def getMetricPlots(workspace: Workspace, category: str, labels: str, runNames: Dict, show=False, savePath=None):
        plt.clf()
        experiment = getHDExperiment(ws, category, labels)
        bestRun = getbestRunFromHD(experiment,runNames[category][labels])
        f, ax = plt.subplots(2)

        for i, metric in enumerate(["accuracy","avg_loss"]):
            f, ax = getMetricPlot(metric,bestRun, subplotID=i, plot=(f,ax))
        if savePath is not None:
            imgFilePath = os.path.join(savePath,"training_{}_{}.pdf".format(category, labels))
            plt.savefig(imgFilePath)
        if show:
            plt.show()
        return f,ax
    #exit()
    #getMetricPlot("avg_loss","Average Loss",bestRun,show=True)


    runNames = {
        "climate" : {
            "leftRightPosition" : "HD_71d73c5f-8d6a-4043-9f13-0080a6ec131a",
            "partyGroupIdeology" : "HD_40c5b23c-f0cb-4afe-bb2f-cb09c1b66eb4"
        },
        "health" : {
            "leftRightPosition" : "HD_4c9b0819-5cf5-45cc-a2a8-4c04107d2fa9",
            "partyGroupIdeology" : "HD_15204d49-52fc-465a-beac-104d6f32710c"
        }
    }


    #category = "health"
    #labels = "partyGroupIdeology"

    for category in ["health", "climate"]:
        for labels in ["leftRightPosition", "partyGroupIdeology"]:
            getMetricPlots(ws, category, labels, runNames, show=False, savePath=plotsPath)


    


    
    # plotting a bar graph


    #plt.clf()

    ##df.plot(**data)
    #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    #plt.legend(loc=(1.04,0))

    #plt.title(category)


    #trainRuns = trainExperiment.get_runs(tags={"thesis":1})
    #trainRun: Run = next(trainRuns)
    #print()
    #trainRunChildren = trainRun.get_children()
    #trainRunChildren: Run = next(trainRunChildren)
    #for i,run in enumerate(trainRunChildren):
    #    print(run)
    #print(trainRunChildren)
    #for trainRunChild in trainRunChildren:
    #    print(trainRunChild)
    #    exit()
    #print(trainRunChildren)
