import os
from icecream import ic
from tools_plot import plotGraph
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from azureml.core import (
    Experiment,
    Environment,
    ScriptRunConfig,
    Dataset,
    Workspace,
    Run
)
from tools_data import saveJSON
# Azure Connection Details
setup = False
azure = "private"
#modelName = "bert-base-uncased" # "openai-gpt" # 'bert-base-uncased'

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


def addPlot(prefix,stage,metric, run, ax,verbose=False,subplotID=0):
        xlabel="epochs"
        bestRunMetrics = run.get_metrics(name=prefix+stage+"_"+metric)[prefix+stage+"_"+metric]
        if verbose:
            print("got metric {}: {}".format(prefix+stage+"_"+metric,bestRunMetrics))
            #print(range(1,bestRunMetrics+1))
            print("Indices: {}".format([*range(1,len(bestRunMetrics)+1)]))

        if metric == "avg_loss":
            metricName = "Average Loss"
        else:
            metricName = metric.capitalize()
        

        df = pd.DataFrame({
            prefix+stage+"_"+metric : bestRunMetrics,
            xlabel : [*range(1,len(bestRunMetrics)+1)]
        })
        ax.plot(df[xlabel], df[prefix+stage+"_"+metric], label="{} {}".format(stage.capitalize(), metricName))
        
        plt.sca(ax)
        plt.xticks([*range(1,len(bestRunMetrics)+1)])
        ax.legend()
        return ax

def addStages(metric,bestRun,ax,prefix="epoch_",activeValidation=False,subplotID=0):
    stages = ["train","test"]
    if activeValidation:
        stages.append("validation")
    for stage in stages:
        ax = addPlot(prefix, stage, metric,bestRun,ax,subplotID=subplotID)
    return ax

def getMetricPlot(metric,run,title=None,subplotID=0, show=False, plot=None, activeValidation=False):
    if plot is None:
        f, ax = plt.subplots(1)
    else:
        f, ax = plot
    ax[subplotID] = addStages(metric,run,ax[subplotID],activeValidation=activeValidation,subplotID=subplotID)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
    return f, ax

def getMetricPlots(workspace: Workspace, category: str, labels: str, runID=None, runNames: Dict=None, show=False, savePath=None, activeValidation=False):
    plt.clf()
    experiment = getHDExperiment(ws, category, labels)
    if runNames is not None:
        run = getbestRunFromHD(experiment,runNames[category][labels])
    elif runID is not None:
        run = Run(experiment, runID)
    f, ax = plt.subplots(2)

    for i, metric in enumerate(["accuracy","avg_loss"]):
        f, ax = getMetricPlot(metric, run, subplotID=i, plot=(f,ax),activeValidation=activeValidation)
    if savePath is not None:
        for extension in ["pdf","png"]:
            imgFilePath = os.path.join(savePath,"training_{}_{}.{}".format(category, labels, extension))
            plt.savefig(imgFilePath)
    if show:
        plt.show()
    return run, f, ax

def getMetricsFromRun(run:Run,activeValidation=False):
    result = {
        "accuracy" : {
            "train" : run.get_metrics("final_train_accuracy"),
            "test" : run.get_metrics("final_test_accuracy"),
        },
        "loss" : {
            "train" : run.get_metrics("final_train_avg_loss"),
            "test" : run.get_metrics("final_test_avg_loss"),
        },
        "shares" : {
            "train" : run.get_metrics("share_train"),
            "test" : run.get_metrics("share_test")
        }   
    }
    result["dates"] = {}
    for stage in ["train","test","validation"]:
        result["dates"][stage] = {}
        for event in ["start", "end"]:
            result["dates"][stage][event] = run.get_metrics("date_"+stage+"_"+event)

    if activeValidation:
        result["accuracy"]["validation"] = run.get_metrics("final_validation_accuracy")
        result["loss"]["validation"] = run.get_metrics("final_validation_avg_loss")
    return result

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


    # HDrunNames = {
    #     "climate" : {
    #         "leftRightPosition" : "HD_71d73c5f-8d6a-4043-9f13-0080a6ec131a",
    #         "partyGroupIdeology" : "HD_40c5b23c-f0cb-4afe-bb2f-cb09c1b66eb4"
    #     },
    #     "health" : {
    #         "leftRightPosition" : "HD_4c9b0819-5cf5-45cc-a2a8-4c04107d2fa9",
    #         "partyGroupIdeology" : "HD_15204d49-52fc-465a-beac-104d6f32710c"
    #     }
    # }
    distributions = [
    {
        "train" : 0.80,
        "test" : 0.15
    },
    {
        "train" : 0.60,
        "test" : 0.10
    },
    {
        "train" : 0.90,
        "test" : 0.05
    }
]


    runIDs = {
        "climate" : {
            #"leftRightPosition" : "Train_climate_leftRightPosition_1618328387_f019d14a",
            "partyGroupIdeology" : "Train_climate_partyGroupIdeology_1618584360_1978a387"
        },
        "health" : {
            #"leftRightPosition" : "Train_health_leftRightPosition_1618328401_d4b41cca",
            "partyGroupIdeology" : "Train_health_partyGroupIdeology_1618584339_0b953964"
        }
    }


    #category = "health"
    #labels = "partyGroupIdeology"




    for category in ["health", "climate"]:
        for labels in ["partyGroupIdeology"]: #"leftRightPosition", 
            run, _, _ = getMetricPlots(ws, category, labels, runID=runIDs[category][labels], show=False, savePath=plotsPath, activeValidation=False)
            result = getMetricsFromRun(run, activeValidation=True)
            saveJSON(result, os.path.join(plotsPath,"training_{}_{}.json".format(category, labels)))
            plotsPath