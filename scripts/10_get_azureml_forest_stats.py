
import os

from tools_data import saveJSON

if __name__ == '__main__':

    subscription_id = "###"
    resource_group = "###"
    workspace_name = "###"
    computeResource = "cluster-nd6"

    repoDir = "/Users/michael/workspaces/MasterThesis"

    dataPath = os.path.join(repoDir, "data")
    plotsPath = os.path.join(dataPath, "plots")

    from azureml.core.authentication import InteractiveLoginAuthentication

    interactive_auth = InteractiveLoginAuthentication(
        tenant_id="b232d827-1e67-4b06-b634-a6a6785fc4bf")

    # Establish Connection to Workspace
    from azureml.core import Dataset, Experiment, Run, Workspace

    ws = Workspace(subscription_id, resource_group,
                   workspace_name, auth=interactive_auth)
    ws

    runIDs = {
        "climate": {
            "partyGroupIdeology": "Forest_climate_partyGroupIdeology_1618608814_a9f93839"
        },
        "health": {
            "partyGroupIdeology": "Forest_health_partyGroupIdeology_1618608825_f7446b28"
        }
    }

    stages = ["train", "test", "validation"]

    for category in runIDs.keys():
        for labels in runIDs[category].keys():
            runID = runIDs[category][labels]
            experiment = Experiment(
                ws, "Forest_{}_{}".format(category, labels))
            run = Run(experiment, runID)

            result = {}

            # accuracy
            result["accuracy"] = {}
            for stage in stages:
                result["accuracy"][stage] = run.get_metrics(
                    "final_"+stage+"_accuracy")

            result["shares"] = {}
            for stage in stages:
                result["shares"][stage] = run.get_metrics("share_"+stage)

            result["dates"] = {}
            for stage in stages:
                result["dates"][stage] = {}
                for event in ["start", "end"]:
                    result["dates"][stage][event] = run.get_metrics(
                        "date_"+stage+"_"+event)
            saveJSON(result, os.path.join(
                plotsPath, "forest_{}_{}.json".format(category, labels)))
