from json import dump
import os
from tools_latex import writeTable
from tools_data import loadJSON, saveJSON

if __name__ == '__main__':
    repoDir = "/Users/michael/workspaces/MasterThesis"

    dataPath = os.path.join(repoDir,"data")
    plotsPath = os.path.join(dataPath,"plots")
    statsPath = os.path.join(dataPath,"stats")

    models = ["training","forest"]
    labels = ["partyGroupIdeology"]
    categories = ["climate","health"]
    stages = ["train","test","validation"]



    dictPath = os.path.join(dataPath,"dictionaries")

    filePath = os.path.join(dictPath, "keywords.json")

    

    keywords = loadJSON(filePath)

    header = [categories[0],categories[1]]
    length = max(len(keywords[categories[0]]),len(keywords[categories[1]]))

    table = []
    for i in range(length):
        row = []
        for category in categories:
            if i < len(keywords[category]):
                row.append(keywords[category][i])
            else:
                row.append("")
        table.append(row)


    filePath=os.path.join(statsPath,"keyword_dictionaries")
    saveJSON([header,table],filePath+".json")
    writeTable("Keyword Dictionaries",header,table,dumpFilePath=filePath+".tex")
    print(table)
    exit()



    model = models[0]
    label = labels[0]
    category = categories[0]
    stage = stages[0]

    forest = loadJSON(os.path.join(plotsPath,"{}_{}_{}.json".format("forest",category,label)))
    bert = loadJSON(os.path.join(plotsPath,"{}_{}_{}.json".format("training",category,label)))

    table = []
    header = ["Model","Training","Testing","Validation"]

    row = [
        "RandomForest"
    ]
    for stage in stages:
        row.append(round(forest["accuracy"][stage]["final_{}_accuracy".format(stage)],2))
    table.append(row)

    row = [
        "fine-tuned BERT"
    ]
    for stage in stages:
        row.append(round(bert["accuracy"][stage]["final_{}_accuracy".format(stage)],2))
    table.append(row)

    filePath=os.path.join(statsPath,"model_accuracy_comparison")
    saveJSON([header,table],filePath+".json")
    writeTable("Model Accuracy",header,table,dumpFilePath=filePath+".tex")
    print(table)





    model = models[0]
    label = labels[0]
    category = categories[0]

    result = loadJSON(os.path.join(plotsPath,"{}_{}_{}.json".format(model,category,label)))

    header = ["stage", "start","end","share"]

    result["shares"]["validation"] = {
        "share_validation" : 1 - result["shares"]["train"]["share_train"] - result["shares"]["test"]["share_test"]
    }

    table = []
    for stage in stages:
        table.append([
            stage.capitalize(),
            result["dates"][stage]["start"]["date_{stage}_start".format(stage=stage)],
            result["dates"][stage]["end"]["date_{stage}_end".format(stage=stage)],
            str(round(result["shares"][stage]["share_{stage}".format(stage=stage)]*100))+"%"
        ])

    filePath=os.path.join(statsPath,"data_plot")
    saveJSON([header,table],filePath+".json")
    writeTable("Dataset Split",header,table,dumpFilePath=filePath+".tex")
    print(table)




# def writeTable(name, headers, value_matrix, dumpFilePath=None):
#     writer = pytablewriter.LatexTableWriter()
#     writer.table_name = name
#     writer.headers = headers
#     writer.value_matrix = value_matrix
#     # writer.value_matrix = [
#     #     [0,   0.1,      "hoge", True,   0,      "2017-01-01 03:04:05+0900"],
#     #     [2,   "-2.23",  "foo",  False,  None,   "2017-12-23 45:01:23+0900"],
#     #     [3,   0,        "bar",  "true",  "inf", "2017-03-03 33:44:55+0900"],
#     #     [-10, -9.9,     "",     "FALSE", "nan", "2017-01-01 00:00:00+0900"],
#     # ]
#     if dumpFilePath is None:
#         writer.write_table()
#     else:
#         writer.dump(dumpFilePath)
#         writer.write_table()
