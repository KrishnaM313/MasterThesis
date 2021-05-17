
import os
from tools_data import download, saveJSON
from tools_meps import reorderNames

baseDir = "/home/user/workspaces/MasterThesis/data"
filename = "mep_list"
mepInfoDir = os.path.join(baseDir, "meps")


if not os.path.exists(mepInfoDir):
    os.makedirs(mepInfoDir)


xlsInfo2014FilePath = os.path.join(mepInfoDir, filename + "_2014" + ".xls")
xlsInfo2014Url = "http://www.europarl.europa.eu/RegData/publications/lmeps/1979/0001/EP-PE_LMEPS(1979)0001_XL.xls"
xlsInfo2014 = download("xls", xlsInfo2014FilePath, xlsInfo2014Url)
xlsInfo2014 = reorderNames(xlsInfo2014)


xlsInfo2019FilePath = os.path.join(mepInfoDir, filename + "_2019" + ".xls")
xlsInfo2019Url = "http://www.europarl.europa.eu/RegData/publications/lmeps/2018/0002/EP-PE_LMEPS(2018)0002_XL.xlsx"
xlsInfo2019 = download("xls", xlsInfo2019FilePath, xlsInfo2019Url)
xlsInfo2019 = reorderNames(xlsInfo2019)


xmlDirectoryFilePath = os.path.join(mepInfoDir, filename + ".xml")
xmlDirectoryUrl = "https://www.europarl.europa.eu/meps/en/directory/xml?letter=&leg="
meps = download("xml", xmlDirectoryFilePath, xmlDirectoryUrl)


mepsByID = {}
mepsByName = {}

notfound = 0
found = 0

for mep in meps:

    entryByID = {}
    for key in mep:
        if key.tag == "fullName":
            name = key.text
            entryByID["name"] = name
            row = xlsInfo2014.loc[xlsInfo2014['name'] == name]
            row2 = xlsInfo2019.loc[xlsInfo2014['name'] == name]
            if len(row) == 1:
                entryByID["politicalGroup"] = row["Groupe politique*"].values[0]
                entryByID["nationalPoliticalGroup"] = row["Parti politique national*"].values[0]
                entryByID["country"] = row["État membre"].values[0]
                found += 1
            elif len(row2) == 1:
                entryByID["politicalGroup"] = row2["Groupe politique*"].values[0]
                entryByID["nationalPoliticalGroup"] = row2["Parti politique national*"].values
                entryByID["country"] = row2["État membre"].values[0]
                found += 1
            else:
                print("'{name}' not found in datasets".format(name=name))
                notfound += 1
                continue
        elif key.tag == "id":
            id = key.text
            entryByID["id"] = id
    mepsByID[id] = entryByID
    entryByName = id
    mepsByName[name] = entryByName

print("Found: {found}, Not found: {notfound}".format(
    found=found, notfound=notfound))

mepListFilepathByID = os.path.join(mepInfoDir, filename + "_by_id.json")
saveJSON(mepsByID, mepListFilepathByID)

mepListFilepathByName = os.path.join(mepInfoDir, filename + "_by_name.json")
saveJSON(mepsByName, mepListFilepathByName)
