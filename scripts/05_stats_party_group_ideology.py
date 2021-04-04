

from tools_parties import getIdeologyID, getPartyIdeologyAssociations
from tools_latex import writeTable
from tools_data import getBaseDir, saveJSON
import os


if __name__ == '__main__':
    
    repoDir = getBaseDir()
    baseDir = os.path.join(repoDir,"data")
    statsDir = os.path.join(baseDir,"stats")

    associations = getPartyIdeologyAssociations()

    header = ["ideology", "parties"]

    value_matrix = []

    for ideology in associations:
        parties_string = ""
        
        parties = associations[ideology]

        for party in parties:
            if parties_string != "":
                prefix = ", "
            else:
                prefix = ""
            parties_string = parties_string + prefix + party
        
        value_matrix += [[ideology, parties_string]]


    texFilePath = os.path.join(statsDir,"ideology_party_associations.tex")
    table = writeTable("Party Ideology Associations",header, value_matrix,texFilePath)
    JSONFilePath = os.path.join(statsDir,"ideology_party_associations.json")
    saveJSON(associations,JSONFilePath)