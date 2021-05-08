from tools_parties import getPartyIdeologyAssociations
from tools_latex import writeTable
from tools_data import getBaseDir
import os


ideologyPartyAssoiation = getPartyIdeologyAssociations()
print(ideologyPartyAssoiation)

header = ["Ideology Category", "Parties"]

value_matrix = []
for ideology in ideologyPartyAssoiation:
    partyString = None
    for party in ideologyPartyAssoiation[ideology]:
        if partyString is not None:
            partyString = "{}, {}".format(partyString,party)
        else:
            partyString = party
    row = [ideology,partyString]
    value_matrix.append(row)

print(value_matrix)

repoDir = getBaseDir()
baseDir = os.path.join(repoDir,"data")
statsDir = os.path.join(baseDir,"stats")
outputFile = os.path.join(statsDir,"party_associations.tex")

table = writeTable("Ideology Party Associations",header, value_matrix,outputFile)
