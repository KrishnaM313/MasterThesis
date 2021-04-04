import pytablewriter


def toValueMatrix(JSONText):
    matrix = []
    for key in JSONText:
        matrix += [key, JSONText[key]]
    return matrix


def writeTable(name, headers, value_matrix, dumpFilePath=None):
    writer = pytablewriter.LatexTableWriter()
    writer.table_name = name
    writer.headers = headers
    writer.value_matrix = value_matrix
    # writer.value_matrix = [
    #     [0,   0.1,      "hoge", True,   0,      "2017-01-01 03:04:05+0900"],
    #     [2,   "-2.23",  "foo",  False,  None,   "2017-12-23 45:01:23+0900"],
    #     [3,   0,        "bar",  "true",  "inf", "2017-03-03 33:44:55+0900"],
    #     [-10, -9.9,     "",     "FALSE", "nan", "2017-01-01 00:00:00+0900"],
    # ]
    if dumpFilePath is None:
        writer.write_table()
    else:
        writer.dump(dumpFilePath)
        writer.write_table()

