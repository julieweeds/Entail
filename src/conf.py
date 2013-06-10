__author__ = 'Julie'

def configure(arguments):

    parameters={}
    parameters["on_apollo"]=True
    parameters["local"]=False
    parameters["at_home"]=False
    parameters["pairset"]="wn-noun-dependencies.json"
    parameters["freqset"]="entries.totals"

    for arg in arguments:
        if arg=="at_home":
            parameters["at_home"]=True

    parameters=setfiles(parameters)
    return parameters

def setfiles(parameters):
    if parameters["at_home"]:
        parameters["datadir"]="C:/Users/Julie/Documents/Github/Entail/data/"

    parameters["pairfile"]=parameters["datadir"]+parameters["pairset"]
    parameters["freqfile"]=parameters["datadir"]+parameters["freqset"]
    return parameters