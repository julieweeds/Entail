__author__ = 'Julie'

def configure(arguments):

    parameters={}
    parameters["on_apollo"]=True
    parameters["local"]=False
    parameters["at_home"]=False
    parameters["pairset"]="wn-noun-dependencies.json"
    parameters["freqset"]="entries_t2.strings"
    parameters["methods"]=[]
    parameters["use_cache"]=False
    parameters["simset"]="neighbours_t2.strings"

    for arg in arguments:
        if arg=="at_home":
            parameters["at_home"]=True
        elif arg=="local":
            parameters["local"]=True
        elif arg=="on_apollo":
            parameters["on_apollo"]=True
        elif arg=="zero_freq":
            parameters["methods"].append("zero_freq")
        elif arg=="freq":
            parameters["methods"].append("freq")
        elif arg=="lin_freq":
            parameters["methods"].append("lin_freq")
        elif arg=="use_cache":
            parameters["use_cache"]=True

    parameters=setfiles(parameters)
    return parameters

def setfiles(parameters):
    if parameters["at_home"]:
        parameters["datadir"]="C:/Users/Julie/Documents/Github/Entail/data/"
    if parameters["local"]:
        parameters["datadir"]="/Volumes/LocalScratchHD/juliewe/Documents/workspace/Entail/data/"

    parameters["pairfile"]=parameters["datadir"]+parameters["pairset"]
    parameters["freqfile"]=parameters["datadir"]+parameters["freqset"]
    parameters["simsfile"]=parameters["datadir"]+parameters["simset"]
    return parameters