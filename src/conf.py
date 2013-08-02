__author__ = 'Julie'

def configure(arguments):

    known_methods=["freq","zero_freq","lin_freq","CR","CR_thresh","clarke","clarke_thresh","invCL","lin"]
    parameters={}
    parameters["on_apollo"]=True
    parameters["local"]=False
    parameters["at_home"]=False
    parameters["pairset"]="wn-noun-dependencies.transform.json"
    parameters["freqset"]="entries_t10.strings"
    parameters["methods"]=[]
    parameters["use_cache"]=False
    parameters["simset"]="neighbours_t10.strings"
    #parameters["vectorset"]="events_t2.strings"
    parameters["vectorset"]="events_t10.pmi.strings"

    for arg in arguments:
        if arg=="at_home":
            parameters["at_home"]=True
            parameters["on_apollo"]=False
            parameters["local"]=False
        elif arg=="local":
            parameters["local"]=True
            parameters["on_apollo"]=False
            parameters["at_home"]=False
        elif arg=="on_apollo":
            parameters["on_apollo"]=True
            parameters["at_home"]=False
            parameters["local"]=False
        elif arg in known_methods:
            parameters["methods"].append(arg)
        elif arg == "all":
            for method in known_methods:
                parameters["methods"].append(method)
        elif arg=="use_cache":
            parameters["use_cache"]=True
        elif arg=="bless":
            parameters["pairset"]="BLESS_ent-pairs.json"
        else:
            print "Ignoring argument "+arg


    parameters=setfiles(parameters)
    return parameters

def setfiles(parameters):
    if parameters["at_home"]:
        parameters["datadir"]="C:/Users/Julie/Documents/Github/Entail/data/"
    if parameters["local"]:
        parameters["datadir"]="/Volumes/LocalScratchHD/juliewe/Documents/workspace/Entail/data/"
    if parameters["on_apollo"]:
        parameters["datadir"]="/mnt/lustre/scratch/inf/juliewe/Entail/data/"

    parameters["pairfile"]=parameters["datadir"]+parameters["pairset"]
    parameters["freqfile"]=parameters["datadir"]+parameters["freqset"]
    parameters["simsfile"]=parameters["datadir"]+parameters["simset"]
    parameters["vectorfile"]=parameters["datadir"]+parameters["vectorset"]
    return parameters