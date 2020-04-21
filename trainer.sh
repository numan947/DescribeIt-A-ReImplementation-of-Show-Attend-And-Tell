#!/bin/bash
# Datasets: FLICKR30K, FLICKR8K, COCO
# Seeds: 43, 947, 94743
# python main.py <SEED> <DATANAME> <TRAIN MODE> <RESUME> <STATS MODE>
declare -a Datasets=("FLICKR30K" "FLICKR8K" "COCO")
declare -a Seeds=("94743" "947" "43")
declare -a Resume=( "Frue" "F" "F")

for i in "${!Seeds[@]}";do
    for d in "${Datasets[@]}";do
        python main.py ${Seeds[$i]} $d "False" ${Resume[$i]} "LATEST|BEST" "true"
    done
done