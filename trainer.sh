#!/bin/bash
# Datasets: FLICKR30K, FLICKR8K, COCO
# Seeds: 43, 947, 94743
# python main.py <SEED> <DATANAME> <TRAIN MODE> <RESUME> <STATS MODE>
declare -a Datasets=("FLICKR30K")
declare -a Seeds=("94743")
declare -a Resume=( "True")

for i in "${!Seeds[@]}";do
    for d in "${Datasets[@]}";do
        python main.py ${Seeds[$i]} $d "False" ${Resume[$i]} "latest"
    done
done