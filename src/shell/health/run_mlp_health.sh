#!/bin/bash


export DATANAME="health"
export LOGS_FOLDER=result/logs/
export FOLDER=result/$DATANAME/mlp
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER

string=$(echo output_"$DATANAME"_mlp)
log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
result_folder=$(echo "$FOLDER")

python3 -m src.scripts.main -c config/config_mlp.py --exp_name "$FOLDER"/ --result_folder "$result_folder" --device cpu --data.name "$DATANAME" >"$log_file" 2>&1