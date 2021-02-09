#!/bin/bash

function job() {
  # normal notation
  beta=$1
  lambda=$beta
  device=$2


  string=$(echo output_"$DATANAME"_fcrl_l="$lambda"_b="$beta")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/l="$lambda"_b="$beta")

  echo "Running FCRL for beta=$beta lambda=$lambda"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"

  python3 -m src.scripts.main -c config/config_fcrl.py \
    --exp_name "$FOLDER"/l="$lambda"_b="$beta" \
    --result_folder "$result_folder" --device "$device" --data.name "$DATANAME" \
    --model.arch_file src/arch/adult/adult_fcrl.py \
    --model.lambda_ "$lambda" --model.beta "$beta" \
    --train.max_epoch 200  >"$log_file" 2>&1
  #    --train.stopping_criteria loss --train.stopping_criteria_direction lower >"$log_file" 2>&1
}

export DATANAME="adult"
export LOGS_FOLDER=result/logs/
export FOLDER=result/ablatons/$DATANAME/fcrl
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
export DEVICE="cuda"
for beta in 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0
do
  job $beta $DEVICE
done

#export -f job
#parallel -j3 job {1} {2} ::: 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 ::: cuda:0
