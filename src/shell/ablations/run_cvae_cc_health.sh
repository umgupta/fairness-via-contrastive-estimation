#!/bin/bash

function job() {
  beta=$1
  lambda=$beta
  device=$2
  string=$(echo output_"$DATANAME"_cvae_cc_supervised_l="$lambda"_b="$beta")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/l="$lambda"_b="$beta")

  echo "Running CVAE CC Supervised for beta=$beta beta=$lambda"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"
  python3 -m src.scripts.main -c config/config_cvae_cc_supervised.py \
    --exp_name "$FOLDER"/l="$lambda"_b="$beta" \
    --result_folder "$result_folder" --device "$device" --data.name "$DATANAME" \
    --model.arch_file src/arch/health/health_cvae_cc_supervised.py \
    --model.lambda_ "$lambda" --model.beta "$beta" \
    --train.max_epoch 200 --train.batch_size 256 >"$log_file" 2>&1
  #    --train.stopping_criteria loss --train.stopping_criteria_direction lower >"$log_file" 2>&1
}

export DATANAME="health"
export LOGS_FOLDER=result/logs/
export FOLDER=result/ablations/$DATANAME/cvae_cc_supervised
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
export DEVICE="cuda"

for beta in 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0
do
  job $beta $DEVICE
done

# if you have gnu parallel you can use below to speed up
#export -f job
#parallel -j4 job {1} {2} ::: 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 ::: cuda:0
