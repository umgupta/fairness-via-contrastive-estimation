#!/bin/bash

function job() {
  beta=$2
  lambda=$1
  device=$3
  string=$(echo output_"$DATANAME"_cvib_supervised_l="$lambda"_b="$beta")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/l="$lambda"_b="$beta")

  echo "Running CVIB Supervised for beta=$beta beta=$lambda"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"
  python3 -m src.scripts.main -c config/config_cvib_supervised.py \
    --exp_name "$FOLDER"/l="$lambda"_b="$beta" \
    --result_folder "$result_folder" --device "$device" --data.name "$DATANAME" \
    --model.arch_file src/arch/health/health_cvib_supervised.py \
    --model.lambda_ "$lambda" --model.beta "$beta" \
     --train.max_epoch 200 --train.batch_size 256  >"$log_file" 2>&1
  #    --train.stopping_criteria loss --train.stopping_criteria_direction lower >"$log_file" 2>&1
}

export DATANAME="health"
export LOGS_FOLDER=result/logs/
export FOLDER=result/$DATANAME/cvib_supervised
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
export DEVICE="cuda"

for lambda in  1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0
do
  for beta in 1e-3 1e-2 1e-1
  do
    job $lambda $beta $DEVICE
  done
done

# if you have gnu parallel you can use below to speed up
#export -f job
#parallel -j10 job {1} {2} {3} ::: 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 ::: 1e-3 ::: cuda:2
#parallel -j10 job {1} {2} {3} ::: 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 ::: 1e-1 ::: cuda:2
#parallel -j10 job {1} {2} {3} ::: 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 ::: 1e-2 ::: cuda:2

