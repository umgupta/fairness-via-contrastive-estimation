#!/bin/bash

function job() {
  beta=$1
  device=$2
  string=$(echo output_"$DATANAME"_maxent_arl_b="$beta")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/b="$beta")

  echo "Running maxent arl for beta=$beta"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"

  python3 -m src.scripts.main -c config/config_maxent_arl.py \
    --exp_name "$result_folder" \
    --result_folder "$result_folder" --device "$device" --data.name $DATANAME \
    --model.arch_file src/arch/adult/adult_maxent_arl_bn.py  \
    --model.beta "$beta" \
    --train.max_epoch 200 >"$log_file" 2>&1
}

export DATANAME="adult"
export LOGS_FOLDER=result/logs/
export FOLDER=result/$DATANAME/maxent_arl_bn
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
export DEVICE="cuda"

for beta in 1e-1 2e-1 5e-1 1e0 2e0 5e0 1e1 2e1 5e1 1e2
do
  job $beta $DEVICE
done

#if you have gnu parallel you can use below to speed up
#export -f job
#parallel -j5 job {1} {2} ::: 1e-1 2e-1 5e-1 1e0 2e0 5e0 1e1 2e1 5e1 1e2 ::: cuda:2

