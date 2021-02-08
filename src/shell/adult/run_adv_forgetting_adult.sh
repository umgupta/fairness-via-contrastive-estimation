#!/bin/bash

# rho is for distortion
# delta for discriminator loss
# lambda for x(1-x) mask regularization

function job() {
  delta=$1
  rho=$2
  lambda=$3
  device=$4

  string=$(echo output_"$DATANAME"_adv_forgetting_l="$lambda"_d="$delta"_r="$rho")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/l="$lambda"_d="$delta"_r="$rho")

  echo "Running adversarial forgetting for lambda=$lambda, rho=$rho, delta=$delta"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"

  python3 -m src.scripts.main -c config/config_adv_forgetting.py \
    --exp_name "$result_folder" \
    --result_folder "$result_folder" --device "$device" --data.name "$DATANAME" \
    --model.arch_file src/arch/adult/adult_adv_forgetting.py \
    --model.lambda "$lambda" --model.rho "$rho" --model.delta "$delta" \
    --train.max_epoch 200  >"$log_file" 2>&1
}

export DATANAME="adult"
export LOGS_FOLDER=result/logs/
export FOLDER=result/$DATANAME/adv_forgetting
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
export DEVICE="cuda"

for rho in 1e-3 1e-2 1e-1
do
  for lambda in 1e-3 1e-2 1e-1
  do
    for delta in  1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1
    do
      job $delta $rho $lambda $DEVICE
    done
  done
done

# if you have gnu parallel you can use below to speed up
#export -f job
##                                                           delta                   ::: rho  ::: lambda ::: device
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-3 ::: 1e-2 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-3 ::: 1e-3 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-2 ::: 1e-3 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-2 ::: 1e-2 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-2 ::: 1e-1 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-3 ::: 1e-1 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-1 ::: 1e-3 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-1 ::: 1e-2 ::: cuda:2
#parallel -j10 job {1} {2} {3} {4} ::: 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0 1e1 5e1 ::: 1e-1 ::: 1e-1 ::: cuda:2
