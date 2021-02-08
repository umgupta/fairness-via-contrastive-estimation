# We will use a model with decent performance and then change the params and finetune sequentially

export DATANAME="adult"
export LOGS_FOLDER=result/finetune/logs/
export FOLDER=result/finetune/$DATANAME/fcrl
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
device=cuda:0

statefile=result/adult/fcrl/l=0.01_b=0.005/run_0001/last_model.pt

max_epoch=200
epoch_to_train=20

for lambda in 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0;
do
  max_epoch=$(($max_epoch+$epoch_to_train))
  echo Running upto "$max_epoch"
  lambda=$(printf "%1.2f" "$lambda")
  beta=$(awk '{print $1*$2}' <<<"${lambda} 0.5")
  device=$device
  string=$(echo output_"$DATANAME"_fcrl_l="$lambda"_b="$beta")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/l="$lambda"_b="$beta")

  echo "Running NCE CC Supervised for beta=$beta lambda=$lambda"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"
  python3 -m src.scripts.main -c config/config_fcrl.py --exp_name "$FOLDER"/l="$lambda"_b="$beta" \
   --result_folder "$result_folder" --device "$device" --data.name "$DATANAME" --model.arch_file src/arch/adult_fcrl.py \
   --model.lambda_ "$lambda" --model.beta "$beta" --train.max_epoch "$max_epoch" --statefile $statefile >"$log_file" 2>&1
  statefile="$result_folder"/run_0001/last_model.pt

done
python3 -m src.scripts.eval_embeddings -D -f result/finetune/$DATANAME/fcrl -r result/eval/finetune/$DATANAME -m "fcrl" --force -c config/eval_config.py
