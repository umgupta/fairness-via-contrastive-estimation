# FCRL
bash src/shell/adult/run_fcrl_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/adult/fcrl -r result/eval/adult -m "fcrl" --force -c config/eval_config.py

# CVIB
bash src/shell/adult/run_cvib_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/adult/cvib_supervised -r result/eval/adult -m "cvib" --force -c config/eval_config.py

# MAXENT
bash src/shell/adult/run_maxent_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/adult/maxent -r result/eval/adult -m "maxent_arl" --force -c config/eval_config.py


# ADV FORGETTING
bash src/shell/adult/run_adv_forgetting_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/adult/adv_forgetting -r result/eval/adult -m "adv_forgetting" --force -c config/eval_config.py


# You might like to add commands to switch environment here
# conda activate laftr
# LAFTR
cd laftr || exit
bash adult_experiments.sh
cd ..
python3 -m src.scripts.eval_embeddings -D -f laftr/experiments/adult/ -r result/eval/adult -m "laftr"  --force -c config/eval_config.py

# You might like to add commands to switch environment here
# conda activate lag-fairness
# LAG Fairness
cd lag-fairness || exit
bash adult_experiments.sh
cd ..
python3 -m src.scripts.eval_embeddings -D -f lag-fairness/result/fair/mifr_n/adult -r result/eval/adult -m "lag-fairness"  --force -c config/eval_config.py
