# FCRL
bash src/shell/adult/run_fcrl_health.sh
python3 -m src.scripts.eval_embeddings -D -f result/health/fcrl -r result/eval/health -m "fcrl" --force -c config/eval_config.py

# CVIB
bash src/shell/health/run_cvib_health.sh
python3 -m src.scripts.eval_embeddings -D -f result/health/cvib_supervised -r result/eval/health -m "cvib" --force -c config/eval_config.py

# MAXENT
bash src/shell/health/run_maxent_health.sh
python3 -m src.scripts.eval_embeddings -D -f result/health/maxent -r result/eval/health -m "maxent" --force -c config/eval_config.py

# ADV FORGETTING
bash src/shell/health/run_adv_forgetting_health.sh
python3 -m src.scripts.eval_embeddings -D -f result/health/adv_forgetting -r result/eval/health -m "adv_forgetting" --force -c config/eval_config.py

# You might like to add commands to switch environment here
# conda activate lag-fairness
# LAG Fairness
cd lag-fairness || exit
bash health_experiments.sh
cd ..
python3 -m src.scripts.eval_embeddings -D -f lag-fairness/result/fair/mifr_n/health -r result/eval/health -m "lag-fairness"  --force -c config/eval_config.py
