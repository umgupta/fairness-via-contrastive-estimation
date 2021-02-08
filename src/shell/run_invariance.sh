bash src/shell/adult/run_maxent_bn_adult.sh
python3 -m src.scripts.eval_invariance_embeddings -D -f result/adult/maxent_arl_bn -r result/eval/invariance/adult -m "maxent_arl_bn" --force -c config/eval_invariance_config.py

python3 -m src.scripts.eval_invariance_embeddings -D -f result/adult/maxent_arl -r result/eval/invariance/adult -m "maxent_arl" --force -c config/eval_invariance_config.py
python3 -m src.scripts.eval_invariance_embeddings -D -f result/adult/adv_forgetting -r result/eval/invariance/adult -m "adv_forgetting" --force -c config/eval_invariance_config.py

