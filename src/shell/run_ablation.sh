# FCRL
bash src/shell/ablations/run_fcrl_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/ablations/adult/fcrl -r result/eval/ablations/adult -m "fcrl" --force -c config/eval_config.py

# FCRL No Conditioning
bash src/shell/ablations/run_fcrl_no_conditioning_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/ablations/adult/fcrl_no_conditioning -r result/eval/ablations/adult -m "fcrl_no_conditioning" --force -c config/eval_config.py

# CVAE CC
bash src/shell/ablations/run_cvae_cc_supervised_adult.sh
python3 -m src.scripts.eval_embeddings -D -f result/ablations/adult/cvae_cc_supervised -r result/eval/ablations/adult -m "cvae_cc_supervised" --force -c config/eval_config.py

