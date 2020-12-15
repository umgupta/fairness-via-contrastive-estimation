#!/bin/bash

# .run_<num> is important as it is used for generating and processing logs
function job(){
    echo "running for adult with coeff $1"
    python src/run_laftr.py conf/transfer/laftr_then_naive.json \
        -o exp_name="adult/laftr_g_"$1".run_1",train.n_epochs=500,model.fair_coeff="$1" \
        --data adult --dirs local > logs/adult/laftr_g_"$1".logs 2>&1
}
mkdir -p logs/adult/
export -f job
parallel -j3 job {1} ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0 2.5 3.0 3.5 4.0
