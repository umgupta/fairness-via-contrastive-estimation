
function job(){
    echo "running for adult with coeff e1=$1 e2=$2"
    python -um examples.adult_ours --mi 1.0 --e1 "$1" --e2 "$2" > logs/adult/mifr_"$1"_"$2".logs 2>&1
}
mkdir -p logs/adult/
export -f job
parallel -j5 job {1} {2} ::: 0.0 0.2 0.1 1.0 2.0 5.0 ::: 0.1 0.2 1.0 2.0 5.0
