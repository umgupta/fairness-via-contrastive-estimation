
function job(){
    echo "running for health with coeff e1=$1 e2=$2"
    python -um examples.health_ours --mi 1.0 --e1 "$1" --e2 "$2" > logs/health/mifr_"$1"_"$2".logs 2>&1
}
mkdir -p logs/health/
export -f job
parallel -j5 job {1} {2} ::: 0.0 0.2 0.1 1.0 2.0 5.0 ::: 0.1 0.2 1.0 2.0 5.0
