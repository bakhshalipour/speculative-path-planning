#! /usr/bin/env bash

ASTAR_BINARY="speculative_astar.out"
if [[ ! -f $ASTAR_BINARY ]]; then
    echo "The binary does not exist"
    exit 1
fi

execute() {
    local EPSILON_VALUE=$1
    local THREAD_COUNT=$2
    local ENABLE_SPECULATION=$3
    local INPUT_MAP=$4
    local PATH_FILE="/dev/null"
    local CSV_FILE="/dev/null"

    local EXPERIMENT_SIG="<$INPUT_FILE, $EPSILON_VALUE, $THREAD_COUNT, $ENABLE_SPECULATION, $PATH_FILE, $CSV_FILE>"

    echo "Running $EXPERIMENT_SIG ..."

    SECONDS=0

    ./$ASTAR_BINARY $EPSILON_VALUE $THREAD_COUNT $ENABLE_SPECULATION $INPUT_MAP $PATH_FILE $CSV_FILE

    echo "`date` | Experiment: $EXPERIMENT_SIG | Exec Time: $SECONDS seconds"
    echo
    echo
}

date
echo "Experiment Signature: <INPUT_FILE, EPSILON_VALUE, THREAD_COUNT, ENABLE_SPECULATION, PATH_FILE, CSV_FILE>"

ALL_EPSIOLONS=(1 1000 1000000)
ALL_THREADS=(1 2 4 8 16 32)

for INPUT_FILE in inputset/*; do
    for EPSILON in ${ALL_EPSIOLONS[@]}; do
        for NUM_THREADS in ${ALL_THREADS[@]}; do
            for DO_SPECULATION in {0..1}; do
                if [[ $NUM_THREADS == 1 && $DO_SPECULATION == 1 ]]; then
                    echo "Skipped t$NUM_THREADS-s$DO_SPECULATION"
                    continue
                fi
                execute "$EPSILON" "$NUM_THREADS" "$DO_SPECULATION" "$INPUT_FILE"
                sleep 3
            done
        done
    done
done

wait

