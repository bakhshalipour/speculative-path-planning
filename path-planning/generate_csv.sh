#! /usr/bin/env bash

ASTAR_BINARY="speculative_astar.out"
if [[ ! -f $ASTAR_BINARY ]]; then
    echo "The binary does not exist"
    exit 1
fi

execute() {
    local EPSILON=$1
    local NUM_THREADS=$2
    local DO_SPECULATION=$3
    local INPUT_MAP=$4
    local PATH_FILE=$5
    local CSV_FILE=$6

    local EXPERIMENT_SIG="<$INPUT_FILE, $EPSILON, $NUM_THREADS, $DO_SPECULATION, $PATH_FILE, $CSV_FILE>"

    echo "Running $EXPERIMENT_SIG ..."

    SECONDS=0

    ./$ASTAR_BINARY $EPSILON $NUM_THREADS $DO_SPECULATION $INPUT_MAP $PATH_FILE $CSV_FILE

    echo "`date` | Experiment: $EXPERIMENT_SIG | Exec Time: $SECONDS seconds"
    echo
    echo
}

date
echo "Experiment Signature: <INPUT_FILE, EPSILON, NUM_THREADS, DO_SPECULATION, PATH_FILE, CSV_FILE>"

ALL_THREADS=(1 16)
EPSILON=1000

for INPUT_FILE in csv_input_maps/*; do
    for NUM_THREADS in ${ALL_THREADS[@]}; do
        for DO_SPECULATION in {0..1}; do

            if [[ $NUM_THREADS == 1 && $DO_SPECULATION == 1 ]]; then
                echo "Skipping t$NUM_THREADS-s$DO_SPECULATION"
                continue
            fi

            MAP_NAME=$INPUT_FILE
            MAP_NAME=${MAP_NAME%%.txt}
            MAP_NAME=${MAP_NAME##*/}

            PATH_FILE="path_files/t$NUM_THREADS-s$DO_SPECULATION"
            CSV_FILE="csv_files/t$NUM_THREADS-s$DO_SPECULATION"

            mkdir -p $PATH_FILE $CSV_FILE

            PATH_FILE="$PATH_FILE/$MAP_NAME.txt"
            CSV_FILE="$CSV_FILE/$MAP_NAME.csv"

            execute "$EPSILON" "$NUM_THREADS" "$DO_SPECULATION" "$INPUT_FILE" "$PATH_FILE" "$CSV_FILE"
            sleep 1

        done
    done
    exit
done

wait

