#!/usr/bin/env bash

# ./parallel_2d.sh map# t# s# t# s#

#map_number = $1
#threads_count_1 = $2
#speculation_1 = $3
#threads_count_2 = $4
#speculation_2 = $5

echo "./animate.py \\"
echo "  --map   ../path-planning/inputset/map${1}.txt \\"
echo "  --soln1 ../path-planning/csv_files/t${2}-s${3}/map${1}.csv \\"
echo "  --soln2 ../path-planning/csv_files/t${4}-s${5}/map${1}.csv \\"
echo "  --soln3 ../path-planning/csv_files/t${6}-s${7}/map${1}.csv"

./animate.py \
    --map   ../path-planning/inputset/map${1}.txt \
    --soln1 ../path-planning/csv_files/t${2}-s${3}/map${1}.csv \
    --soln2 ../path-planning/csv_files/t${4}-s${5}/map${1}.csv
    --soln3 ../path-planning/csv_files/t${6}-s${7}/map${1}.csv
