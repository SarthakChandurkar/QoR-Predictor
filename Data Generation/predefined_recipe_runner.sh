#!/bin/bash

OUTPUT_FILE="abc_stats_small.csv"
echo "Command, Delay, Area" > $OUTPUT_FILE
COMMANDS=("resyn" "resyn2" "resyn2a" "resyn3" "compress" "compress2" "choice" "choice2" "rwsat" "drwsat2" "share" "addinit" "resyn2rs")


for cmd in "${COMMANDS[@]}"
do
    echo "Running command: $cmd"
    FINAL_DELAY=$(./abc -c "read_lib nangate_45.lib; read_bench simple_spi_orig.bench; $cmd; map; print_stats" | awk '/delay/ {print $(NF-3)}' | tr -d '=')
    FINAL_AREA=$(./abc -c "read_lib nangate_45.lib; read_bench simple_spi_orig.bench; $cmd; map; print_stats" | awk '/area/ {print $(NF-5)}' | tr -d '=')
    	

    echo "Final Delay for $cmd: $FINAL_DELAY"
    echo "Final Area for $cmd: $FINAL_AREA"
    
    
    echo "$cmd ,$FINAL_DELAY ,$FINAL_AREA" >> $OUTPUT_FILE


done

echo "Stats have been saved to $OUTPUT_FILE"

