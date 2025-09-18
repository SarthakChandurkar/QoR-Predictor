#!/bin/bash

OUTPUT_FILE="abc_stats.csv"
echo "Command, Delay, Area" > $OUTPUT_FILE

SUBCOMMANDS=("b" "rf" "rwz" "rfz" "rs -K 6" "rs -K 8" "rs -K 10" "rs -K 12" "rs -K 14" "rs -K 16" "b" "ftso" "fres")

# Function to generate a random command of size 20
generate_random_command() {
    RANDOM_COMMAND=""
    for i in {1..20}
    do
        RANDOM_COMMAND+="${SUBCOMMANDS[$RANDOM % ${#SUBCOMMANDS[@]}]}; "
    done
    echo $RANDOM_COMMAND
}

for i in {1..40}  # Run 10 iterations of random commands
do
    # Generate random command sequence
    RANDOM_CMD=$(generate_random_command)
    echo "Running command: $RANDOM_CMD"
    
    # Run the ABC tool with the random command sequence
    FINAL_DELAY=$(./abc -c "read_lib nangate_45.lib; read_bench simple_spi_orig.bench; st; $RANDOM_CMD; map; print_stats" | awk '/delay/ {print $(NF-3)}' | tr -d '=')
    FINAL_AREA=$(./abc -c "read_lib nangate_45.lib; read_bench simple_spi_orig.bench; st; $RANDOM_CMD; map; print_stats" | awk '/area/ {print $(NF-5)}' | tr -d '=')
    
    echo "Final Delay for $RANDOM_CMD: $FINAL_DELAY"
    echo "Final Area for $RANDOM_CMD: $FINAL_AREA"
    
    echo "$RANDOM_CMD ,$FINAL_DELAY ,$FINAL_AREA" >> $OUTPUT_FILE
done

echo "Stats have been saved to $OUTPUT_FILE"
