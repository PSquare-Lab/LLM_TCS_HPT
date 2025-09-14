# Redirect all output to a log file and the terminal
(
for i in {0..4}
do
    echo "###################### --- Run : $((i+1)) --- ######################"
    export TEMP_DIR="temp_$i"
    bash run.sh
    wait  # This is redundant unless run.sh is backgrounded
done
) | tee -a run_log.txt
