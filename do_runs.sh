#!/bin/bash

run_cs() {
	echo "Starting l_simstart =" $1 "l_simeq =" $2
	sbatch -N 1 \
		--mem=4G \
		--cpus-per-task=16 \
		--time=2:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./single_run.sh $1 $2
}


# for L in {0..15}
# do
# 	run_cs 0 $L 
# 	run_cs $L $L
# 	run_cs "$L.25" "$L.25"
# 	run_cs "$L.5" "$L.5"
# 	run_cs "$L.75" "$L.75"
# done

# run_cs 0 1
# run_cs 1 2
# run_cs 2 3
# run_cs 3 4
# run_cs 4 5
# run_cs 5 6
# run_cs 6 7
# run_cs 7 8
# run_cs 8 9
# run_cs 9 10
# run_cs 10 11
# run_cs 11 12
# run_cs 12 13
# run_cs 13 14
# run_cs 14 15


# Aggregate results into one file
# ls output/*.txt | xargs -n 1 -I % sh -c 'echo -n % " " ; grep "Final bounds" %' > runs.txt
