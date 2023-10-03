#!/bin/bash

run_cp() {
	echo "Critical point run \"$3\" with l_simstart = $1, l_simeq = $2"
	sbatch -N 1 \
		--mem=4G \
		--cpus-per-task=16 \
		--time=2:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./point_run.sh $1 $2 $3
}

run_cs() {
	echo "Critical scaling run \"$3\" with l_simstart = $1, l_simeq = $2"
	sbatch -N 1 \
		--mem=4G \
		--cpus-per-task=24 \
		--time=12:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./scaling_run.sh $1 $2 $3
}

for L in {0..5}
do
	run_cs 0 $L run
	run_cs $L $L run
	run_cs "$L.25" "$L.25" run
	run_cs "$L.5" "$L.5" run
	run_cs "$L.75" "$L.75" run
done


# for L in {0..15}
# do
# 	# run_cp 0 $L fixw
# 	run_cp $L $L fixw
# 	run_cp "$L.25" "$L.25" fixw
# 	run_cp "$L.5" "$L.5" fixw
# 	run_cp "$L.75" "$L.75" fixw
# done

# run_cp 0 1 run
# run_cp 1 2 run
# run_cp 2 3 run
# run_cp 3 4 run
# run_cp 4 5 run
# run_cp 5 6 run
# run_cp 6 7 run
# run_cp 7 8 run
# run_cp 8 9 run
# run_cp 9 10 run
# run_cp 10 11 run
# run_cp 11 12 run
# run_cp 12 13 run
# run_cp 13 14 run
# run_cp 14 15 run


# Aggregate results into one file
# ls output/*.txt | xargs -n 1 -I % sh -c 'echo -n % " " ; grep "Final bounds" %' > runs.txt
