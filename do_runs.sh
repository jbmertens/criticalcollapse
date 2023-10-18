#!/bin/bash

run_cp() {
	echo "Critical point run \"$3\" with l_simstart = $1, l_simeq = $2"
	sbatch -N 1 \
		--mem=4G \
		--cpus-per-task=24 \
		--time=12:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./point_run.sh $1 $2 $3
}

run_cs() {
	echo "Critical scaling run \"$3\" with l_simstart = $1, l_simeq = $2"
	sbatch -N 1 \
		--mem=16G \
		--cpus-per-task=24 \
		--time=24:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./scaling_run.sh $1 $2 $3
}

run_cp 0 0 fixw
# run_cs 0 0 fixw

# Run both full QCD and fixed W
# for type in run
# do
# 	for L in 10
# 	do
# 		# Runs starting at l=0
# 		# run_cp 0 $L $type
# 		# run_cp 0 "$L.25" $type
# 		# run_cp 0 "$L.5" $type
# 		# run_cp 0 "$L.75" $type
# 		# Runs starting at l~lH
# 		run_cp $L $L $type
# 		run_cp "$L.25" "$L.25" $type
# 		run_cp "$L.5" "$L.5" $type
# 		run_cp "$L.75" "$L.75" $type
# 	done
# done


# Run both full QCD and fixed W
# for type in fixw
# do
# 	for L in {0..15}
# 	do
# 		# Runs starting at l~lH
# 		run_cs $L $L $type
# 		run_cs "$L.25" "$L.25" $type
# 		run_cs "$L.5" "$L.5" $type
# 		run_cs "$L.75" "$L.75" $type
# 	done
# done
