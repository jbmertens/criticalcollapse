#!/bin/bash

run_cp() {
	echo "Critical point run \"$3\" with l_simstart = $1, l_simeq = $2"
	sbatch -N 1 \
		--mem=4G \
		--cpus-per-task=24 \
		--time=24:00:00 \
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

for type in run fixw
do
	for L in {0..15}
	do
		run_cp "$L.0" "$L.0" $type
		run_cp "$L.33" "$L.33" $type
		run_cp "$L.66" "$L.66" $type
	done
done

# Run both full QCD and fixed W
# for type in run fixw
# do
# 	for L in {0..15}
# 	do
#		run_cs $L $L $type
#		run_cs "$L.33" "$L.33" $type
#		run_cs "$L.66" "$L.66" $type
# 	done
# done
