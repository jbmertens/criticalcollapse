#!/bin/bash

run_cp() {
	echo "Critical point run \"$3\" with l_simstart = $1, l_simeq = $2, N = $4"
	sbatch -N 1 \
		--mem=4G \
		--cpus-per-task=20 \
		--time=24:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./point_run.sh $1 $2 $3 $4
}

run_cs() {
	echo "Critical scaling run \"$3\" with l_simstart = $1, l_simeq = $2, N = $4"
	sbatch -N 1 \
		--mem=16G \
		--cpus-per-task=20 \
		--time=24:00:00 \
		--output="slurm_out/$1-$2.out" \
		--job-name "critical_collapse_$1_$2" \
		./scaling_run.sh $1 $2 $3 $4
}

N=8192
# run_cp "0.1" "0.1" fixw $N
# run_cp "5.1" "5.1" fixw $N
# run_cp "10.1" "10.1" fixw $N
# run_cp "15.1" "15.1" fixw $N
# run_cs "0.0" "0.0" run

for type in run fixw
do
	for L in {0..15}
	do
		run_cp "$L.0" "$L.0" $type $N
		run_cp "$L.33" "$L.33" $type $N
		run_cp "$L.66" "$L.66" $type $N
	done
done
