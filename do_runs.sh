#!/bin/bash

for l_simeq in {0..16}
do
	for l_simstart in 0 $(($l_simeq-1)) $l_simeq
	do
		# echo $l_simstart $l_simeq
		python3 Critical_scaling.py $l_simeq $l_simstart 400 > "output/run1_${l_simeq}_${l_simstart}.txt"
	done
	# wait $!
done


# Aggregate results into one file
# ls output_*.txt | xargs -n 1 -I % sh -c 'echo -n % " " ; grep Critical %' > runs.txt
