#!/bin/bash

for EXPONENT in {0..20}
do
	for fixw in 0 1 2
	do
		for AMP in 1.0 1.259 1.585 1.995 2.511 3.162 3.981 5.012 6.310 7.943
		do
			NUMBER="${AMP}e${EXPONENT}"
			ipython3 Critical_scaling.py $NUMBER $fixw > "output_${fixw}_${NUMBER}.txt" &
		done
		wait $!
	done
done


# Aggregate results into one file
# ls output_*.txt | xargs -n 1 -I % sh -c 'echo -n % " " ; grep Critical %' > runs.txt
