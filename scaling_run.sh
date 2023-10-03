#!/bin/bash

module load SciPy-bundle/2022.05-foss-2022a

echo "Starting run \"$3\" with l_simstart = $1, l_simeq = $2"
python3 -u critical_scaling.py $1 $2 $3 > "output/scaling_$3_$1_$2.tmp"
mv -f "output/scaling_$3_$1_$2.tmp" "output/scaling_$3_$1_$2.txt"
echo "Done!"
