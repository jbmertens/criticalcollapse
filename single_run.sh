#!/bin/bash

module load SciPy-bundle/2022.05-foss-2022a

echo "l_simstart =" $1 "l_simeq =" $2
python3 -u critical_point.py $1 $2 > "output/run_$1_$2.tmp"
mv -f "output/run_$1_$2.tmp" "output/run_$1_$2.txt"
echo "Done!"
