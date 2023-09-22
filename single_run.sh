#!/bin/bash

module load SciPy-bundle/2022.05-foss-2022a

echo "l_simeq =" $1 "l_simstart =" $2
python3 -u Critical_scaling.py $1 $2 > "output/run1_$1_$2.tmp"
mv -f "output/run1_$1_$2.tmp" "output/run1_$1_$2.txt"
echo "Done!"
