#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob, re
files = glob.glob('output/fixw_*.txt')
print(files)

run_data = []
for file in files :
    with open(file, 'r') as f:
        lines = f.readlines()

        # Get amplitude bounds
        lower_amp = 0.0
        upper_amp = 0.0
        l_simstart = -1.0
        l_simeq = -1.0
        for row in lines:
            if row.find('Final bounds are') != -1 :
                matches = re.findall(r"[0-9\.]+", row)
                lower_amp = float(matches[0])
                upper_amp = float(matches[1])
                l_simstart = float(matches[2])
                l_simeq = float(matches[3])
                break

        # Get deltaH and 
        deltaH = 0.0
        l = 0.0
        lH = 0.0
        after_upper = False
        for row in lines:
            if row.find('amplitude '+str(upper_amp)) != -1 :
                after_upper = True
            if after_upper and row.find('near l=') != -1 :
                matches = re.findall(r"l=[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                lH = float(matches[0][2:])
            if after_upper and row.find('3 c_double') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                deltaH = float(matches[3])
                l = float(matches[1])

        if after_upper :
            run_data.append([l_simstart, l_simeq, l, lH, deltaH, lower_amp, upper_amp])
        else :
            print("Error getting data from", file)

print(run_data)
