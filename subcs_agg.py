#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob, re
files = glob.glob('output/scaling_fixw*')
print(files)

run_data = []
for file in files :
    with open(file, 'r') as f:
        lines = f.readlines()

        ls = []
        maxr0s = []
        dHs = []

        maxr0 = 0

        # Get amplitude bounds
        for row in lines:
            if row.find('max_rho0 was') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                if(len(matches)>0) :
                    maxr0 = float(matches[1])
            if row.find('2 c_double') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                if(len(matches)>4) :
                    dHs.append(float(matches[3]))
                    ls.append(float(matches[1]))
                    maxr0s.append(maxr0)

        if(len(ls) > 1) :
            matches = re.findall(r"[0-9\.]+", file)
            if matches[1][-1] == '.' :
                matches[1] = matches[1][:-1]
            if len(matches) != 2 or matches[0] == '' or matches[1] == '' :
                print("Error:", matches, file)
                break
            run_data.append([ float(matches[0]), float(matches[1]), ls, dHs, maxr0s ])
        else :
            print("Not enough matches found for", file, ls, dHs, maxr0s)

print(run_data)
