#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob, re
files = glob.glob('output/scaling_fixw_0_0.tm*')
print(files)

run_data = []
for file in files :
    with open(file, 'r') as f:
        lines = f.readlines()

        ls = []
        masses = []
        dHs = []

        # Get amplitude bounds
        for row in lines:
            if row.find('BH mass at horizon formation') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                if(len(matches)>0) :
                    masses.append(float(matches[0]))
            if row.find('3 c_double') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                if(len(matches)>4) :
                    dHs.append(float(matches[3]))
                    ls.append(float(matches[1]))

        if(len(ls) > 0 and len(dHs) > 0 and len(masses) > 0) :
            if(len(ls) == len(masses)) :
                matches = re.findall(r"[0-9\.]+", file)
                if matches[1][-1] == '.' :
                    matches[1] = matches[1][:-1]
                if len(matches) != 2 or matches[0] == '' or matches[1] == '' :
                    print("Error:", matches, file)
                    break
                run_data.append([ float(matches[0]), float(matches[1]), ls, dHs, masses ])
            else :
                print("Malformed data in file", file)
        else :
            print("No matches found for", file, ls, dHs, masses)

print(run_data)
