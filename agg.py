#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob, re

pattern = "fixw*.1.txt"


# Critical point data

files = glob.glob('output/'+pattern)
print("# CP Files:", files)

cp_data = []
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

        # Get deltaH, etc, from last (upper) run
        deltaH = 0.0
        amp = 0.0
        l = 0.0
        lH = 0.0
        for row in lines:
            if row.find('result: 3') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                l = float(matches[1])
                amp = float(matches[2])
                deltaH = float(matches[3])
        if amp > 0 :
            cp_data.append([l_simstart, l_simeq, l, lH, deltaH, lower_amp, upper_amp])
        else :
            print("# Error getting data from", file)

print("cp_data = np.array(", sorted(cp_data), ")")


# scaling amplitudes

files = glob.glob('output/scaling_'+pattern)
print("# CS Files:", files)

# Sub-critical scaling

cs_data = []
sc_data = []
run_ls = []
for file in files :
    with open(file, 'r') as f:
        lines = f.readlines()

        # Get super-critical amplitudes
        ls = []
        masses = []
        dHs = []
        amps = []
        for row in lines:
            if row.find('result: 3') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                if(len(matches)>5) :
                    ls.append(float(matches[1]))
                    amps.append(float(matches[2]))
                    dHs.append(float(matches[3]))
                    masses.append(float(matches[5]))

        # Get sub-critical amplitudes
        maxr0 = 0
        scls = []
        maxr0s = []
        scdHs = []
        scamps = []
        for row in lines:
            if row.find('result: 2') != -1 :
                matches = re.findall(r"[+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?", row)
                if(len(matches)>5) :
                    scls.append(float(matches[1]))
                    scamps.append(float(matches[2]))
                    scdHs.append(float(matches[3]))
                    maxr0s.append(float(matches[4]))

        # Append data
        if(len(scls) > 1 and len(scdHs) == len(scls) and len(maxr0s) == len(scls)
            and len(ls) > 1 and len(dHs) == len(ls) and len(masses) == len(ls) ) :

            # Append sorted data
            cs_data.append(sorted(zip(dHs, masses, amps, ls)))
            sc_data.append(sorted(zip(scdHs, maxr0s, scamps, scls)))

            # Run info from filename
            matches = re.findall(r"[0-9\.]+", file)
            if matches[1][-1] == '.' :
                matches[1] = matches[1][:-1]
            if len(matches) != 2 or matches[0] == '' or matches[1] == '' :
                print("# Error in file:", file, matches)
                break
            run_ls.append([float(matches[0]), float(matches[1])])

        else :
            print("# Not enough matches found for", file, ls, dHs, maxr0s)

print("scl = np.array(", run_ls, ")")
print("csd = [np.array(x) for x in ", cs_data, "]")
print("scd = [np.array(x) for x in ", sc_data, "]")
