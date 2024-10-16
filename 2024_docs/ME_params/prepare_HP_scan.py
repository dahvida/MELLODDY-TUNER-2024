#!/usr/bin/env python3

import os
import shutil
import fileinput

# read hidden list, currently hard-coded to double layer setup
with open('../list_hidden_double.txt', 'r') as f:
    l_hidden = f.read().split('\n')[:-1]

# Loop over hyperparameter combinations and edit script
i = 0
for mid_dropout in ["0.4", "0.6", "0.8"]:
    for last_dropout in ["0.4", "0.6", "0.8"]:
        for hidden in l_hidden:
            i+=1
            num=str(i).zfill(3)

            # Remove spaces in hidden layers (for file name)
            hidden_name=hidden.replace(" ", "")
            run_name=f"Run_{num}_drop_{mid_dropout}_{last_dropout}_size_{hidden_name}"
        
            # Create script folder and create script
            os.mkdir(f"HP_scan/{run_name}")	
            filename=f"HP_scan/{run_name}/run_sparsechem.sh"

            # Read in template
            with open("run_sparsechem.TEMPLATE.sh", "rt") as template:

                # Open new script file
                with open(filename, "wt") as script:

                    # Replace placeholders for hyperparameters
                    for line in template:
                        script.write(line.replace("HIDDEN", hidden).replace("MIDDLE_DROPOUT", mid_dropout).replace("LAST_DROPOUT", last_dropout))

