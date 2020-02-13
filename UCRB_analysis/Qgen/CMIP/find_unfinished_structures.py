import numpy as np
import pandas as pd

completed = pd.read_csv('completed_sensitivities.txt',header=None)

deltas = []
deltas_conf = []
S1s = []
S1s_conf = []
R2s = []
structures = []
for i in range(len(completed)):
    if "_DELTA.csv" in completed.iloc[i][0]:
        deltas.append(i)
    if "_DELTA_conf.csv" in completed.iloc[i][0]:
        deltas_conf.append(i)
    if "_S1.csv" in completed.iloc[i][0]:
        S1s.append(i)
        structures.append(completed.iloc[i][0][0:-7])
    if "_S1_conf.csv" in completed.iloc[i][0]:
        S1s_conf.append(i)
    if "_R2.csv" in completed.iloc[i][0]:
        R2s.append(i)
        
all_IDs = np.genfromtxt('D:/GoogleDrive/Colorado/cdss-app-statemod-fortran/UCRB_analysis/Structures_files/metrics_structures.txt',dtype='str').tolist()

unfinished = []
for i in range(len(all_IDs)):
    if all_IDs[i] not in structures:
        unfinished.append(all_IDs[i])