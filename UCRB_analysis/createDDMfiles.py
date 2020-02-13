import pandas as pd
import numpy as np
from mpi4py import MPI
import sys
import math

# Read/define relevant structures
transbasin = np.genfromtxt('./Structures_files/TBD.txt',dtype='str').tolist() # need to use their max (uncurtailed demand)
irrigation = np.genfromtxt('./Structures_files/irrigation.txt',dtype='str').tolist() # correlated with streamflows

# Get historical irrigation data 
with open('./Statemod_files/cm2015B.iwr','r') as f:
    hist_IWR = [x.split() for x in f.readlines()[463:]] # data starts on line 463   
f.close() 


# Get uncurtailed demands

# split data on periods (splitting on spaces/tabs doesn't work because some columns are next to each other)
with open('./Statemod_files/cm2015B.ddm','r') as f:
    all_split_data_DDM = [x.split('.') for x in f.readlines()]       
f.close()        
# get unsplit data to rewrite firstLine # of rows
with open('./Statemod_files/cm2015B.ddm','r') as f:
    all_data_DDM = [x for x in f.readlines()]       
f.close() 

# compute maximum historical TBD demands
max_values = pd.DataFrame(np.zeros([6,13]),index=transbasin) # 6 structures x 12 months + total
for i in range(len(all_split_data_DDM)-779): # data starts on line 779
    row_data = []
    row_data.extend(all_split_data_DDM[i+779][0].split())
    if row_data[1] in transbasin:
        current_values = max_values.loc[row_data[1]].values
        if float(row_data[2])>current_values[0]:
            current_values[0] = float(row_data[2])
        for j in range(len(all_split_data_DDM[i+779])-3):
            if float(all_split_data_DDM[i+779][j+1])>current_values[j+1]:
                current_values[j+1]=float(all_split_data_DDM[i+779][j+1])
        max_values.loc[row_data[1]]=current_values

for index, row in max_values.iterrows():
    row[12] = row.values[:-1].sum()

# Function for DDM files
def writenewDDM(design, nSamples, structures, firstLine):
    allstructures = []
    for m in range(len(structures)):
        allstructures.extend(structures[m])

    # Begin parallel simulation
    comm = MPI.COMM_WORLD

    # Get the number of processors and the rank of processors
    rank = comm.rank
    nprocs = comm.size

    # Determine the chunk which each processor will neeed to do
    count = int(math.floor(nSamples/nprocs))
    remainder = nSamples % nprocs

    # Use the processor rank to determine the chunk of work each processor will do
    if rank < remainder:
        start = rank*(count+1)
        stop = start + count + 1
    else:
        start = remainder*(count+1) + (rank-remainder)*count
        stop = start + count

    for n in range(start, stop):
        for r in range(10):
            with open('./../../' + design + '/cm2015B_S'+ str(n) + '_' + str(r+1) + '.iwr') as f:
                sample_IWR = [x.split() for x in f.readlines()[463:]]       
            f.close() 
    
            new_data = []
            irrigation_encounters = np.zeros(len(structures[0]))

            for i in range(len(all_split_data_DDM)-firstLine):
                row_data = []
                # To store the change between historical and sample irrigation demand (12 months + Total)
                change = np.zeros(13) 
                # Split first 3 columns of row on space
                # This is because the first month is lumped together with the year and the ID when spliting on periods
                row_data.extend(all_split_data_DDM[i+firstLine][0].split())
                # If the structure is not in the ones we care about then do nothing
                if row_data[1] in structures[0]: #If the structure is irrigation (correlated with streamflows)            
                    line_in_iwr = int(irrigation_encounters[structures[0].index(row_data[1])]*len(structures[0]) + structures[0].index(row_data[1]))
                    irrigation_encounters[structures[0].index(row_data[1])]=+1
                    for m in range(len(change)):
                        change[m]= float(sample_IWR[line_in_iwr][2+m])-float(hist_IWR[line_in_iwr][2+m])
                    # apply change to 1st month
                    row_data[2] = str(int(float(row_data[2])+change[0]))
                    # apply changes to rest of the columns
                    for j in range(len(all_split_data_DDM[i+firstLine])-2):
                        row_data.append(str(int(float(all_split_data_DDM[i+firstLine][j+1])+change[j+1])))
                elif row_data[1] in structures[1]: #If the structure is transbasin (to uncurtail)   
                    # apply max demand to 1st month
                    row_data[2] = str(int(max_values.loc[row_data[1]][0]))
                    # apply max demands to rest of the columns
                    for j in range(1,13):
                        row_data.append(str(int(max_values.loc[row_data[1]][j])))
                elif row_data[1] not in allstructures:
                    for j in range(len(all_split_data_DDM[i+firstLine])-2):
                        row_data.append(str(int(float(all_split_data_DDM[i+firstLine][j+1]))))                      
                # append row of adjusted data
                new_data.append(row_data)                
            # write new data to file
            f = open('./../../' + design + '/cm2015B_S' + str(n) + '_' + str(r+1) + '.ddm','w')
            # write firstLine # of rows as in initial file
            for i in range(firstLine):
                f.write(all_data_DDM[i])
            for i in range(len(new_data)):
                # write year, ID and first month of adjusted data
                f.write(new_data[i][0] + ' ' + new_data[i][1] + (19-len(new_data[i][1])-len(new_data[i][2]))*' ' + new_data[i][2] + '.')
                # write all but last month of adjusted data
                for j in range(len(new_data[i])-4):
                    f.write((7-len(new_data[i][j+3]))*' ' + new_data[i][j+3] + '.')                
                # write last month of adjusted data
                f.write((9-len(new_data[i][-1]))*' ' + new_data[i][-1] + '.' + '\n')            
            f.close()
    
    return None

design = str(sys.argv[1])
LHsamples = np.loadtxt('./Qgen/' + design + '.txt')
nSamples = np.shape(LHsamples)[0] + 1
writenewDDM(design, nSamples, [irrigation, transbasin], 779)
    

