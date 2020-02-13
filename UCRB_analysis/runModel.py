from mpi4py import MPI
import numpy as np
import math
import os
import sys

# =============================================================================
# Experiment set up
# =============================================================================

realizations = 10
years = 105

design = str(sys.argv[1]) # experimental design passed to function call
LHsamples = np.loadtxt('./Qgen/' + design + '.txt') # corresponding samples
nSamples = np.shape(LHsamples)[0] + 1 # add one for stationarity sample (S0)

# List IDs of structures of interest for output files
IDs = np.genfromtxt('./Structures_files/metrics_structures.txt',dtype='str').tolist()

os.chdir(os.getcwd() + '/../../' + design)
#for ID in IDs: # create path for infofiles if it doesn't exist yet
#    if not os.path.exists('./Infofiles/' + ID):
#        os.makedirs('./Infofiles/' + ID)

# =============================================================================
# Define output extraction function
# =============================================================================
  
def getinfo(s):
    # create matrix to store shortage and demand for each structure (ID) in every realization of this SOW
    # initialize at -999.9 for realizations that failed
    info_matrix = np.zeros([len(IDs),years*12,2*realizations+1]) - 999.9
    Q15_matrix = np.zeros([years*12,realizations+1]) - 999.9 # store flow shortages in 15-mile reach

    # populate first column of info_matrix with year, which is the same across all structures
    info_matrix[:,0:3,0] = 1908 # first 3 months are in 1908
    info_matrix[:,-9::,0] = 2013 # last 9 months are in 2013
    x = np.tile(np.arange(1909,2013,1),12) # repeat 1909-2012 12 times
    info_matrix[:,3:-9,0] = np.transpose(np.reshape(x,[12,104])).flatten()

    # years for Q15_matrix are the same as for info_matrix
    Q15_matrix[:,0] = info_matrix[0,:,0]

    # loop through all realizations for this SOW and store demands and shortages in info_matrix and Q15_matrix
    for r in range(realizations):
        try:
            xdd_file = open('cm2015B_S' + str(s) + '_' + str(r+1) + '.xdd', 'rt')
            
            count = np.zeros([len(IDs)]) # keep track of how many months have been recorded for each structure in realization r
            
            for line in xdd_file.readlines():
                data = line.split()
                if data: # if line has something on it
                    if data[0] in IDs: # see if this is a structure of interest
                        if data[3] != 'TOT': # monthly demand and shortage, not total
                            index = IDs.index(data[0]) # find corresponding row of IDs for this structure
                            info_matrix[index,int(count[index]),r*2+1] = data[4] # demand in realization r
                            info_matrix[index,int(count[index]),r*2+2] = data[17] # shortage in realization r
                            if data[0] == '7202003': # 15-mile reach
                                Q15_matrix[int(count[index]),r+1] = data[24] # shortage in realization r

                            count[index]+=1 # increase number of months recorded for this structure in realization r
                    
            xdd_file.close()
        except:
            print('File cm2015B_S' + str(s) + '_' + str(r+1) + '.xdd not found')
            
    for i in range(len(IDs)): # write demands and shortages to a separate file for each structure
        np.savetxt('./Infofiles/' + IDs[i] + '/' + IDs[i] + '_info_' + str(s) + '.txt',info_matrix[i,:,:])

    np.savetxt('./Infofiles/7202003/7202003_streamflow_' + str(s) + '.txt',Q15_matrix)

    return None

# =============================================================================
# Start parallelization
# =============================================================================
    
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

# =============================================================================
# Loop though all SOWs of this experimental design
# =============================================================================
for s in range(start, stop): # loop through SOWs
    #for r in range(realizations): # loop through realizations
    #    os.system('./statemod cm2015B_S' + str(s) + '_' + str(r+1) + ' -simulate') # run model
    
    getinfo(s) # extract demand and shortage at structures of interest for this SOW in all realizations

#os.system('rm cm2015B_S*.xre ' +  'cm2015B_S*.xss ./../../' + 
#'cm2015B_S*.b*') # delete output files we no longer need (all realizations of this SOW)
