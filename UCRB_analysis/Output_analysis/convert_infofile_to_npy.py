import numpy as np
from mpi4py import MPI
import os
import sys
import math

design = str(sys.argv[1])
nsamples = int(sys.argv[2])

all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
nStructures = len(all_IDs)
nyears = 105
nrealizations = 10

def convert_infofile_to_npy(ID):
	npy_array = np.zeros([12*nyears,1+2*nrealizations,nsamples])
	for i in range(nsamples):
		npy_array[:,:,i] = np.loadtxt('../../../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_' + str(i+1) + '.txt')

	np.save('../../../'+design+'/Infofiles/' +  ID + '/' + ID +'_info.npy', npy_array)

	return None

# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

# Determine the chunk which each processor will neeed to do
count = int(math.floor(nStructures/nprocs))
remainder = nStructures % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
    start = rank*(count+1)
    stop = start + count + 1
else:
    start = remainder*(count+1) + (rank-remainder)*count
    stop = start + count
    
# Run simulation
for i in range(start, stop):
    convert_infofile_to_npy(all_IDs[i])