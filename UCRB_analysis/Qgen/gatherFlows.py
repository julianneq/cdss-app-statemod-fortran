import numpy as np
from mpi4py import MPI
import math

# Experimental designs
designs = ['LHsamples_original_1000_AnnQonly','LHsamples_original_200_AnnQonly',\
           'LHsamples_narrowed_1000_AnnQonly','LHsamples_narrowed_200_AnnQonly',\
           'LHsamples_wider_1000_AnnQonly','LHsamples_wider_200_AnnQonly',\
           'CMIP_SOWs','CMIPunscaled_SOWs','Paleo_SOWs']
nSamples = [1000, 200, 1000, 200, 1000, 200, 209, 97, 366]

# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

# Determine the chunk which each processor will neeed to do
count = int(math.floor(len(designs)/nprocs))
remainder = len(designs) % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
	start = rank*(count+1)
	stop = start + count + 1
else:
	start = remainder*(count+1) + (rank-remainder)*count
	stop = start + count

# =============================================================================
# Loop through all SOWs of this experimental design
# =============================================================================
for j in range(start, stop):
    synthetic_flows = np.zeros([nSamples[j]*10, 105, 12])
    for s in range(nSamples[j]):
       for k in range(10):
           synthetic_file = open('../../../' + designs[j] + '/cm2015x_S'+str(s+1)+'_'+str(k+1)+'.xbm', 'r')
           all_split_data = [x.split('.') for x in synthetic_file.readlines()]
           yearcount = 0
           for i in range(16, len(all_split_data)):
               row_data = []
               row_data.extend(all_split_data[i][0].split())
               if row_data[1] == '09163500':
                   data_to_write = [row_data[2]]+all_split_data[i][1:12]
                   synthetic_flows[s*10+k,yearcount,:] = [int(n) for n in data_to_write]
                   yearcount+=1
                
    np.save(designs[j]+'_flows.npy', synthetic_flows)