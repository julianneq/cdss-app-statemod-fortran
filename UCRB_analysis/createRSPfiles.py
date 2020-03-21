from string import Template
from mpi4py import MPI
import sys
import numpy as np
import math

design = str(sys.argv[1])
LHsamples = np.loadtxt('./Qgen/' + design + '.txt')
nSamples = np.shape(LHsamples)[0] + 1

# load data
template_name = 'cm2015B_template.rsp'

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

# create RSP files
with open(template_name, 'r') as T:
    template = Template(T.read())
    for i in range(start, stop):
        for j in range(10):
            d = {}
            d['IWR'] = 'cm2015B_S' + str(i) + '_' + str(j+1) + '.iwr'
            d['XBM'] = 'cm2015x_S' + str(i) + '_' + str(j+1) + '.xbm'
            d['DDM'] = 'cm2015B_S' + str(i) + '_' + str(j+1) + '.ddm'
            S1 = template.safe_substitute(d)
            with open('./../../' + design + '/cm2015B_S' + str(i) + '_' + str(j+1) + '.rsp', 'w') as f1:
                f1.write(S1)
                
            f1.close()