from mpi4py import MPI
import math
import sys
import numpy as np

design = str(sys.argv[1])
all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist() 
nStructures = len(all_IDs)

'''
Get data for historic file.
'''
def get_hist_info(ID):
    line_out = '' #Empty line for storing data to print in file   
    # Get summarizing files for each structure and aspect of interest from the .xdd or .xss files
    with open ('../../../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_hist.txt','w') as f:
        try:
            with open ('../../../LHsamples_original_1000_AnnQonly/cm2015B.xdd', 'rt') as xdd_file:
                for line in xdd_file:
                    data = line.split()
                    if data:
                        if data[0]==ID:
                            if data[3]!='TOT':
                                for o in [2, 4, 17]:
                                    line_out+=(data[o]+'\t')
                                f.write(line_out)
                                f.write('\n')
                                line_out = ''
            xdd_file.close()
            f.close()
        except IOError:
            f.write('-999.9 -999.9 -999.9')
            f.close()

    if ID == '7202003':
        with open ('../../../'+design+'/Infofiles/' +  ID + '/' + ID + '_streamflow_hist.txt','w') as f:
            try:
                with open ('../../../LHsamples_original_1000_AnnQonly/cm2015B.xdd', 'rt') as xdd_file:
                    for line in xdd_file:
                        data = line.split()
                        if data:
                            if data[0]==ID:
                                if data[3]!='TOT':
                                    for o in [2, 24]:
                                        line_out+=(data[o]+'\t')
                                    f.write(line_out)
                                    f.write('\n')
                                    line_out = ''
                xdd_file.close()
                f.close()
            except IOError:
                f.write('-999.9 -999.9')
                f.close()        

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
    
for i in range(start, stop):
    get_hist_info(all_IDs[i])