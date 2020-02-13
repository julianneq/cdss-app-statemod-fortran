#!/bin/bash
#SBATCH -D /oasis/scratch/comet/jdq2101/temp_project/Colorado
#SBATCH --partition=compute
#SBATCH --nodes=1             # specify number of nodes
#SBATCH --ntasks-per-node=11  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 1:00:00            # set max wallclock time
#SBATCH --job-name="CMIP_unfinished" # name your job
#SBATCH --output="CMIP_unfinished.out"
#SBATCH --mail-user=jdq2101@gmail.com
#SBATCH --mail-type=ALL

module load python mpi4py scipy
# ibrun python runUnfinishedModels.py LHsamples_original_200_AnnQonly
# ibrun python runUnfinishedModels.py LHsamples_original_1000_AnnQonly
# ibrun python runUnfinishedModels.py LHsamples_narrowed_200_AnnQonly
# ibrun python runUnfinishedModels.py LHsamples_narrowed_1000_AnnQonly
# ibrun python runUnfinishedModels.py LHsamples_wider_200_AnnQonly
# ibrun python runUnfinishedModels.py LHsamples_wider_1000_AnnQonly
 ibrun python runUnfinishedModels.py CMIP_SOWs
# ibrun python runUnfinishedModels.py Paleo_SOWs
