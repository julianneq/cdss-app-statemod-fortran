#!/bin/bash
#SBATCH -D /oasis/scratch/comet/jdq2101/temp_project/Colorado/cdss-app-statemod-fortran/UCRB_analysis/Qgen
#SBATCH --partition=compute
#SBATCH --nodes=1             # specify number of nodes
#SBATCH --ntasks-per-node=7  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 1:00:00            # set max wallclock time
#SBATCH --job-name="gatherFlows" # name your job
#SBATCH --output="gatherFlows.out"
#SBATCH --mail-user=jdq2101@gmail.com
#SBATCH --mail-type=ALL

module load python
module load scipy/3.6
export MODULEPATH=/share/apps/compute/modulefiles/applications:$MODULEPATH
module load mpi4py
export MV2_ENABLE_AFFINITY=0
ibrun python3 gatherFlows.py