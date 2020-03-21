#!/bin/bash
#SBATCH -D /oasis/scratch/comet/jdq2101/temp_project/Colorado/cdss-app-statemod-fortran/UCRB_analysis/Qgen
#SBATCH --partition=compute
#SBATCH --nodes=5             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 1:00:00            # set max wallclock time
#SBATCH --job-name="generateQs_LHsamples100" # name your job
#SBATCH --output="generateQs_LHsamples100.out"
#SBATCH --mail-user=jdq2101@gmail.com
#SBATCH --mail-type=ALL

module load python
module load scipy/3.6
export MODULEPATH=/share/apps/compute/modulefiles/applications:$MODULEPATH
module load mpi4py
export MV2_ENABLE_AFFINITY=0
ibrun python3 generateQs_spatialDisaggregation.py LHsamples_original_100_AnnQonly.txt
ibrun python3 generateQs_spatialDisaggregation.py LHsamples_wider_100_AnnQonly.txt