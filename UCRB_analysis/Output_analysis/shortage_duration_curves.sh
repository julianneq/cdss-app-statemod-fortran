#!/bin/bash
#SBATCH -D /oasis/scratch/comet/jdq2101/temp_project/Colorado/cdss-app-statemod-fortran/UCRB_analysis/Output_analysis
#SBATCH --partition=compute
#SBATCH --nodes=15             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 6:00:00            # set max wallclock time
#SBATCH --job-name="shortage_duration_curves" # name your job
#SBATCH --output="shortage_duration_curves.out"
#SBATCH --mail-user=jdq2101@gmail.com
#SBATCH --mail-type=ALL

module load python
module load scipy/3.6
export MODULEPATH=/share/apps/compute/modulefiles/applications:$MODULEPATH
module load mpi4py
export MV2_ENABLE_AFFINITY=0
ibrun python3 shortage_duration_curves.py LHsamples_original_1000_AnnQonly
ibrun python3 shortage_duration_curves.py LHsamples_wider_1000_AnnQonly
ibrun python3 shortage_duration_curves.py LHsamples_original_100_AnnQonly
ibrun python3 shortage_duration_curves.py LHsamples_wider_100_AnnQonly
ibrun python3 shortage_duration_curves.py Paleo_SOWs
ibrun python3 shortage_duration_curves.py CMIPunscaled_SOWs
