#!/bin/bash

#SBATCH --partition sched_mit_raffaele
#SBATCH --nodes 1
#SBATCH --ntasks 20
#SBATCH --exclusive
#SBATCH --mem 60000
#SBATCH --time=48:00:00
#SBATCH --error stderr
#SBATCH --output stdout
#SBATCH --job-name TrStats
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user hdrake@mit.edu
#SBATCH --no-kill
#SBATCH --nodelist=node264

module load engaging/anaconda

source activate sim-bbtre

SECONDS=0

jupyter nbconvert --to script 1_compute_tracer_moments.ipynb
python3 1_compute_tracer_moments.py > 1_compute_tracer_moments_stdout.txt
echo "Completed in $SECONDS seconds"

