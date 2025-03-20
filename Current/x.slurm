#!/bin/bash

#SBATCH --job-name=h2_air                # 
#SBATCH --time=3:00:00                  #
#SBATCH --nodes=10
##SBATCH --ntasks=128                      #
#SBATCH --ntasks-per-node=4             # 
#SBATCH --mem=30G                      # 
#SBATCH --output=h2airOut.%j 
#SBATCH --error=amrex_error_%j.txt       # 

##SBATCH --account=132736325198             #Set billing account 
##SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=tugbakarasu@tamu.edu # 


cd /scratch/user/tugbakarasu/amrex/Tutorials/LRFPFCTEB_Exec/phi_calc/

module load GCC/11.3.0
module load OpenMPI/4.1.4


make -j10 > make_output.txt 2>&1

date
time mpirun -np 40 ./CNS2d.gnu.MPI.OMP.ex inputs
