import os
def no_success_bsub(server, path_file):
    content_1 = ''
    if server == 1:
        content_1 = f'''#!/bin/bash
#BSUB -J a_single_calculation
#BSUB -q 38
#BSUB -n 64
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -R "span[ptile=64]"

hostfile=`echo $LSB_DJOB_HOSTFILE`
NP=`cat $hostfile | wc -l`
cd $LS_SUBCWD
module load compiler/2022.1.0 mpi/2021.6.0 mkl/2022.2.0
export PATH=/work/phy-huangj/apps/il/abacus/3.6.5/bin:$PATH
COMMAND_std="mpirun -machinefile $LSB_DJOB_HOSTFILE -np $NP abacus"
'''
    elif server == 2:
        content_1 = f'''#!/bin/bash
#SBATCH --job-name a_single_calculation
#SBATCH --partition 256G56c
##SBATCH --nodelist c0[01-40]
#SBATCH --ntasks 56  #number of core
#SBATCH --qos=840cpu
#####
module load mkl/2022.1.0 mpi/2021.6.0 compiler/2022.1.0
export LD_LIBRARY_PATH=/share/home/wuyb/opts/apps/hdf5-1.10.9/lib:$LD_LIBRARY_PATH
conda activate abacus_env
ulimit -s unlimited
ulimit -l unlimited
export PATH=/share/home/wuyb/appbins/abacus-3.6.5/bin:$PATH
COMMAND_std="mpirun abacus"
'''

    content_2 = f'''$COMMAND_std > logout 2>&1'''

    with open(os.path.join(path_file, f'bsub.lsf'), 'w') as file:
        file.write(content_1 + content_2)
