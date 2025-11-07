#!/bin/bash
#SBATCH --account=MST114281                             # parent project to access twcc system
#SBATCH --job-name=train_3dfullres                        # jobName
#SBATCH --nodes=1                                       # request node number
#SBATCH --ntasks-per-node=1                             # number of tasks can execute on the node
#SBATCH --gpus-per-node=1                               # gpus per node
#SBATCH --cpus-per-task=4                               # cpus per gpu
#SBATCH --partition=gp1d                               # how long task can run
#SBATCH --output=./log/twnia2/train_3dfullres.out   # specify output Diectory and fileName

# environment setting
module purge
ml miniforge
conda activate nnUNet

### This shell script is edited by Russ as template script for other job ###
echo $(date)
source script/nchc/env.sh
### fold might not contain any class 3 (cal)
nnUNetv2_train 306 3d_fullres 0 