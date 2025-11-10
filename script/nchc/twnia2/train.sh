#!/bin/bash
#SBATCH --account=MST114281                                     # parent project to access twcc system
#SBATCH --job-name=train_3d_cascade_fullres                     # jobName
#SBATCH --nodes=1                                               # request node number
#SBATCH --ntasks-per-node=1                                     # number of tasks can execute on the node
#SBATCH --gpus-per-node=1                                       # gpus per node
#SBATCH --cpus-per-task=4                                       # cpus per gpu
#SBATCH --partition=gp2d                                        # how long task can run
#SBATCH --output=./log/twnia2/train_3d_cascade_fullres.out      # specify output Diectory and fileName

# environment setting
module purge
ml miniforge
conda activate nnUNet

### This shell script is edited by Russ as template script for other job ###
echo $(date)
source script/nchc/env.sh

### fold might not contain any class 3 (cal)
N_FOLD="all"
### ensure the suffix of the job name and log file are identical
CONFIG="3d_cascade_fullres"

# DICE + CE
#nnUNetv2_train 306 2d $N_FOLD 
#nnUNetv2_train 306 3d_fullres $N_FOLD 
#nnUNetv2_train 306 3d_lowres $N_FOLD 
nnUNetv2_train 306 $CONFIG $N_FOLD 

# DICE + Focal
#nnUNetv2_train 306 3d_fullres $N_FOLD -tr