#!/bin/bash
#SBATCH --account=MST114281                         # parent project to access twcc system
#SBATCH --job-name=prepare_dataset306               # jobName
#SBATCH --nodes=1                                   # request node number
#SBATCH --ntasks-per-node=1                         # number of tasks can execute on the node
#SBATCH --gpus-per-node=1                           # gpus per node
#SBATCH --cpus-per-task=4                           # cpus per gpu
#SBATCH --partition=gtest                           # how long task can run
#SBATCH --output=./log/twnia2/prepare_dataset.out   # specify output Diectory and fileName

# environment setting
module purge
ml miniforge
conda activate nnUNet

### This shell script is edited by Russ as template script for other job ###
source script/twnia2/env.sh
python nnunetv2/dataset_conversion/Dataset306_aicup2025.py

