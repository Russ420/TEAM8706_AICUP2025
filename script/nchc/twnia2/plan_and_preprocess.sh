#!/bin/bash
#SBATCH --account=MST114281                             # parent project to access twcc system
#SBATCH --job-name=plan_and_preprocess                  # jobName
#SBATCH --nodes=1                                       # request node number
#SBATCH --ntasks-per-node=1                             # number of tasks can execute on the node
#SBATCH --gpus-per-node=1                               # gpus per node
#SBATCH --cpus-per-task=4                               # cpus per gpu
#SBATCH --partition=gtest                               # how long task can run
#SBATCH --output=./log/twnia2/plan_and_preprocess.out   # specify output Diectory and fileName

### This shell script is deprecated since twnia2 is unsuitable for cpu-bounded task.
echo "deprecated script"
exit 0

# environment setting
module purge
ml miniforge
conda activate nnUNet

### This shell script is edited by Russ as template script for other job ###
source script/nchc/env.sh
nnUNetv2_plan_and_preprocess -d 306 --verify_dataset_integrity