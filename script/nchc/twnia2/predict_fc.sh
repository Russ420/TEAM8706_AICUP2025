#!/bin/bash
#SBATCH --account=MST114281                             	# parent project to access twcc system
#SBATCH --job-name=predict_fc_3d_fullres                    # jobName
#SBATCH --nodes=1                                       	# request node number
#SBATCH --ntasks-per-node=1                             	# number of tasks can execute on the node
#SBATCH --gpus-per-node=1                               	# gpus per node
#SBATCH --cpus-per-task=4                               	# cpus per gpu
#SBATCH --partition=gp1d                              		# how long task can run
#SBATCH --output=./log/twnia2/predict_fc_3d_fullres.out 	# specify output Diectory and fileName

# environment setting
module purge
ml miniforge
conda activate nnUNet
CONFIG="3d_fullres"

### This shell script is edited by Russ as template script for other job ###
echo $(date)
source script/nchc/env.sh

DIR_INPUT="$nnUNet_raw/Dataset306_aicup2025/imagesTs"

## vanilla
#DIR_OUTPUT="$nnUNet_predictions/Dataset306_aicup2025/nnUNetTrainer__nnUNetPlans__$CONFIG"
#nnUNetv2_predict -i $DIR_INPUT -o $DIR_OUTPUT -d 306 -c $CONFIG -chk checkpoint_best.pth
DIR_OUTPUT="$nnUNet_predictions/Dataset306_aicup2025/nnUNetTrainerFocalLoss__nnUNetPlans__$CONFIG"
nnUNetv2_predict -i $DIR_INPUT -o $DIR_OUTPUT -d 306 -c $CONFIG -tr nnUNetTrainerFocalLoss -chk checkpoint_best.pth

# vanilla cascade
#DIR_PREV="$nnUNet_predictions/Dataset306_aicup2025/nnUNetTrainer__nnUNetPlans__3d_lowres"
#nnUNetv2_predict -i $DIR_INPUT -o $DIR_OUTPUT -d 306 -c $CONFIG -f all -prev_stage_predictions $DIR_PREV -chk checkpoint_best.pth

# residual
#DIR_OUTPUT="$nnUNet_predictions/Dataset306_aicup2025/nnUNetTrainer__nnUNetResEncUNetLPlans__$CONFIG"
#nnUNetv2_predict -i $DIR_INPUT -o $DIR_OUTPUT -d 306 -c $CONFIG -f all
