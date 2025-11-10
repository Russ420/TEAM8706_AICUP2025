#!/bin/bash
export nnUNet="/work/u4526725/ntuh/nnUNet"
export nnUNet_raw="/work/u4526725/Dataset/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/work/u4526725/Dataset/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/work/u4526725/Dataset/nnUNet/nnUNet_results"
export nnUNet_predictions="/work/u4526725/Dataset/nnUNet/nnUNet_predictions"

PATH="$PATH:/home/u4526725/.local/bin"
echo $nnUNet
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
echo $nnUNet_predictions
echo $PATH