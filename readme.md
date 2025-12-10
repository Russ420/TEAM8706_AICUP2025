# Team 8706 AICUP 2025
## 1. Installation Guide:
First, create a conda environment with:

```bash
conda create -n nnUNet python=3.12
```
Then, activate the environment 
  ```
  conda activate nnUNet
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
  ``` 
Lastly,  install nnunetv2 and all dependencies. However, there is an error that needs to be avoided.
When installing the Python package locally, the top hierarchy cannot have `script/` and `log/` so we need to move them to somewhere else. After the installation, move them back.
```
git clone https://github.com/Russ420/TEAM8706_AICUP2025.git
cd TEAM8706_AICUP2025
mv script/ log/ -t /path/to/tmp
pip install -e .
mv /path/to/tmp/log /path/to/tmp/script -t .
```

## 2. Dataset Preparation:
### Step 1: Create Directory Structure
Create your own nnUNet dataset directory with the following subdirectories: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, and `nnUNet_predictions`.

Create your own Decathlon-like dataset directory with the following subdirectories: `imagesTr`, `imagesTs`, and `labelsTr`.
Place the train/test/label files correctly in the above directories.
### Step 2: Set Environment Variables

Add these paths (`nnUNet_*`) to your system environment variables. We have provided a sample script to add them to `$PATH`, located at:
```
./nnUNet/script/nchc/env.sh
```

It is essential to modify the paths.

### Step 3: Configure Dataset Path

Before creating the dataset with:
```
bash ./TEAM8706_AICUP2025/script/nchc/twnia2/prepare_dataset.sh
```

You should modify the `base` variable in the Python script located at:
`./TEAM8706_AICUP2025/nnunetv2/dataset_conversion/Dataset306_aicup2025.py:10`

Change it to `/path/to/your/aicup2025`.

### Step 4: Generate nnU-Net Format Dataset

Finally, run the following command to generate the nnU-Net-format dataset:
```
bash ./TEAM8706_AICUP2025/script/nchc/twnia2/prepare_dataset.sh
```

### My File Structure (Parallel Directory Setup)
```
ProjRoot/
    Dataset/
        aicup2025/ (original)
            imagesTr/
            imagesTs/
            labelsTr/
        nnUNet/
            nnUNet_raw/
            nnUNet_preprocessed/
            nnUNet_results/
            nnUNet_predictions/
    TEAM8706_AICUP2025/ (codebase)
```

## 3. Plan and Preprocess (Taiwania 3)

Submit the preprocessing job:
```
sbatch ./TEAM8706_AICUP2025/script/nchc/twnia3/plan_and_preprocess.sh
```

**For local machine:** If you use a local machine to preprocess your data, create a new file and directly run:
```
nnUNetv2_plan_and_preprocess -d 306 -np 8 --verify_dataset_integrity
```
Where:
- `-np` is the number of CPUs used to preprocess data. The more, the better.
- `-d` is the dataset identifier, here we set it to 306. If you wish to change it, you can modify variable `target_dataset_id` at `./TEAM8706_AICUP2025/nnunetv2/dataset_conversion/Dataset306_aicup2025.py:11`. Only three-digit identifiers are allowed.


## 4. Train (Taiwania 2)

### Option 1: Cross-Entropy + Dice Loss
```
sbatch ./TEAM8706_AICUP2025/script/nchc/twnia2/train.sh
```

### Option 2: Focal + Dice Loss
```bash
sbatch ./TEAM8706_AICUP2025/script/nchc/twnia2/train_fc.sh
```

**For local machine:** If you use a local machine to train your model, run:
```
nnUNetv2_train 306 3d_fullres 0
```
Where:
- `306` is the dataset identifier.
- `3d_fullres` is the neural network configuration.
- `0` is the fold identifier, it ranges from zero to four.
If you wish to use focal loss, please append `-tr nnUNetTrainerFocalLoss` to the above command.
## 5. Inference (Taiwania 2)

### Option 1: Cross-Entropy + Dice Model
```
sbatch ./TEAM8706_AICUP2025/script/nchc/twnia2/predict.sh
```

### Option 2: Focal + Dice Model
```
sbatch ./TEAM8706_AICUP2025/script/nchc/twnia2/predict_fc.sh
```

**For local machine:** If you use a local machine for inference, run:
```
nnUNetv2_predict -i $DIR_INPUT -o $DIR_OUTPUT -d 306 -c 3d_fullres
```

Where:
- `$DIR_INPUT` is the test images directory located at:
```
  ./Dataset/nnUNet/nnUNet_raw/imagesTs
```
- `$DIR_OUTPUT` is the directory to save prediction results. In this project, we set it to:
```
  ./Dataset/nnUNet/nnUNet_predictions/Dataset306_aicup2025/nnUNetTrainer__nnUNetPlans__3d_fullres/
```

**Note:** This output directory needs to be manually created before running inference.

**Best Results:** Our best competition result was achieved by ensembling models trained on all different folds (5-fold cross-validation).

## 6. Troubleshooting

For additional help, you can reference the [original nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet). However, the documentation above should be sufficient to reproduce our experiments. Or you can contact us at `m11102145@gapps.ntust.edu.tw`.

# Acknowledgements

This code is heavily based on nnU-Net. If you find it useful, please cite the following paper:
```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

**Credit:** nnU-Net is an awesome framework for medical image segmentation, and we are grateful for the authors' contribution to the research community.