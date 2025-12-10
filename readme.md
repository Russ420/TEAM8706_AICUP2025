# Team 8706
1. Installation Guides:
First, create an environment with ```conda create -n nnUNet python=3.12```
Then, activate the environment ```conda activate nnUNet``` and install pytorch with ```pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126```.
Lastly, type ```pip install nnunetv2``` to install all the dependency.

2. Dataset Preparation:
Please create your own nnUNet dataset directory which contains subdirectory: ```nnUNet_raw```, ```nnUNet_preprocessed```, ```nnUNet_results```, and ```nnUNet_predictions```. 
Please add those paths (```nnUNet_*```) into system variables. 
We have provided a sample script to add them into ```$PATH``` which is located at ```./nnUNet/script/nchc/env.sh```
Before creating dataset with ```bash ./nnUNet/script/nchc/twnia2/prepare_dataset.sh```, you should modify the ```base``` variable in python script located at ```./nnUNet/nnunetv2/dataset_conversion/Dataset306_aicup2025.py:10```. 
Change it to ```/path/to/yourdataset```.
Finally, you can ```bash ./nnUNet/script/nchc/twnia2/prepare_dataset.sh``` to generate the nnUNet-format dataset.

    My File Structure (Parallel):
    ```
    home/
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
        nnUNet/ (codebase)
    ```

3. Plan and Preprocess (twnia3):
* ```sbatch ./nnUNet/script/nchc/twnia3/plan_and_preprocess.sh```

    If you use local machine to preprocess your data. Please create a new file and directly use ```nnUNetv2_plan_and_process -d 306 -np 8 --verify_dataset_integerity```

4. Train (twnia2):
* ```sbatch script/nchc/twnia2/train.sh``` (CE+Dice)
* ```sbatch script/nchc/twnia2/train_fc.sh``` (Focal+Dice)

    If you use local machine to train your model, use ```nnUNetv2_train 306 3d_fullres 0```

5. Inference (twnia2):
* ```sbatch script/nchc/twnia2/predict.sh (CE+Dice)```
* ```sbatch script/nchc/twnia2/predict_fc.sh (Focal+Dice)```

    If you use local machine to train your model, use ```nnUNetv2_predict -i $DIR_INPUT -o $DIR_OUTPUT -d 306 -c 3d_fullres```, ```$DIR_INPUT``` is the imagesTs which is loacated at ```./Dataset/nnUNet/nnUNet_raw/imagesTs``` and ```DIR_OUTPUT``` is the directory to save the prediction results. In this project, we set it to ```./Dataset/nnUNet/nnUNet_predictions/Dataset306_aicup2025/nnUNetTrainer__nnUNetPlans__3d_fullres/```. Noted this directory needs to be manually created.

    The best result of our team needs to be esemble all the models trained on different folds.
6. TroubleShooting: 
   
   You can reference the [original nnUNet document](https://github.com/MIC-DKFZ/nnUNet) but I think the above document is sufficient to reproduce the experiments.

# Acknowledgements
This code is heavily based on nnUNet. If you find it useful, please cite the following paper. nnUNet is the best no doubt.
```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
```
