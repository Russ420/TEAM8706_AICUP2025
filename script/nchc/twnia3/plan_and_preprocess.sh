#!/bin/bash
#SBATCH --account=MST114281                             # -A，指定計畫代號     
#SBATCH --partition=ct56                                # -p，指定partition/queue (佇列)
#SBATCH --job-name=plan_and_preprocess                  # -j，設定工作名稱
#SBATCH --output=./log/twnia3/plan_and_preprocess.out   # -o，設定輸出檔案名稱，%j 會替換成job ID
#SBATCH --error=./log/twnia3/plan_and_preprocess.err    # -e，設定錯誤輸出檔案名稱
#SBATCH --ntasks=1                                      # -n how many jobs
#SBATCH --nodes=1                                      # -N how many nodes utilized
#SBATCH --cpus-per-task=28                              # -c how many cpus alloaced for one task
#SBATCH --time=02:00:00                                 # -t，設定執行時間限制 (格式: DD-HH:MM:SS)
#SBATCH --mem-per-cpu=12G                               # 記憶體需求〔K|M|G|T〕
#SBATCH --mail-type=ALL           
#SBATCH --mail-user=m11102145@gapps.ntust.edu.tw  # 當mail-type事件觸發時，欲通知的E-mail

# environment setting
module purge
ml miniconda3
conda activate nnUNet

# 以下部分是要執行的程式
source script/nchc/env.sh
srun nnUNetv2_plan_and_preprocess -d 306 -np 28 --verify_dataset_integrity