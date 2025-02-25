#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=GNS_Baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=96:00:00
#SBATCH --mem=120000M
#SBATCH --output=/home/vkyriacou88/output_gns_base_results_%A.out

module purge
module load 2021
module load Anaconda3/2021.05
# source start_venv.sh
source activate gns
# fail on error
set -e

# start in slurm_scripts
cd ~/gns

# assume data is already downloaded and hardcode WaterDropSample
data="Water-3D"
SCRATCH="/scratch-shared/vkyriacou88/"
~/.conda/envs/gns/bin/python -m gns.train --data_path="${SCRATCH}/datasets/${data}/" \
--model_path="${SCRATCH}/models/${data}/baseline/" \
--output_path="${SCRATCH}/rollout/${data}/baseline/" \
--nsave_steps=50000 \
--cuda_device_number=0 \
--ntraining_steps=18000000 \
--model_name="gns" \
--batch_size=2 \
# --model_file="latest" \
# --train_state_file="latest"
