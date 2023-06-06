#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=GNS_locs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=96:00:00
#SBATCH --mem=120000M
#SBATCH --output=/home/vkyriacou88/output_gns_locs_st_results_%A.out

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
--model_path="${SCRATCH}/models/${data}/locs/st_transformer" \
--output_path="${SCRATCH}/rollout/${data}/locs/st_transformer" \
--nsave_steps=50000 \
--cuda_device_number=0 \
--ntraining_steps=20000000 \
--batch_size=2 \
--model_name="locs_st_transformer" \
# --lr_init=0.001 \
# --noise_std=3e-4 
--model_file="latest" \
--train_state_file="latest"

