#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=GNS_locs_rollout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=2:00:00
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

python3 -m gns.train --mode="rollout" \
--data_path="${SCRATCH}/datasets/${data}/" \
--model_path="${SCRATCH}/models/${data}/baseline/" \
--model_file="model-360000.pt" \
--output_path="${SCRATCH}/rollout/${data}/baseline/"
