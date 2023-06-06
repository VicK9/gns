#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=GNS_locs_rollout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=5:00:00
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
for i in {29..99}
do
    python3 -m gns.render_rollout \
    --rollout_dir="/scratch-shared/vkyriacou88/rollout/${data}/locs/" \
    --rollout_name="rollout_${i}" 
done


