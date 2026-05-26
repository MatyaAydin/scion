#!/bin/bash

BETAS=(0.75 0.8 0.85 0.95 0.99)
LOGDIR="logs_ratio_ema"

for B in "${BETAS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_gpt_betascale_${B}
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=vllm2026-container
#SBATCH --account=a0114
#SBATCH --gpus-per-node=4
#SBATCH --output=result_betascale_${B}_%j.out
#SBATCH --error=error_betascale_${B}_%j.err

cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=4 ./train_mousse_scion.py --beta-scale ${B} --eig-update-freq 125 --log-dir ${LOGDIR} --grafting "ratio"
EOF
done