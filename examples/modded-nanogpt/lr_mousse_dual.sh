#!/bin/bash

LEARNING_RATES=(5e-3 3e-3 2e-3 1.5e-3 1.2e-3)

for LR in "${LEARNING_RATES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_gpt_lr_${LR}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=vllm2026-container
#SBATCH --account=a0114
#SBATCH --gpus-per-node=4
#SBATCH --output=result_lr_${LR}_%j.out
#SBATCH --error=error_lr_${LR}_%j.err

cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=4 ./train_mousse_scion.py --lr ${LR}
EOF
done