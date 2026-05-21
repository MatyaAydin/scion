#!/bin/bash


RHOS=(10 25 50 100 250 500)
LOGDIR="logs_rho_unconstr"

for R in "${RHOS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_gpt_rho_${R}
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=vllm2026-container
#SBATCH --account=a0114
#SBATCH --gpus-per-node=4
#SBATCH --output=result_rho_${R}_%j.out
#SBATCH --error=error_rho_${R}_%j.err

cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=4 ./train_mousse_scion.py --thresh ${R} --log-dir ${LOGDIR} --unconstrained True
EOF
done