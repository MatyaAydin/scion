#!/bin/bash

STEPS=(2 3 4 5 6)
LOGDIR="logs_ns_scion"

for S in "${STEPS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_gpt_ns_scion_${S}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=vllm2026-container
#SBATCH --account=a0114
#SBATCH --gpus-per-node=4
#SBATCH --output=result_ns_scion_${S}_%j.out
#SBATCH --error=error_ns_scion_${S}_%j.err

cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=4 ./train_gpt_scion.py --ns-steps ${S} --log-dir ${LOGDIR}
EOF
done