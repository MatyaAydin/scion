#!/bin/bash


ORDERS=(0.125 0.25 0.5)
LOGDIR="logs_order"

for ORD in "${ORDERS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_gpt_order_${ORD}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=vllm2026-container
#SBATCH --account=a0114
#SBATCH --gpus-per-node=4
#SBATCH --output=result_order_${ORD}_%j.out
#SBATCH --error=error_order_${ORD}_%j.err

cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=4 ./train_mousse_scion.py --alpha ${ORD} --log_dir ${LOGDIR}
EOF
done