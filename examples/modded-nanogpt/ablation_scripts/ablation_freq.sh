#!/bin/bash


FREQUENCIES=(10 25 50 100 250 500)
LOGDIR="logs_freq"

for FREQ in "${FREQUENCIES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_gpt_freq_${FREQ}
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=vllm2026-container
#SBATCH --account=a0114
#SBATCH --gpus-per-node=4
#SBATCH --output=result_freq_${FREQ}_%j.out
#SBATCH --error=error_freq_${FREQ}_%j.err

cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=4 ./train_mousse_scion.py --eig-update-freq ${FREQ} --log-dir ${LOGDIR}
EOF
done