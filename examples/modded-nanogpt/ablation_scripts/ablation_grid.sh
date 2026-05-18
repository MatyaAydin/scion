#!/bin/bash

cd ..

FREQUENCIES=(10 25 50 100 200 250 500)
BETAS=(0.85 0.9 0.95 0.99 0.999)
LOGIDR="logs_grid"

for FREQ in "${FREQUENCIES[@]}"; do
    for BETA in "${BETAS[@]}"; do
        sbatch <<EOF
    #!/bin/bash
    #SBATCH --job-name=train_gpt_grid_${FREQ}_${BETA}
    #SBATCH --time=2:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=288
    #SBATCH --environment=vllm2026-container
    #SBATCH --account=a0114
    #SBATCH --gpus-per-node=4
    #SBATCH --output=result_grid_${FREQ}_${BETA}_%j.out
    #SBATCH --error=error_grid_${FREQ}_${BETA}_%j.err

    cd /iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt

    pip install -r ./requirements.txt
    pip install -r ./data/requirements.txt

    torchrun --standalone --nproc_per_node=4 ./train_mousse_scion.py --beta ${BETA} --eig_update_freq ${FREQ} --log_dir ${LOGDIR}
    EOF
    done