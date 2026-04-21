#!/bin/bash
# launch_ablations.sh
#
# Submits 4 SLURM jobs — one per ablation group.
# Each job loops over all values of its hyperparameter internally.
#
# Usage:
#   bash launch_ablations.sh          # submit all 4 sweeps
#   bash launch_ablations.sh lr       # submit only the lr sweep
#   bash launch_ablations.sh momentum
#   bash launch_ablations.sh beta_LR
#   bash launch_ablations.sh eps

set -euo pipefail

SLURM_ACCOUNT="a0114"
SLURM_ENV="vllm2026-container"
WORK_DIR="/iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt"
TRAIN_SCRIPT="./train_gpt_steepest_scion.py"
NPROC=4
TIME_LIMIT="2:00:00"

submit_sweep() {
    local sweep="$1"
    echo "Submitting sweep: ${sweep}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=scion_sweep_${sweep}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=${SLURM_ENV}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --gpus-per-node=${NPROC}
#SBATCH --output=result_sweep_${sweep}_%j.out
#SBATCH --error=error_sweep_${sweep}_%j.err

cd ${WORK_DIR}

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=${NPROC} ${TRAIN_SCRIPT} --sweep ${sweep}
EOF
}

TARGET="${1:-all}"

case "${TARGET}" in
    lr|momentum|beta_LR|eps)
        submit_sweep "${TARGET}"
        ;;
    all)
        for sweep in lr momentum beta_LR eps; do
            submit_sweep "${sweep}"
        done
        ;;
    *)
        echo "Unknown target '${TARGET}'. Choose: lr | momentum | beta_LR | eps | all"
        exit 1
        ;;
esac

echo ""
echo "Done. Check queued jobs with: squeue -u \$USER"
