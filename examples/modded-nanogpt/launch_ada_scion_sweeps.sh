#!/bin/bash
# launch_ada_scion_sweeps.sh
#
# Submits 5 SLURM jobs — one per ablation group.
# Each job loops over all values of its hyperparameter internally.
#
# Usage:
#   bash launch_ada_scion_sweeps.sh               # submit all 5 sweeps
#   bash launch_ada_scion_sweeps.sh lr
#   bash launch_ada_scion_sweeps.sh momentum
#   bash launch_ada_scion_sweeps.sh beta
#   bash launch_ada_scion_sweeps.sh eps
#   bash launch_ada_scion_sweeps.sh power_frequency

set -euo pipefail

SLURM_ACCOUNT="a0114"
SLURM_ENV="vllm2026-container"
WORK_DIR="/iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt"
TRAIN_SCRIPT="./train_gpt_ada_scion_sweep.py"
NPROC=4
TIME_LIMIT="2:00:00"

submit_sweep() {
    local sweep="$1"
    echo "Submitting sweep: ${sweep}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=adascion_sweep_${sweep}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=${SLURM_ENV}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --gpus-per-node=${NPROC}
#SBATCH --output=result_adascion_sweep_${sweep}_%j.out
#SBATCH --error=error_adascion_sweep_${sweep}_%j.err

cd ${WORK_DIR}

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=${NPROC} ${TRAIN_SCRIPT} --sweep ${sweep}
EOF
}

TARGET="${1:-all}"

case "${TARGET}" in
    lr|momentum|beta|eps|power_frequency)
        submit_sweep "${TARGET}"
        ;;
    all)
        for sweep in lr momentum beta eps power_frequency; do
            submit_sweep "${sweep}"
        done
        ;;
    *)
        echo "Unknown target '${TARGET}'. Choose: lr | momentum | beta | eps | power_frequency | all"
        exit 1
        ;;
esac
