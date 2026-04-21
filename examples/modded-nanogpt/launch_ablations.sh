#!/bin/bash
# launch_ablations.sh
#
# Submits one SLURM job per ablation value.
# Each job varies exactly one optim_arg; the rest stay at their defaults.
#
# Usage:
#   bash launch_ablations.sh              # submit all four ablation groups
#   bash launch_ablations.sh lr           # submit only the lr sweep
#   bash launch_ablations.sh momentum     # submit only the momentum sweep
#   bash launch_ablations.sh beta_LR      # etc.
#   bash launch_ablations.sh eps
#
# Prerequisites:
#   1. Apply train_gpt_steepest_scion_patch.py to your training script so it
#      accepts --lr / --momentum / --beta_LR / --eps as CLI flags.
#   2. Commit & push the updated training script before running this.

set -euo pipefail

# ── Defaults (match what the patched script uses) ────────────────────────────
DEFAULT_LR=5e-5
DEFAULT_MOMENTUM=0.9
DEFAULT_BETA_LR=0.999
DEFAULT_EPS=1.0

# ── Ablation value lists ──────────────────────────────────────────────────────
LRS=(1.29e-06 2.78e-06 5.99e-06 1.29e-05 2.78e-05 5.99e-05 1.29e-04 2.78e-04 5.99e-04 1e-04)
EPSILONS=(1e-06 1e-05 1e-04 1e-03 1e-02 1e-01 1e+00)
MOMENTA=(0.4 0.5 0.6 0.7 0.8 0.9 0.96 0.99)
BETAS=(0.85 0.9 0.99 0.999)

# ── SLURM settings — edit to match your cluster ──────────────────────────────
SLURM_ACCOUNT="a0114"
SLURM_ENV="vllm2026-container"
WORK_DIR="/iopsstor/scratch/cscs/maydin/scion/examples/modded-nanogpt"
TRAIN_SCRIPT="./train_gpt_steepest_scion.py"
NPROC=4
TIME_LIMIT="2:00:00"

# ── Helpers ───────────────────────────────────────────────────────────────────

submit_job() {
    local job_name="$1"
    local lr="$2"
    local momentum="$3"
    local beta_lr="$4"
    local eps="$5"

    echo "Submitting: ${job_name}  lr=${lr}  momentum=${momentum}  beta_LR=${beta_lr}  eps=${eps}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=${SLURM_ENV}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --gpus-per-node=${NPROC}
#SBATCH --output=result_${job_name}_%j.out
#SBATCH --error=error_${job_name}_%j.err

cd ${WORK_DIR}

pip install -r ./requirements.txt
pip install -r ./data/requirements.txt

torchrun --standalone --nproc_per_node=${NPROC} ${TRAIN_SCRIPT} \
    --lr ${lr} \
    --momentum ${momentum} \
    --beta_LR ${beta_lr} \
    --eps ${eps}
EOF
}

sweep_lr() {
    echo "=== LR sweep ==="
    for lr in "${LRS[@]}"; do
        submit_job "scion_lr_${lr}" \
            "${lr}" "${DEFAULT_MOMENTUM}" "${DEFAULT_BETA_LR}" "${DEFAULT_EPS}"
    done
}

sweep_momentum() {
    echo "=== Momentum sweep ==="
    for m in "${MOMENTA[@]}"; do
        submit_job "scion_mom_${m}" \
            "${DEFAULT_LR}" "${m}" "${DEFAULT_BETA_LR}" "${DEFAULT_EPS}"
    done
}

sweep_beta() {
    echo "=== beta_LR sweep ==="
    for b in "${BETAS[@]}"; do
        submit_job "scion_beta_${b}" \
            "${DEFAULT_LR}" "${DEFAULT_MOMENTUM}" "${b}" "${DEFAULT_EPS}"
    done
}

sweep_eps() {
    echo "=== eps sweep ==="
    for e in "${EPSILONS[@]}"; do
        submit_job "scion_eps_${e}" \
            "${DEFAULT_LR}" "${DEFAULT_MOMENTUM}" "${DEFAULT_BETA_LR}" "${e}"
    done
}

# ── Entry point ───────────────────────────────────────────────────────────────

TARGET="${1:-all}"

case "${TARGET}" in
    lr)       sweep_lr ;;
    momentum) sweep_momentum ;;
    beta_LR)  sweep_beta ;;
    eps)      sweep_eps ;;
    all)
        sweep_lr
        sweep_momentum
        sweep_beta
        sweep_eps
        ;;
    *)
        echo "Unknown target '${TARGET}'. Choose: lr | momentum | beta_LR | eps | all"
        exit 1
        ;;
esac

echo ""
echo "Done. Check queued jobs with: squeue -u \$USER"
