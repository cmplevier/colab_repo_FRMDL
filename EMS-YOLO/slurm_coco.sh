#!/bin/bash
#SBATCH --job-name=ems-yolo-coco
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17         # 8 workers * 2 GPUs + 1 torchrun process
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=18:00:00
#SBATCH --output=logs/coco_%j.out
#SBATCH --error=logs/coco_%j.err


set -euo pipefail

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
module purge
module load cuda/12.1
module load miniforge3

conda activate ems-yolo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B key — must be set in your environment or .bashrc before submitting
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Set it in ~/.bashrc or pass it via:"
    echo "  sbatch --export=ALL,WANDB_API_KEY=your_key slurm_coco.sh"
    exit 1
fi

# -----------------------------------------------------------------------
# Paths — adjust REPO and COCO_SRC to match your DelftBlue layout
# -----------------------------------------------------------------------
REPO="$HOME/colab_repo_FRMDL"
COCO_SRC="/scratch/$USER/coco"          # COCO dataset already on scratch
OUTPUT="/scratch/$USER/runs/coco-resnet34"
CONFIG="$REPO/EMS-YOLO/configs/coco_delftblue_416.yaml"

mkdir -p "$OUTPUT"
mkdir -p "$REPO/logs"

# -----------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------
echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $SLURM_GPUS_ON_NODE"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Python     : $(which python)"
echo "PyTorch    : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name   : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "=============================="

# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------
# Fresh run:
#   sbatch EMS-YOLO/slurm_coco.sh
#
# Resume (pass full path to last.pt):
#   sbatch --export=ALL,RESUME=/scratch/hzaharia/runs/coco-resnet34/20260611_120000/last.pt EMS-YOLO/slurm_coco.sh
RESUME_ARG=""
if [[ -n "${RESUME:-}" ]]; then
    echo "Resuming from $RESUME"
    RESUME_ARG="--resume $RESUME"
fi

torchrun --standalone --nproc_per_node=2 "$REPO/EMS-YOLO/train_coco.py" \
    --data-root  "$COCO_SRC"  \
    --output     "$OUTPUT"    \
    --config     "$CONFIG"    \
    $RESUME_ARG

echo "Done. Checkpoints in $OUTPUT"
