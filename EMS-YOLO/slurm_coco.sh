#!/bin/bash
#SBATCH --job-name=ems-yolo-coco
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9          # 8 DataLoader workers + 1 main process
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/coco_%j.out
#SBATCH --error=logs/coco_%j.err

# -----------------------------------------------------------------------
# One-time setup (run these manually BEFORE submitting, not inside the job):
#
#   module load cuda/12.1
#   python -m venv $HOME/venvs/ems-yolo
#   source $HOME/venvs/ems-yolo/bin/activate
#   pip install --upgrade pip
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install "numpy<2" Pillow pycocotools wandb pyyaml
#
# If the node has Blackwell GPUs (SM 12.0), replace cu121 with cu128:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#
# Store your W&B key once:
#   echo "export WANDB_API_KEY=your_key_here" >> $HOME/.bashrc
# -----------------------------------------------------------------------

set -euo pipefail

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
module purge
module load cuda/12.1

source "$HOME/venvs/ems-yolo/bin/activate"

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
CONFIG="$REPO/EMS-YOLO/configs/coco_delftblue.yaml"

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
python "$REPO/EMS-YOLO/train_coco.py" \
    --data-root  "$COCO_SRC"  \
    --output     "$OUTPUT"    \
    --config     "$CONFIG"

echo "Done. Checkpoints in $OUTPUT"
