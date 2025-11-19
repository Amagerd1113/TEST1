#!/bin/bash
# Automated dataset download script for Slingshot-VLN
# Total download size: ≈50GB
# Estimated time: 30-60 minutes depending on network speed

set -e  # Exit on error

echo "=========================================="
echo " Slingshot-VLN Dataset Setup"
echo " Total download: ≈50GB"
echo "=========================================="
echo ""

# Create directories
echo "[1/3] Creating data directories..."
mkdir -p data/scene_datasets/mp3d
mkdir -p data/datasets/vlnce
mkdir -p data/datasets/reverie

echo "✓ Directories created"
echo ""

# Download Matterport3D scenes
echo "[2/3] Downloading Matterport3D scenes (≈35GB)..."
echo "This may take 20-40 minutes depending on your connection..."
python -m habitat_sim.utils.datasets_download --uids mp3d_habitat --data-path data/scene_datasets/mp3d/

echo "✓ Matterport3D download complete"
echo ""

# Download VLN-CE episodes
echo "[3/3] Downloading VLN-CE episodes (≈5GB)..."
wget -O data/datasets/vlnce/vlnce_episodes.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip
unzip -o data/datasets/vlnce/vlnce_episodes.zip -d data/datasets/vlnce/
rm data/datasets/vlnce/vlnce_episodes.zip

echo "✓ VLN-CE episodes downloaded"
echo ""

# Download REVERIE (optional)
echo "[OPTIONAL] Downloading REVERIE dataset (≈10GB)..."
read -p "Download REVERIE? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    wget -O data/datasets/reverie/REVERIE_VLNCE_v1.zip https://dl.fbaipublicfiles.com/reverie/REVERIE_VLNCE_v1.zip
    unzip -o data/datasets/reverie/REVERIE_VLNCE_v1.zip -d data/datasets/reverie/
    rm data/datasets/reverie/REVERIE_VLNCE_v1.zip
    echo "✓ REVERIE downloaded"
else
    echo "⊘ REVERIE download skipped"
fi

echo ""
echo "=========================================="
echo " ✓ All datasets ready!"
echo "=========================================="
echo ""
echo "Total disk usage:"
du -sh data/
echo ""
echo "Next steps:"
echo "  1. Run demo: python demo.py --visualize"
echo "  2. Start training: bash scripts/train.sh configs/train_vln_ce.yaml 1"
echo ""
