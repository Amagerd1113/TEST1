#!/bin/bash
# Gravitational Slingshot Navigation - Real Robot Deployment Script
# Deploy on Xiaomi/Roborock vacuum robot with ROS2

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}==== Slingshot Navigation - Real Robot Demo ====${NC}"
echo "Deploying on vacuum robot with ROS2 + Nav2"
echo ""

# Arguments
CHECKPOINT_PATH="${1}"
INSTRUCTION="${2:-Navigate to the blue chair}"
CONFIG_PATH="${3:-configs/infer_real_robot.yaml}"

if [ -z "${CHECKPOINT_PATH}" ]; then
    echo -e "${RED}Error: Checkpoint path required${NC}"
    echo "Usage: $0 <checkpoint_path> [instruction] [config_path]"
    echo "Example: $0 experiments/best_model.pt \"Go to the sofa\""
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${CHECKPOINT_PATH}${NC}"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Instruction: ${INSTRUCTION}"
echo "  Config: ${CONFIG_PATH}"
echo ""

# Check ROS2 environment
if [ -z "${ROS_DISTRO}" ]; then
    echo -e "${YELLOW}Warning: ROS_DISTRO not set. Attempting to source ROS2...${NC}"
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "Sourced ROS2 Humble"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo "Sourced ROS2 Foxy"
    else
        echo -e "${RED}Error: ROS2 not found. Please install ROS2 first.${NC}"
        exit 1
    fi
fi

echo "ROS2 Distribution: ${ROS_DISTRO}"
echo ""

# Check required ROS2 topics
echo -e "${YELLOW}Checking ROS2 topics...${NC}"
timeout 2 ros2 topic list > /tmp/ros2_topics.txt 2>/dev/null || {
    echo -e "${RED}Error: Cannot list ROS2 topics. Is roscore running?${NC}"
    echo "Please ensure your robot is powered on and ROS2 is running."
    exit 1
}

REQUIRED_TOPICS=(
    "/camera/color/image_raw"
    "/camera/depth/image_raw"
    "/odom"
)

for topic in "${REQUIRED_TOPICS[@]}"; do
    if grep -q "${topic}" /tmp/ros2_topics.txt; then
        echo -e "  ${GREEN}✓${NC} ${topic}"
    else
        echo -e "  ${RED}✗${NC} ${topic} (not found)"
        echo -e "${YELLOW}Warning: Required topic missing. Proceeding anyway...${NC}"
    fi
done
echo ""

# Create log directory
LOG_DIR="experiments/real_robot_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logging to: ${LOG_DIR}"
echo ""

# Safety check
echo -e "${BLUE}==== SAFETY NOTICE ====${NC}"
echo "The robot will start moving based on the instruction:"
echo "  \"${INSTRUCTION}\""
echo ""
echo "Please ensure:"
echo "  1. The robot has clear space to navigate"
echo "  2. Emergency stop is accessible"
echo "  3. You are ready to intervene if needed"
echo ""
read -p "Press ENTER to continue, or Ctrl+C to abort..."
echo ""

# Launch navigation node
echo -e "${GREEN}Launching Slingshot Navigation node...${NC}"
python src/real_robot/ros2_wrapper.py \
    --checkpoint="${CHECKPOINT_PATH}" \
    --config="${CONFIG_PATH}" \
    --instruction="${INSTRUCTION}" \
    --log_dir="${LOG_DIR}" \
    2>&1 | tee "${LOG_DIR}/console.log"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Navigation completed successfully!${NC}"
    echo "Logs saved to: ${LOG_DIR}"
else
    echo ""
    echo -e "${RED}Navigation failed or was interrupted${NC}"
    echo "Check logs in: ${LOG_DIR}"
    exit 1
fi
