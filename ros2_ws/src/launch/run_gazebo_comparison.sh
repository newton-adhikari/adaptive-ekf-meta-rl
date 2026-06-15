#!/bin/bash
# =============================================================
# Gazebo Comparison Pipeline
# Runs TurtleBot4 through indoor_lab.world with each EKF method
# and records NEES consistency.


# Usage:
#   chmod +x ros2_ws/src/launch/run_gazebo_comparison.sh
#   ./ros2_ws/src/launch/run_gazebo_comparison.sh
# =============================================================

set -e

echo "============================================================="
echo "CC-MetaEKF Gazebo Demo (ROS2 Real-Time Integration)"
echo "============================================================="

# === WSL Compatibility ===
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=1
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export TURTLEBOT3_MODEL=burger

# Source ROS2
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash 2>/dev/null || true

export PYTHONPATH="$(pwd):$PYTHONPATH"

WORLD_FILE="$(pwd)/ros2_ws/src/gazebo_worlds/indoor_lab.world"
TB3_MODEL="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
CHECKPOINT="results/run_s42/best_stsie_pid_s42.pt"

echo ""
echo "  World: indoor_lab"
echo "  Checkpoint: $CHECKPOINT"
echo "-------------------------------------------------------------"

# Launch gzserver
ros2 launch gazebo_ros gzserver.launch.py world:=$WORLD_FILE &
GZSERVER_PID=$!
sleep 6

# Launch gzclient (GUI)
ros2 launch gazebo_ros gzclient.launch.py &
GZCLIENT_PID=$!
sleep 3

# Spawn TB3
ros2 run gazebo_ros spawn_entity.py \
    -entity burger -file $TB3_MODEL \
    -x 0.0 -y 0.0 -z 0.01 -robot_namespace / &
sleep 8


# Run the CC-MetaEKF node
python3 ros2_ws/src/meta_ekf_node/meta_ekf_node/meta_ekf_node.py \
    --ros-args -p method:=ccmetaekf -p checkpoint_path:=$CHECKPOINT \
    -p use_sim_time:=true &
EKF_PID=$!

# Wait for Ctrl+C
wait $EKF_PID 2>/dev/null || true

# Cleanup
kill $GZCLIENT_PID $GZSERVER_PID 2>/dev/null || true
pkill -f "gzserver" 2>/dev/null || true
pkill -f "gzclient" 2>/dev/null || true
echo ""
echo "  Demo stopped."
 