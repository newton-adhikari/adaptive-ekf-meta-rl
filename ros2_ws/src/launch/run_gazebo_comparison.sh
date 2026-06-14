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
echo "CC-MetaEKF Gazebo Comparison"
echo "============================================================="

# === WSL Compatibility Fixes ===
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=1
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# TurtleBot3 model
export TURTLEBOT3_MODEL=burger

# Source ROS2
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash 2>/dev/null || echo "  Note: build workspace first with 'cd ros2_ws && colcon build'"

# Add project root to PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Custom world file (absolute path)
WORLD_FILE="$(pwd)/ros2_ws/src/gazebo_worlds/indoor_lab.world"

# TB3 model SDF for spawning
TB3_MODEL="/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
TB3_URDF="/opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf"

RESULTS_DIR="results/gazebo"
mkdir -p $RESULTS_DIR

METHODS=("fixed" "sage_husa" "oracle" "ccmetaekf")
DURATION=60  # seconds per trial
CHECKPOINT="results/run_s42/best_stsie_pid_s42.pt"

echo ""
echo "World: indoor_lab ($WORLD_FILE)"
echo "-------------------------------------------------------------"

for method in "${METHODS[@]}"; do
    echo "  Method: $method"

    # 1) Start gzserver via gazebo_ros (loads ROS plugins including /spawn_entity)
    ros2 launch gazebo_ros gzserver.launch.py world:=$WORLD_FILE &
    GZSERVER_PID=$!
    sleep 6

    # 2) Start gzclient (GUI)
    ros2 launch gazebo_ros gzclient.launch.py &
    GZCLIENT_PID=$!
    sleep 3

    # 3) Publish robot_description and start robot_state_publisher
    ros2 run robot_state_publisher robot_state_publisher \
        --ros-args -p robot_description:="$(xacro /opt/ros/humble/share/turtlebot3_description/urdf/turtlebot3_burger.urdf.xacro)" \
        -p use_sim_time:=true &
    RSP_PID=$!
    sleep 1

    # 4) Spawn TB3 into our world
    ros2 run gazebo_ros spawn_entity.py \
        -entity burger -file $TB3_MODEL \
        -x 0.0 -y 0.0 -z 0.01 \
        -robot_namespace / &
    SPAWN_PID=$!
    sleep 8

    # 5) Run meta_ekf_node
    python3 ros2_ws/src/meta_ekf_node/meta_ekf_node/meta_ekf_node.py \
        --ros-args -p method:=$method -p checkpoint_path:=$CHECKPOINT \
        -p use_sim_time:=true &
    EKF_PID=$!
    sleep 2

    # 6) Run evaluation node
    python3 ros2_ws/src/evaluation/evaluation/evaluation_node.py \
        --ros-args -p output_file:=$RESULTS_DIR/${method}_indoor_lab.json \
        -p method_name:=$method -p use_sim_time:=true &
    EVAL_PID=$!
    sleep 2

    # 7) Drive robot forward
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 0.2}, angular: {z: 0.0}}" --rate 10 &
    CMD_PID=$!

    # Record for DURATION seconds
    sleep $DURATION

    echo "    Completed $DURATION seconds"

    # Kill all processes
    kill $CMD_PID $EKF_PID $EVAL_PID $RSP_PID $GZCLIENT_PID $GZSERVER_PID 2>/dev/null || true
    wait $CMD_PID $EKF_PID $EVAL_PID $RSP_PID $GZCLIENT_PID $GZSERVER_PID 2>/dev/null || true
    pkill -f "gzserver" 2>/dev/null || true
    pkill -f "gzclient" 2>/dev/null || true
    sleep 5

    echo "    Done. Results in $RESULTS_DIR/${method}_indoor_lab.json"
done

echo ""
echo "============================================================="
echo "Gazebo comparison complete. Results in: $RESULTS_DIR/"
echo "============================================================="
 