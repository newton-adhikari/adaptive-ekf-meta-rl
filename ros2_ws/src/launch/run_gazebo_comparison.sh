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

# Source ROS2
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash 2>/dev/null || echo "we have, to build workspace first 'cd ros2_ws && colcon build'"

RESULTS_DIR="results/gazebo"
mkdir -p $RESULTS_DIR

METHODS=("fixed" "sage_husa" "oracle" "ccmetaekf")
WORLDS=("indoor_lab")
DURATION=60  # seconds per trial
CHECKPOINT="results/run_s42/best_stsie_pid_s42.pt"

for world in "${WORLDS[@]}"; do
    echo ""
    echo "World: $world"
    echo "-------------------------------------------------------------"

    for method in "${METHODS[@]}"; do
        echo "  Method: $method"

        # Launch Gazebo + robot + EKF in background
        ros2 launch ros2_ws/src/launch/gazebo_comparison.launch.py \
            method:=$method \
            world:=$world \
            checkpoint:=$CHECKPOINT &
        LAUNCH_PID=$!

        # Wait for system to start
        sleep 10

        # Send navigation goal (drive forward through corridor)
        ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
            "{header: {frame_id: 'map'}, pose: {position: {x: 10.0, y: 0.0, z: 0.0}}}" &

        # Record for DURATION seconds
        sleep $DURATION

        # Save evaluation results
        ros2 topic echo --once /meta_ekf/diagnostics > "$RESULTS_DIR/${method}_${world}.txt"

        # Kill launch
        kill $LAUNCH_PID 2>/dev/null
        sleep 3

        echo "    Done. Results in $RESULTS_DIR/${method}_${world}.txt"
    done
done

echo ""
echo "============================================================="
echo "Gazebo comparison complete."
echo "Results in: $RESULTS_DIR/"
echo "============================================================="

# Summarize
echo ""
echo "SUMMARY:"
for method in "${METHODS[@]}"; do
    if [ -f "$RESULTS_DIR/${method}_indoor_lab.txt" ]; then
        echo "  $method: $(cat $RESULTS_DIR/${method}_indoor_lab.txt | head -1)"
    fi
done
 