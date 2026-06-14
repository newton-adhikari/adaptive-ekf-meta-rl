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

# Add project root to PYTHONPATH so meta_rl can be imported
export PYTHONPATH="$(pwd):$PYTHONPATH"

RESULTS_DIR="results/gazebo"
mkdir -p $RESULTS_DIR

METHODS=("fixed" "sage_husa" "oracle" "ccmetaekf")
WORLDS=("indoor_lab")
DURATION=60  # seconds per trial
CHECKPOINT="results/run_s42/best_stsie_pid_s42.pt"
WORLD_DIR="$(pwd)/ros2_ws/src/gazebo_worlds"

for world in "${WORLDS[@]}"; do
    echo ""
    echo "World: $world"
    echo "-------------------------------------------------------------"

    for method in "${METHODS[@]}"; do
        echo "  Method: $method"

        # Run nodes directly (bypassing ament package system)
        # Start Gazebo with world file (absolute path)
        gazebo --verbose ${WORLD_DIR}/${world}.world &
        GAZEBO_PID=$!
        sleep 5

        # Run meta_ekf_node directly (PYTHONPATH includes project root)
        python3 ros2_ws/src/meta_ekf_node/meta_ekf_node/meta_ekf_node.py \
            --ros-args -p method:=$method -p checkpoint_path:=$CHECKPOINT &
        EKF_PID=$!
        sleep 2

        # Run evaluation node
        python3 ros2_ws/src/evaluation/evaluation/evaluation_node.py \
            --ros-args -p output_file:=$RESULTS_DIR/${method}_${world}.json -p method_name:=$method &
        EVAL_PID=$!

        # Wait for system to start
        sleep 5

        # Publish a simple velocity command to drive robot forward
        ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
            "{linear: {x: 0.3}, angular: {z: 0.0}}" --rate 10 &
        CMD_PID=$!

        # Record for DURATION seconds
        sleep $DURATION

        # Save results
        echo "    Completed $DURATION seconds"

        # Kill all processes
        kill $CMD_PID $EKF_PID $EVAL_PID $GAZEBO_PID 2>/dev/null
        wait $CMD_PID $EKF_PID $EVAL_PID $GAZEBO_PID 2>/dev/null
        sleep 3

        echo "    Done. Results in $RESULTS_DIR/${method}_${world}.json"
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
 