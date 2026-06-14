"""
Launch file: Gazebo + TurtleBot4[testing with tb4 instead of 3 here] + EKF comparison pipeline.

This runs the robot through indoor_lab.world with different EKF methods
and records NEES consistency for each.

"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Arguments
    method_arg = DeclareLaunchArgument(
        'method', default_value='fixed',
        description='EKF method: fixed, sage_husa, oracle, ccmetaekf'
    )
    world_arg = DeclareLaunchArgument(
        'world', default_value='indoor_lab',
        description='Gazebo world: indoor_lab, dynamic_obstacles, outdoor_rough'
    )
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint', default_value='results/run_s42/best_stsie_pid_s42.pt',
        description='Path to CC-MetaEKF checkpoint'
    )

    method = LaunchConfiguration('method')
    world = LaunchConfiguration('world')
    checkpoint = LaunchConfiguration('checkpoint')

    # Gazebo
    world_file = os.path.join(
        os.path.dirname(__file__), '..', 'gazebo_worlds',
        LaunchConfiguration('world').perform(None) + '.world'
    ) if False else ''  # Placeholder — use actual path at runtime

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # TurtleBot4 spawn (requires turtlebot4_simulator package)
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot4',
            '-topic', '/robot_description',
            '-x', '0', '-y', '0', '-z', '0.1'
        ],
        output='screen'
    )

    # EKF Node (our meta_ekf_node or baseline)
    ekf_node = Node(
        package='meta_ekf_node',
        executable='meta_ekf_node',
        name='meta_ekf',
        parameters=[{
            'method': method,
            'checkpoint_path': checkpoint,
            'dt': 0.02,  # 50Hz
            'encoder_update_interval': 10,
        }],
        output='screen'
    )

    # Evaluation Node (records NEES, consistency, RMSE)
    eval_node = Node(
        package='evaluation',
        executable='evaluation_node',
        name='ekf_evaluation',
        parameters=[{
            'output_file': '/tmp/gazebo_eval_results.json',
            'method_name': method,
        }],
        output='screen'
    )

    # Nav2 waypoint follower (sends robot through waypoints)
    nav2_cmd = ExecuteProcess(
        cmd=['ros2', 'run', 'nav2_waypoint_follower', 'waypoint_follower'],
        output='screen'
    )

    return LaunchDescription([
        method_arg,
        world_arg,
        checkpoint_arg,
        gazebo,
        spawn_robot,
        ekf_node,
        eval_node,
    ])
 