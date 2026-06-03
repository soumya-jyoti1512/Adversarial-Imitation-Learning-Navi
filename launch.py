#!/usr/bin/env python3


import os
import sys
from pathlib import Path
try:
    from launch import LaunchDescription
    from launch.actions import (
        DeclareLaunchArgument,
        ExecuteProcess,
        OpaqueFunction,
        SetEnvironmentVariable,
        TimerAction,
    )
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node

LAUNCH_DIR  = Path(__file__).resolve().parent
FILE_ROOT= LAUNCH_DIR.parent
WORLD_FILE= REPO_ROOT / "gazebo" / "worlds" / "nav_arena.sdf"
MODELS_DIR= REPO_ROOT / "gazebo" / "models"
BRIDGE_YAML = REPO_ROOT / "config" / "ros_gz_bridge.yaml"
RANDOMIZER_PY= REPO_ROOT / "scripts" / "obstacle_randomizer_node.py"
POLICY_PY = REPO_ROOT / "scripts" / "trained_policy.py"


def _check_paths() -> None:
    for name, p in [
        ("world", WORLD_FILE),
        ("models dir", MODELS_DIR),
        ("bridge YAML",BRIDGE_YAML),
        ("randomizer",RANDOMIZER_PY),
        ("policy node", POLICY_PY),
    ]:


def _build_launch(context, *args, **kwargs):
    checkpoint= LaunchConfiguration("checkpoint").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower()
    world_name = LaunchConfiguration("world_name").perform(context)
    use_sim_time= LaunchConfiguration("use_sim_time").perform(context).lower() == "true"

    headless = gui in ("false", "0", "no")

    gz_args = ["-r", "-v", "3", str(WORLD_FILE)]
    if headless:
        gz_args.insert(0, "-s")

    gz_sim = ExecuteProcess(
        cmd=["gz", "sim"] + gz_args,
        output="screen",
        additional_env={
            "GZ_SIM_RESOURCE_PATH":
                f"{MODELS_DIR}:" + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        },
    )

    
    spawn_robot = TimerAction(
        period=2.5,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                name="spawn_holonomic_robot",
                output="screen",
                arguments=[
                    "-name",  "holonomic_robot",
                    "-file",  str(MODELS_DIR / "holonomic_robot" / "model.sdf"),
                    "-x", "-1.8",
                    "-y", "-1.8",
                    "-z", "0.05",
                    "-Y", "0.0",     
                    "-world", world_name,
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ],
    )

    topic_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="topic_bridge",
        output="screen",
        parameters=[
            {"config_file":  str(BRIDGE_YAML)},
            {"use_sim_time": use_sim_time},
        ],
    )

   
    set_pose_path = f"/world/{world_name}/set_pose"
    service_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="set_pose_service_bridge",
        output="screen",
        arguments=[
            f"{set_pose_path}@ros_gz_interfaces/srv/SetEntityPose",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

   
    randomizer = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=["python3", str(RANDOMIZER_PY)],
                name="obstacle_randomizer",
                output="screen",
                additional_env={"PYTHONPATH":
                    f"{REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")},
            ),
        ],
    )

   
    policy_runner = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "python3", str(POLICY_PY),
                    "--checkpoint", checkpoint,
                    "--world",world_name,
                ],
                name="policy_runner",
                output="screen",
                additional_env={"PYTHONPATH":
                    f"{REPO_ROOT}:" + os.environ.get("PYTHONPATH", "")},
            ),
        ],
    )

    return [
        gz_sim,
        spawn_robot,
        topic_bridge,
        service_bridge,
        randomizer,
        policy_runner,
    ]



def generate_launch_description() -> "LaunchDescription":
  
    _check_paths()

    return LaunchDescription([
        DeclareLaunchArgument(
            "checkpoint",
            description="Path to the trained policy .pt checkpoint "
                        "(produced by src/train.py)",
        ),
        DeclareLaunchArgument(
            "gui", default_value="true",
            description="Show the Gazebo GUI(false-> headless)",
        ),
        DeclareLaunchArgument(
            "world_name", default_value="nav_arena",
            description="Gazebo world name(should match the SDF)",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="If true, nodes use /clock instead of wall time.",
        ),

        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=f"{MODELS_DIR}:" + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        ),

        OpaqueFunction(function=_build_launch),
    ])
