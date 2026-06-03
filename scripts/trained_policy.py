#!/usr/bin/env python3

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_FILE_ROOT = Path(__file__).resolve().parent.parent
if str(_FILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILE_ROOT))

from src.algorithms.sac import SACAgent 
from src.train import TrainConfig  
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
    )
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from std_srvs.srv import Trigger
    from ros_gz_interfaces.srv import SetEntityPose
    from ros_gz_interfaces.msg import Entity

ROBOT_MODEL_NAME= "holonomic_robot"
WORLD_NAME_DEFAULT= "nav_arena"
GOAL_XY = np.array([1.8, 1.8], dtype=np.float32)
START_XY = np.array([-1.8,-1.8], dtype=np.float32)
LIDAR_NUM_BEAMS = 20
LIDAR_MAX_RANGE= 3.0
STATE_DIM = 22
ACTION_DIM = 3
GOAL_THRESHOLD =0.30          
COLLISION_THRESHOLD = 0.15          
MAX_EPISODE_STEPS = 300           
CONTROL_HZ= 10            


def _quat_to_yaw(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _yaw_to_quat(yaw):
    return (0.0, 0.0, float(math.sin(yaw / 2.0)), float(math.cos(yaw / 2.0)))


def _compute_beam_indices(angle_min: float, angle_increment: float,
                          n_raw: int) -> np.ndarray:
                            
    target = np.linspace(0.0, 2.0 * np.pi, LIDAR_NUM_BEAMS, endpoint=False)
    target = (target + np.pi) % (2.0 * np.pi) - np.pi
    idx = np.round((target - angle_min) / angle_increment).astype(int)
    return np.clip(idx, 0, n_raw - 1)

class PolicyRunner(Node if HAS_ROS else object):

    NODE_NAME = "policy_runner"

    def __init__(
        self,
        checkpoint: str,
        config_path: Optional[str] = None,
        world_name: str = WORLD_NAME_DEFAULT,
        device: Optional[str] = None,
    ) 
        super().__init__(self.NODE_NAME)
        self.world_name = world_name

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found:{ckpt_path}")
        cfg = self._resolve_config(ckpt_path, config_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.get_logger().info(f"device = {self.device}")

        self.sac = SACAgent(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            action_scale=torch.tensor(
                [cfg.v_max, cfg.v_max, cfg.omega_max], dtype=torch.float32
            ),
            action_bias=0.0,
            hidden_dims=tuple(cfg.hidden_dims),
            automatic_entropy=cfg.automatic_entropy,
            device=self.device,
        )
        payload = torch.load(ckpt_path, map_location=self.device,
                             weights_only=False)
        self.sac.load_state_dict(payload["sac"])
        step = payload.get("step", "?")
        self.get_logger().info(
            f"loaded checkpoint @ step {step} from {ckpt_path}"
        )

        self._scan_lock = threading.Lock()
        self._odom_lock =threading.Lock()
        self._latest_scan: Optional[LaserScan] = None
        self._latest_odom: Optional[Odometry] = None
        self._beam_idx: Optional[np.ndarray] = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", reliable_qos)
        self.create_subscription(LaserScan, "/scan", self._on_scan, sensor_qos)
        self.create_subscription(Odometry, "/odom", self._on_odom, sensor_qos)

        self._randomize_client = self.create_client(
            Trigger, "/randomize_obstacles"
        )
        self._set_pose_client = self.create_client(
            SetEntityPose, f"/world/{world_name}/set_pose"
        )

        self._episode_idx = 0
        self._step_in_ep = 0
        self._rng = np.random.default_rng(int(time.time()))

        self._state = "boot"     
        self._stopped = False
        self.create_timer(1.0 / CONTROL_HZ, self._tick)

        self.get_logger().info(
            "policy_runner is ready and id waiting for /scan, /odom, and services"
        )

    @staticmethod
    def _resolve_config(ckpt: Path, explicit: Optional[str]) -> TrainConfig:
        if explicit:
            return TrainConfig.from_yaml(explicit)
        auto = ckpt.parent.parent / "config.yaml"
        if auto.exists():
            return TrainConfig.from_yaml(auto)
        return TrainConfig()

    def _on_scan(self, msg: "LaserScan") -> None:
        with self._scan_lock:
            self._latest_scan = msg
            if self._beam_idx is None:
                self._beam_idx =self._compute_beam_indices(msg)

    def _on_odom(self, msg: "Odometry") -> None:
        with self._odom_lock:
            self._latest_odom = msg

    def _compute_beam_indices(self, msg: "LaserScan") -> np.ndarray:
        return _compute_beam_indices(
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            n_raw=len(msg.ranges),
        )

    def _build_obs(self) -> Optional[np.ndarray]:
        with self._scan_lock:
            scan = self._latest_scan
            beam_idx = self._beam_idx
        with self._odom_lock:
            odom = self._latest_odom
        if scan is None or beam_idx is None or odom is None:
            return None

        ranges = np.asarray(scan.ranges, dtype=np.float32)[beam_idx]
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX_RANGE)
        ranges = np.clip(ranges, 0.0, LIDAR_MAX_RANGE)

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        q = odom.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

        dx_w = GOAL_XY[0] - x
        dy_w = GOAL_XY[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        dx_robot =  dx_w * c + dy_w * s
        dy_robot = -dx_w * s + dy_w * c

        return np.concatenate([
            ranges,
            np.array([dx_robot, dy_robot], dtype=np.float32),
        ]).astype(np.float32)

    def _robot_xy(self) -> Optional[np.ndarray]:
        with self._odom_lock:
            odom = self._latest_odom
        if odom is None:
            return None
        return np.array(
            [odom.pose.pose.position.x, odom.pose.pose.position.y],
            dtype=np.float32,
        )

   
    def _publish_action(self, a: np.ndarray) -> None:
        cmd = Twist()
        cmd.linear.x  = float(a[0])
        cmd.linear.y  = float(a[1])
        cmd.angular.z = float(a[2])
        self._cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _tick(self) -> None:
        if self._stopped:
            return

        if self._state == "boot":
            ready = (
                self._latest_scan is not None
                and self._latest_odom is not None
                and self._randomize_client.service_is_ready()
                and self._set_pose_client.service_is_ready()
            )
            if ready:
                self.get_logger().info()
                self._begin_episode()
            return

        if self._state == "running":
            self._step_episode()
            return

       
  

    def _step_episode(self) -> None:
        obs = self._build_obs()
        if obs is None:
            return 

        action =self.sac.act(obs, deterministic=True)
        self._publish_action(action)
        self._step_in_ep += 1

  
        lidar_min = float(obs[:LIDAR_NUM_BEAMS].min())
        dist_to_goal = float(np.linalg.norm(obs[LIDAR_NUM_BEAMS:]))
        reached = dist_to_goal < GOAL_THRESHOLD
        collided = lidar_min < COLLISION_THRESHOLD
        timed_out = self._step_in_ep >= MAX_EPISODE_STEPS

        if reached or collided or timed_out:
            outcome = (
                "SUCCESS" if reached
                else "COLLISION" if collided
                else "TIMEOUT"
            )
            self.get_logger().info(
                f"episode {self._episode_idx} ended: {outcome}  "
                f"steps={self._step_in_ep}  dist_to_goal={dist_to_goal:.2f} m"
            )
            self._end_episode()

   
    def _begin_episode(self) -> None:
        self._episode_idx += 1
        self._step_in_ep = 0
        self._state = "running"
        self.get_logger().info(f"episode {self._episode_idx}")

    def _end_episode(self) -> None:
        self._state = "reset"
        self._publish_stop()

        randomize_future = self._randomize_client.call_async(Trigger.Request())

        def after_randomize(_fut):
            ok = False
            try:
                result = _fut.result()
                ok = bool(result and result.success)
            

            yaw = float(self._rng.uniform(-math.pi, math.pi))
            qx, qy, qz, qw = _yaw_to_quat(yaw)
            req = SetEntityPose.Request()
            req.entity = Entity(name=ROBOT_MODEL_NAME, type=Entity.MODEL)
            req.pose.position.x =float(START_XY[0])
            req.pose.position.y = float(START_XY[1])
            req.pose.position.z= 0.05
            req.pose.orientation.x = qx
            req.pose.orientation.y = qy
            req.pose.orientation.z =qz
            req.pose.orientation.w = qw
            sp_future = self._set_pose_client.call_async(req)

            def after_teleport(_sp):
                try:
                    res = _sp.result()
                    if res is None or not getattr(res, "success", False):
                        
                except Exception as e:
                    self.get_logger().warn(f"set_pose raised:{e}")
        
                self._reset_done_at = time.time()
                self.create_timer(0.20, self._finish_reset)

            sp_future.add_done_callback(after_teleport)

        randomize_future.add_done_callback(after_randomize)

    def _finish_reset(self) -> None:
       
        if self._state == "reset":
            self._begin_episode()

   
    def shutdown(self) -> None:
        self._stopped = True
        try:
            self._publish_stop()
        except Exception:
            pass


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drive a trained SAC policy autonomously in Gazebo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint .pt file (e.g. "
                        "runs/exp1/checkpoints/latest.pt).")
    p.add_argument("--config", type=str, default=None,
                   help="Override config.yaml lookup. Auto-discovers "
                        "<run>/config.yaml next to the checkpoint by default.")
    p.add_argument("--world", type=str, default=WORLD_NAME_DEFAULT,
                   help="Name of the Gazebo world (must match the SDF).")
    p.add_argument("--device", type=str, default=None,
                   help="cpu, cuda, or cuda:N. Auto-detected if omitted.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)


    rclpy.init()
    node: Optional[PolicyRunner] = None
    try:
        node = PolicyRunner(
            checkpoint=args.checkpoint,
            config_path=args.config,
            world_name=args.world,
            device=args.device,
        )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl C")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
