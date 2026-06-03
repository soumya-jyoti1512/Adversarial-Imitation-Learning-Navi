#!/usr/bin/env python3

import sys
import time
from typing import Optional
try:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    from ros_gz_interfaces.srv import SetEntityPose
    from ros_gz_interfaces.msg import Entity
import math
import random
from dataclasses import dataclass


ARENA_SIZE= 5.4          
CELL_SIZE= 1.8          
OBS_HALF= 0.175        
OBS_Z= 0.15        
MIN_SEP= 0.55        

OCCUPIED_CELLS = [(0, 1), (0, 2),
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1)]


@dataclass
class CellSpec:
    row: int
    col: int

    @property
    def center(self) -> tuple[float, float]:
        cx = -1.8 + 1.8 * self.col
        cy = -1.8 + 1.8 * self.row
        return cx, cy

    @property
    def spawn_bounds(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        margin = CELL_SIZE / 2.0 - OBS_HALF
        return cx - margin, cx + margin, cy - margin, cy + margin


def sample_cell_positions(
    cell: CellSpec,
    rng: random.Random,
    max_attempts: int = 200,
) -> list[tuple[float, float]]:
    x_lo, x_hi, y_lo, y_hi = cell.spawn_bounds
    a = (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))

    for _ in range(max_attempts):
        b = (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= MIN_SEP:
            return [a, b]


def sample_all_positions(
    seed: Optional[int] = None,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    positions: dict[str, tuple[float, float]] = {}
    for row, col in OCCUPIED_CELLS:
        cell = CellSpec(row=row,col=col)
        a_xy, b_xy = sample_cell_positions(cell, rng)
        positions[f"obs_{row}{col}_a"] = a_xy
        positions[f"obs_{row}{col}_b"] = b_xy
    return positions


class ObstacleRandomizer(Node if HAS_ROS else object):

    NODE_NAME = "obstacle_randomizer"

    def __init__(
        self,
        world_name: str = "nav_arena",
        per_call_timeout: float = 1.0,
        startup_wait_timeout: float = 15.0,
    ) 
        super().__init__(self.NODE_NAME)
        self.world_name = world_name
        self.per_call_timeout = float(per_call_timeout)

        set_pose_topic = f"/world/{world_name}/set_pose"
        self._set_pose = self.create_client(SetEntityPose, set_pose_topic)
    
        self.get_logger().info("set_pose service is ready")

        self._svc = self.create_service(
            Trigger, "/randomize_obstacles", self._on_randomize
        )
        self._rng_seed_offset = int(time.time_ns() & 0xFFFFFFFF)
        self._call_count = 0


    def _on_randomize(self, request, response):
        self._call_count += 1
        seed = self._rng_seed_offset + self._call_count
        positions = sample_all_positions(seed=seed)

        n_ok = 0
        failures: list[str] = []
        for name, (x, y) in positions.items():
            ok = self._teleport(name, x, y)
            if ok:
                n_ok += 1
            else:
                failures.append(name)

        if failures:
            response.success = False
        else:
            response.success = True
        return response

  
    def _teleport(self, name: str, x: float, y: float) -> bool:
        req = SetEntityPose.Request()
        req.entity = Entity(name=name, type=Entity.MODEL)
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(OBS_Z)
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = 0.0
        req.pose.orientation.w = 1.0

        future = self._set_pose.call_async(req)

        deadline = time.time() + self.per_call_timeout
        while not future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.005)

        result = future.result()
        if result is None or not getattr(result, "success", False):
            return False
        return True


def main(argv: Optional[list[str]] = None) -> None:

    rclpy.init(args=argv)
    try:
        node = ObstacleRandomizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
