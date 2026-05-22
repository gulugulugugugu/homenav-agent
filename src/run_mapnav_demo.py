import argparse
import heapq
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.habitat_env import HabitatEnvWrapper
from agent.dialogue import parse_target, completion_message

try:
    from perception.detector import DetrTargetDetector, DebugTargetDetector
except Exception:
    DetrTargetDetector = None
    DebugTargetDetector = None


UNKNOWN = -1
FREE = 0
OCCUPIED = 1
VISITED = 2


@dataclass
class ObjectCandidate:
    x: float
    y: float
    score: float
    count: int = 1


class MapBasedObjectNavAgent:
    """
    SemExp/VLFM-style lightweight ObjectNav.

    Core idea:
    - Use RGB only for object detection.
    - Use depth as geometry.
    - Convert depth into a top-down obstacle/free map.
    - Treat detected object as an anchor, not as a visual servoing target.
    - Navigate to a reachable free standoff cell near the object.
    """

    def __init__(self, sim_env, detector):
        self.sim_env = sim_env
        self.detector = detector

        self.target = None
        self.done = False
        self.failed = False

        # Map config.
        self.map_size_m = 12.0
        self.resolution = 0.10
        self.grid_size = int(self.map_size_m / self.resolution)
        self.center = self.grid_size // 2

        self.grid = np.full((self.grid_size, self.grid_size), UNKNOWN, dtype=np.int8)
        self.collision_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Camera / depth assumptions.
        self.camera_height_m = 1.50
        self.hfov_deg = 90.0
        self.vfov_deg = 67.5
        self.min_depth_m = 0.10
        self.max_depth_m = 5.00

        # Height filter for obstacle map.
        # Points close to floor are traversible. Points above floor are obstacles.
        self.floor_height_min = -0.10
        self.floor_height_max = 0.18
        self.obstacle_height_min = 0.18
        self.obstacle_height_max = 1.60

        # Robot safety.
        self.robot_radius_m = 0.25
        self.safety_margin_m = 0.15
        self.inflation_cells = int(math.ceil((self.robot_radius_m + self.safety_margin_m) / self.resolution))

        # Frontier exploration.
        self.min_frontier_dist_m = 0.9
        self.max_frontier_candidates = 150
        self.exhausted_frontiers = []

        # Object memory.
        self.pending = None
        self.confirmed_object = None

        # Multi-instance object memory.
        # For targets like potted plant/chair, there may be several instances.
        # We keep all confirmed instances and plan to the easiest reachable one,
        # instead of blindly using the first-confirmed instance.
        self.object_instances = []
        self.active_object_index = None
        self.instance_match_radius_m = 0.95

        self.confirm_required = 2
        self.confirm_radius_m = 0.95
        self.update_radius_m = 1.10

        # Target standoff.
        self.standoff_min_m = 0.70
        self.standoff_max_m = 1.35
        self.arrival_radius_cells = 2

        # Target planning is allowed to be more relaxed than exploration planning.
        # For "go near a table/sofa", we need to stand near furniture, not avoid it by 0.4m+.
        self.target_inflation_cells = 1
        self.target_min_standoff_m = 0.45
        self.target_preferred_standoff_m = 0.90
        self.target_max_standoff_m = 1.85
        self.target_search_radius_m = 2.60

        # Arrival validation.
        # For large furniture, "near" can be loose.
        # For small objects like potted plant, "near" must be stricter.
        self.target_arrival_max_m = 1.85
        self.strict_target_standoff = False

        # Path following.
        self.path = []
        self.path_goal = None
        self.plan_type = "none"
        self.steps_since_replan = 999
        self.replan_interval = 8

        self.step_idx = 0
        self.last_detection = None
        self.last_action = "none"

        # Path-following safety counters.
        # Avoid one-frame depth/map noise causing endless replan loops.
        self.path_blocked_steps = 0
        self.max_path_blocked_steps = 3

        # Debug-only planner diagnostics.
        # These do not change navigation decisions.
        self.debug_target_grid = None
        self.debug_standoff_all = []
        self.debug_standoff_traversible = []
        self.debug_standoff_reachable = []
        self.debug_best_path_len = 0
        self.debug_planner_reason = "not_planned"

    # -----------------------------
    # Command
    # -----------------------------

    def receive_command(self, command):
        self.target = parse_target(command)

        self.done = False
        self.failed = False
        self.pending = None
        self.confirmed_object = None
        self.object_instances = []
        self.active_object_index = None
        self.path = []
        self.path_goal = None
        self.plan_type = "none"
        self.exhausted_frontiers = []
        self.steps_since_replan = 999
        self.step_idx = 0
        self.last_detection = None
        self.last_action = "none"

        # Path-following safety counters.
        # Avoid one-frame depth/map noise causing endless replan loops.
        self.path_blocked_steps = 0
        self.max_path_blocked_steps = 3

        # Debug-only planner diagnostics.
        # These do not change navigation decisions.
        self.debug_target_grid = None
        self.debug_standoff_all = []
        self.debug_standoff_traversible = []
        self.debug_standoff_reachable = []
        self.debug_best_path_len = 0
        self.debug_planner_reason = "not_planned"

        self.grid[:] = UNKNOWN
        self.collision_grid[:] = False

        if self.target is None:
            self.failed = True
            return "我没识别出目标。你可以说：请到桌子旁边 / 请到沙发旁边。"

        # Target-specific navigation tolerance.
        # Large furniture can use loose standoff. Small objects need stricter "near".
        if self.target == "potted plant":
            self.target_min_standoff_m = 0.25
            self.target_preferred_standoff_m = 0.55
            self.target_max_standoff_m = 0.95
            self.target_search_radius_m = 1.15
            self.target_arrival_max_m = 1.05
            self.strict_target_standoff = True
            self.arrival_radius_cells = 1
        else:
            self.target_min_standoff_m = 0.45
            self.target_preferred_standoff_m = 0.90
            self.target_max_standoff_m = 1.85
            self.target_search_radius_m = 2.60
            self.target_arrival_max_m = 1.85
            self.strict_target_standoff = False
            self.arrival_radius_cells = 2

        return f"收到。我会用 depth 建 traversible map，并导航到 {self.target} 旁边。"

    # -----------------------------
    # Coordinate helpers
    # -----------------------------

    def _pose(self, obs):
        p = obs["robot_pose"]
        return float(p["x"]), float(p["y"]), float(p["yaw"])

    def _world_to_grid(self, x, y):
        gx = int(round(self.center + x / self.resolution))
        gy = int(round(self.center - y / self.resolution))
        return gx, gy

    def _grid_to_world(self, gx, gy):
        x = (gx - self.center) * self.resolution
        y = (self.center - gy) * self.resolution
        return x, y

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size

    def _current_cell(self, obs):
        x, y, _ = self._pose(obs)
        return self._world_to_grid(x, y)

    # -----------------------------
    # Depth -> 3D -> top-down map
    # -----------------------------

    def _bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 >= dy:
                err += dy
                x += sx

            if e2 <= dx:
                err += dx
                y += sy

        return points

    def _mark_free(self, gx, gy):
        if not self._in_bounds(gx, gy):
            return
        if self.grid[gy, gx] == UNKNOWN:
            self.grid[gy, gx] = FREE

    def _mark_occupied(self, gx, gy):
        if not self._in_bounds(gx, gy):
            return
        self.grid[gy, gx] = OCCUPIED

    def _depth_pixel_to_local_point(self, u, v, d, width, height):
        """
        Camera convention:
        local_x: right
        local_y: forward
        local_z: up

        Depth is treated as forward z/depth. This is an approximation but works
        well enough for a lightweight Habitat baseline.
        """
        hfov = math.radians(self.hfov_deg)
        vfov = math.radians(self.vfov_deg)

        fx = (width / 2.0) / math.tan(hfov / 2.0)
        fy = (height / 2.0) / math.tan(vfov / 2.0)

        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

        local_y = d
        local_x = (u - cx) * d / fx
        local_z = -(v - cy) * d / fy

        return local_x, local_y, local_z

    def _local_to_world_xy(self, local_x, local_y, pose):
        x, y, yaw = pose

        wx = x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
        wy = y + local_x * math.sin(yaw) + local_y * math.cos(yaw)

        return wx, wy

    def _update_map_from_depth(self, obs):
        depth = np.asarray(obs["depth"], dtype=np.float32)
        h, w = depth.shape[:2]
        pose = self._pose(obs)

        rx, ry, _ = pose
        rgx, rgy = self._world_to_grid(rx, ry)

        if self._in_bounds(rgx, rgy):
            self.grid[rgy, rgx] = VISITED

        # Sparse point cloud sampling for speed.
        # Use lower/middle image where floor and obstacles usually appear.
        us = np.linspace(int(w * 0.05), int(w * 0.95), 80).astype(int)
        vs = np.linspace(int(h * 0.28), int(h * 0.92), 55).astype(int)

        for v in vs:
            for u in us:
                d = float(depth[v, u])

                if not np.isfinite(d) or d < self.min_depth_m or d > self.max_depth_m:
                    continue

                local_x, local_y, local_z = self._depth_pixel_to_local_point(u, v, d, w, h)
                height_world = self.camera_height_m + local_z

                wx, wy = self._local_to_world_xy(local_x, local_y, pose)
                gx, gy = self._world_to_grid(wx, wy)

                if not self._in_bounds(gx, gy):
                    continue

                # Ray before endpoint is visible free space on the ground plane.
                ray = self._bresenham(rgx, rgy, gx, gy)

                for cgx, cgy in ray[:-1]:
                    self._mark_free(cgx, cgy)

                # Endpoint classification by height.
                if self.floor_height_min <= height_world <= self.floor_height_max:
                    self._mark_free(gx, gy)

                elif self.obstacle_height_min <= height_world <= self.obstacle_height_max:
                    self._mark_occupied(gx, gy)

                # Other heights are ignored: ceiling, weird points, far surfaces.

    def _inflated_obstacles(self, radius_cells=None):
        """
        Inflate obstacles by robot radius.

        radius_cells=None uses the default conservative inflation.
        Target planning can pass a smaller radius to allow standing near furniture.
        """
        occ = (self.grid == OCCUPIED) | self.collision_grid
        inflated = occ.copy()

        ys, xs = np.where(occ)

        if radius_cells is None:
            r = self.inflation_cells
        else:
            r = int(radius_cells)

        for y, x in zip(ys, xs):
            y0 = max(0, y - r)
            y1 = min(self.grid_size, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(self.grid_size, x + r + 1)
            inflated[y0:y1, x0:x1] = True

        return inflated

    def _is_traversible(self, cell, inflated=None, allow_unknown=False):
        gx, gy = cell

        if not self._in_bounds(gx, gy):
            return False

        if inflated is None:
            inflated = self._inflated_obstacles()

        if inflated[gy, gx]:
            return False

        if self.grid[gy, gx] in (FREE, VISITED):
            return True

        if allow_unknown and self.grid[gy, gx] == UNKNOWN:
            return True

        return False

    # -----------------------------
    # Object memory
    # -----------------------------

    def _estimate_object_anchor(self, obs, detection):
        """
        Object anchor from bbox + depth.
        This is only a semantic anchor, not the robot goal.
        The robot goal is a reachable free cell near this anchor.
        """
        depth = np.asarray(obs["depth"], dtype=np.float32)
        h, w = depth.shape[:2]
        pose = self._pose(obs)

        x1, y1, x2, y2 = detection["bbox"]

        # Use a central/lower area of bbox. For sofas/tables this often lands on visible object surface.
        cx = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
        cy = int(np.clip(y1 + 0.62 * (y2 - y1), 0, h - 1))

        r = 12
        patch = depth[
            max(0, cy - r): min(h, cy + r + 1),
            max(0, cx - r): min(w, cx + r + 1),
        ]

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.min_depth_m) & (valid < self.max_depth_m)]

        if valid.size == 0:
            return None

        d = float(np.median(valid))

        lx, ly, _ = self._depth_pixel_to_local_point(cx, cy, d, w, h)
        wx, wy = self._local_to_world_xy(lx, ly, pose)

        return wx, wy

    def _update_object_memory(self, obs, detection):
        """
        Multi-instance object memory.

        Old behavior:
            first-confirmed object wins forever.

        New behavior:
            maintain multiple object instances.
            each detection updates the nearest instance, or creates a new one.
            the planner later chooses the easiest reachable confirmed instance.
        """
        if detection is None:
            return

        anchor = self._estimate_object_anchor(obs, detection)

        if anchor is None:
            return

        wx, wy = anchor
        score = float(detection.get("score", 0.0))

        # Find nearest existing instance.
        best_i = None
        best_dist = float("inf")

        for i, inst in enumerate(self.object_instances):
            dist = math.hypot(wx - inst.x, wy - inst.y)
            if dist < best_dist:
                best_i = i
                best_dist = dist

        # Create new instance if far from all known instances.
        if best_i is None or best_dist > self.instance_match_radius_m:
            inst = ObjectCandidate(wx, wy, score, count=1)
            self.object_instances.append(inst)
            idx = len(self.object_instances) - 1
        else:
            inst = self.object_instances[best_i]
            prev_count = inst.count
            n = inst.count

            # Running average position update.
            inst.x = (inst.x * n + wx) / (n + 1)
            inst.y = (inst.y * n + wy) / (n + 1)
            inst.score = max(inst.score, score)
            inst.count = min(n + 1, 20)
            idx = best_i

            if prev_count < self.confirm_required and inst.count >= self.confirm_required:
                print(
                    f"[Memory] Confirmed new {self.target} instance: "
                    f"idx={idx}, x={inst.x:.2f}, y={inst.y:.2f}, "
                    f"count={inst.count}, score={inst.score:.2f}"
                )

        self.pending = self.object_instances[idx]

        confirmed_idxs = [
            i for i, inst in enumerate(self.object_instances)
            if inst.count >= self.confirm_required
        ]

        if not confirmed_idxs:
            return

        # Keep confirmed_object for backward-compatible debug fields.
        # Actual selection among confirmed objects happens inside the planner.
        if self.confirmed_object is None:
            self.active_object_index = confirmed_idxs[0]
            self.confirmed_object = self.object_instances[self.active_object_index]
            print(
                f"[Memory] First-confirmed {self.target}: "
                f"x={self.confirmed_object.x:.2f}, "
                f"y={self.confirmed_object.y:.2f}, "
                f"count={self.confirmed_object.count}, "
                f"score={self.confirmed_object.score:.2f}"
            )

    # -----------------------------
    # Planning
    # -----------------------------
    # Planning
    # -----------------------------

    def _neighbors4(self, cell):
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def _astar(self, start, goal):
        inflated = self._inflated_obstacles()

        if not self._is_traversible(start, inflated):
            return []

        if not self._is_traversible(goal, inflated):
            return []

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(0, start)]
        came = {}
        g = {start: 0}

        while open_set:
            _, cur = heapq.heappop(open_set)

            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path

            for nb in self._neighbors4(cur):
                if not self._is_traversible(nb, inflated):
                    continue

                ng = g[cur] + 1

                if nb not in g or ng < g[nb]:
                    came[nb] = cur
                    g[nb] = ng
                    heapq.heappush(open_set, (ng + h(nb, goal), nb))

        return []

    def _find_frontiers(self, start):
        inflated = self._inflated_obstacles()
        sx, sy = start
        min_d = int(self.min_frontier_dist_m / self.resolution)

        frontiers = []

        for gy in range(1, self.grid_size - 1):
            for gx in range(1, self.grid_size - 1):
                if not self._is_traversible((gx, gy), inflated):
                    continue

                if abs(gx - sx) + abs(gy - sy) < min_d:
                    continue

                local = self.grid[gy - 1:gy + 2, gx - 1:gx + 2]

                if np.any(local == UNKNOWN):
                    frontiers.append((gx, gy))

        return frontiers

    def _information_gain(self, cell, radius=5):
        gx, gy = cell

        local = self.grid[
            max(0, gy - radius): min(self.grid_size, gy + radius + 1),
            max(0, gx - radius): min(self.grid_size, gx + radius + 1),
        ]

        return int(np.sum(local == UNKNOWN))

    def _plan_to_frontier(self, start):
        frontiers = self._find_frontiers(start)

        if not frontiers:
            return []

        stride = max(1, len(frontiers) // self.max_frontier_candidates)
        candidates = frontiers[::stride]

        best_path = []
        best_score = float("inf")

        for goal in candidates:
            gain = self._information_gain(goal)

            if gain < 8:
                continue

            path = self._astar(start, goal)

            if not path:
                continue

            score = len(path) - 0.30 * gain

            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    def _sample_standoff_goals(self):
        """
        Debug-only standoff ring sampler.

        This still draws the classic ring around the object anchor, but the new planner
        no longer depends only on this ring. It is used for visualization/debug.
        """
        self.debug_target_grid = None
        self.debug_standoff_all = []
        self.debug_standoff_traversible = []
        self.debug_standoff_reachable = []
        self.debug_best_path_len = 0

        if self.confirmed_object is None:
            self.debug_planner_reason = "no_confirmed_object"
            return []

        tx, ty = self.confirmed_object.x, self.confirmed_object.y
        tgx, tgy = self._world_to_grid(tx, ty)
        self.debug_target_grid = (tgx, tgy)

        if not self._in_bounds(tgx, tgy):
            self.debug_planner_reason = "target_anchor_out_of_bounds"
            return []

        min_r = int(self.target_min_standoff_m / self.resolution)
        max_r = int(self.target_max_standoff_m / self.resolution)

        traversible_goals = []
        inflated = self._inflated_obstacles(radius_cells=self.target_inflation_cells)

        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                r = math.sqrt(dx * dx + dy * dy)

                if r < min_r or r > max_r:
                    continue

                gx = tgx + dx
                gy = tgy + dy

                if not self._in_bounds(gx, gy):
                    continue

                self.debug_standoff_all.append((gx, gy))

                if self._is_traversible((gx, gy), inflated):
                    traversible_goals.append((gx, gy))
                    self.debug_standoff_traversible.append((gx, gy))

        if not self.debug_standoff_all:
            self.debug_planner_reason = "no_standoff_ring_cells"

        elif not self.debug_standoff_traversible:
            self.debug_planner_reason = "standoff_ring_all_blocked_or_unknown"

        else:
            self.debug_planner_reason = "standoff_traversible_exists"

        return traversible_goals

    def _reachable_component(self, start, inflated):
        """
        Flood fill from the robot through currently traversible cells.

        Returns:
        - reachable set
        - parent dict for path reconstruction
        - distance dict in grid steps
        """
        if not self._is_traversible(start, inflated):
            # The robot is already physically standing here, so for planning
            # we must not let obstacle inflation erase the robot's own cell.
            sx, sy = start
            if self._in_bounds(sx, sy):
                r = max(1, self.target_inflation_cells)
                y0 = max(0, sy - r)
                y1 = min(self.grid_size, sy + r + 1)
                x0 = max(0, sx - r)
                x1 = min(self.grid_size, sx + r + 1)
                inflated[y0:y1, x0:x1] = False
                self.grid[sy, sx] = VISITED

            if not self._is_traversible(start, inflated):
                return set(), {}, {}

        q = [start]
        head = 0
        reachable = {start}
        parent = {}
        dist = {start: 0}

        while head < len(q):
            cur = q[head]
            head += 1

            for nb in self._neighbors4(cur):
                if nb in reachable:
                    continue

                if not self._is_traversible(nb, inflated):
                    continue

                reachable.add(nb)
                parent[nb] = cur
                dist[nb] = dist[cur] + 1
                q.append(nb)

        return reachable, parent, dist

    def _reconstruct_path_from_parent(self, start, goal, parent):
        if goal == start:
            return [start]

        if goal not in parent:
            return []

        path = [goal]
        cur = goal

        while cur != start:
            cur = parent[cur]
            path.append(cur)

        path.reverse()
        return path

    def _plan_to_object_standoff(self, start):
        """
        Multi-instance target planner.

        Instead of going to the first-confirmed object, evaluate every confirmed
        object instance and choose the one with the shortest reachable path.

        This matches the human rule:
            "If there are several plants/scissors/tables, go to the nearby easy one."
        """
        confirmed = [
            (i, inst)
            for i, inst in enumerate(self.object_instances)
            if inst.count >= self.confirm_required
        ]

        # Backward-compatible fallback.
        if not confirmed and self.confirmed_object is not None:
            confirmed = [(-1, self.confirmed_object)]

        if not confirmed:
            self.debug_planner_reason = "no_confirmed_object"
            return []

        global_best = None
        global_fail_debug = None

        for idx, obj in confirmed:
            tx, ty = obj.x, obj.y
            tgx, tgy = self._world_to_grid(tx, ty)

            debug = {
                "idx": idx,
                "target_grid": (tgx, tgy),
                "standoff_all": [],
                "standoff_trav": [],
                "standoff_reach": [],
                "best_len": 0,
                "reason": "not_planned",
                "path": [],
                "score": float("inf"),
            }

            if not self._in_bounds(tgx, tgy):
                debug["reason"] = "target_anchor_out_of_bounds"
                global_fail_debug = debug
                continue

            min_r = int(self.target_min_standoff_m / self.resolution)
            max_r = int(self.target_max_standoff_m / self.resolution)

            inflated = self._inflated_obstacles(radius_cells=self.target_inflation_cells)

            # Build standoff debug ring for this instance.
            for dy in range(-max_r, max_r + 1):
                for dx in range(-max_r, max_r + 1):
                    rr = math.sqrt(dx * dx + dy * dy)

                    if rr < min_r or rr > max_r:
                        continue

                    gx = tgx + dx
                    gy = tgy + dy

                    if not self._in_bounds(gx, gy):
                        continue

                    debug["standoff_all"].append((gx, gy))

                    if self._is_traversible((gx, gy), inflated):
                        debug["standoff_trav"].append((gx, gy))

            reachable, parent, dist_steps = self._reachable_component(start, inflated)

            if not reachable:
                debug["reason"] = "robot_not_in_reachable_free_component"
                global_fail_debug = debug
                continue

            best_cell = None
            best_score = float("inf")
            reachable_candidates = []

            max_radius_cells = int(self.target_search_radius_m / self.resolution)

            for cell in reachable:
                gx, gy = cell
                dx = gx - tgx
                dy = gy - tgy
                d_cells = math.sqrt(dx * dx + dy * dy)

                if d_cells > max_radius_cells:
                    continue

                d_m = d_cells * self.resolution

                # Strict near-goal band:
                # the goal must be genuinely near the object anchor.
                if d_m < self.target_min_standoff_m or d_m > self.target_max_standoff_m:
                    continue

                reachable_candidates.append(cell)

                standoff_error = abs(d_m - self.target_preferred_standoff_m)
                path_len = dist_steps.get(cell, 999999)

                # Main preference: actually be near the object.
                # Secondary preference: shorter path.
                score = 80.0 * standoff_error + 0.05 * path_len

                if score < best_score:
                    best_score = score
                    best_cell = cell

            debug["standoff_reach"] = reachable_candidates[:900]

            if best_cell is None:
                if self.strict_target_standoff:
                    debug["reason"] = "no_strict_reachable_standoff_near_target"
                else:
                    debug["reason"] = "no_reachable_cell_near_target_anchor"
                global_fail_debug = debug
                continue

            path = self._reconstruct_path_from_parent(start, best_cell, parent)

            if not path:
                debug["reason"] = "parent_reconstruction_failed"
                global_fail_debug = debug
                continue

            debug["path"] = path
            debug["best_len"] = len(path)
            debug["score"] = best_score
            debug["reason"] = "nearest_reachable_instance_found"

            if global_best is None or debug["score"] < global_best["score"]:
                global_best = debug

        if global_best is None:
            if global_fail_debug is not None:
                self.debug_target_grid = global_fail_debug["target_grid"]
                self.debug_standoff_all = global_fail_debug["standoff_all"]
                self.debug_standoff_traversible = global_fail_debug["standoff_trav"]
                self.debug_standoff_reachable = global_fail_debug["standoff_reach"]
                self.debug_best_path_len = global_fail_debug["best_len"]
                self.debug_planner_reason = global_fail_debug["reason"]
            else:
                self.debug_planner_reason = "no_instance_plan_debug"
            return []

        # Activate the selected nearest/easiest confirmed instance.
        chosen_idx = global_best["idx"]

        if chosen_idx >= 0:
            if self.active_object_index != chosen_idx:
                chosen = self.object_instances[chosen_idx]
                print(
                    f"[Memory] Switching active {self.target} instance: "
                    f"idx={chosen_idx}, x={chosen.x:.2f}, y={chosen.y:.2f}, "
                    f"path_len={global_best['best_len']}"
                )

            self.active_object_index = chosen_idx
            self.confirmed_object = self.object_instances[chosen_idx]

        self.debug_target_grid = global_best["target_grid"]
        self.debug_standoff_all = global_best["standoff_all"]
        self.debug_standoff_traversible = global_best["standoff_trav"]
        self.debug_standoff_reachable = global_best["standoff_reach"]
        self.debug_best_path_len = global_best["best_len"]
        self.debug_planner_reason = global_best["reason"]

        self.path_goal = global_best["path"][-1]
        return global_best["path"]

    def _need_replan(self):
        return (not self.path) or self.steps_since_replan >= self.replan_interval

    def _plan(self, obs):
        start = self._current_cell(obs)

        if self.confirmed_object is not None:
            path = self._plan_to_object_standoff(start)

            if path:
                self.path = path
                self.path_goal = path[-1]
                self.plan_type = "target_standoff"
                self.steps_since_replan = 0
                self.path_blocked_steps = 0
                return "target_standoff"

            # If no standoff path, do not chase bbox. Continue exploration to build map.
            path = self._plan_to_frontier(start)

            if path:
                self.path = path
                self.path_goal = path[-1]
                self.plan_type = "frontier_after_target"
                self.steps_since_replan = 0
                self.path_blocked_steps = 0
                return "frontier_after_target"

            self.path = []
            self.path_goal = None
            self.plan_type = "no_target_path"
            return "no_target_path"

        path = self._plan_to_frontier(start)

        if path:
            self.path = path
            self.path_goal = path[-1]
            self.plan_type = "frontier"
            self.steps_since_replan = 0
            self.path_blocked_steps = 0
            return "frontier"

        self.path = []
        self.path_goal = None
        self.plan_type = "none"
        return "none"

    # -----------------------------
    # Path follower
    # -----------------------------

    def _angle_to_cell(self, obs, cell):
        x, y, yaw = self._pose(obs)
        wx, wy = self._grid_to_world(*cell)

        dx = wx - x
        dy = wy - y

        # yaw=0 faces +y.
        desired = math.atan2(-dx, dy)
        diff = desired - yaw
        return math.atan2(math.sin(diff), math.cos(diff))

    def _inflated_for_current_plan(self):
        """
        Planner/follower consistency.

        target_standoff was planned with relaxed inflation, so the follower must
        validate the next cell with the same relaxed inflation. Otherwise the
        planner says "reachable" and the follower immediately says "blocked".
        """
        if self.plan_type == "target_standoff":
            return self._inflated_obstacles(radius_cells=self.target_inflation_cells)

        return self._inflated_obstacles()

    def _target_arrival_is_valid(self, obs):
        """
        A target_standoff path ending is not enough for small objects.
        We also require the robot to be physically near the active object anchor.
        """
        if self.confirmed_object is None:
            return False, "no_confirmed_object"

        rx, ry, _ = self._pose(obs)
        dist = math.hypot(rx - self.confirmed_object.x, ry - self.confirmed_object.y)

        if dist <= self.target_arrival_max_m:
            return True, f"anchor_dist={dist:.2f}"

        return False, f"anchor_dist={dist:.2f} > max={self.target_arrival_max_m:.2f}"

    def _follow_path(self, obs):
        if not self.path:
            self.sim_env.turn_left()
            self.last_action = "turn_left"
            return "[NO-PATH] No path available. Rotating to build map."

        start = self._current_cell(obs)

        # Drop path points already reached.
        while len(self.path) > 1 and self.path[0] == start:
            self.path.pop(0)

        if len(self.path) <= self.arrival_radius_cells:
            if self.plan_type == "target_standoff":
                ok, why = self._target_arrival_is_valid(obs)

                if ok:
                    self.sim_env.stop()
                    self.last_action = "stop"
                    self.done = True
                    return completion_message() + f" [MAP-ARRIVED] reached reachable standoff cell. {why}"

                # Do not accept a too-far standoff, especially for small objects.
                self.path = []
                self.path_goal = None
                self.steps_since_replan = 999
                self.sim_env.turn_left()
                self.last_action = "turn_left"
                return f"[ARRIVAL-REJECTED] Reached path end but not actually near target. {why}. Replanning."

            if self.path_goal is not None and "frontier" in self.plan_type:
                self.exhausted_frontiers.append(self.path_goal)

            self.path = []
            self.path_goal = None
            self.steps_since_replan = 999
            self.sim_env.turn_left()
            self.last_action = "turn_left"
            return f"[FRONTIER] Reached frontier for map expansion. plan={self.plan_type}"

        lookahead = min(3, len(self.path) - 1)
        next_cell = self.path[lookahead]
        angle = self._angle_to_cell(obs, next_cell)

        if angle > math.radians(12):
            self.sim_env.turn_left()
            self.last_action = "turn_left"
            return f"[PATH] Turning left toward {self.plan_type}. angle={math.degrees(angle):.1f}"

        if angle < -math.radians(12):
            self.sim_env.turn_right()
            self.last_action = "turn_right"
            return f"[PATH] Turning right toward {self.plan_type}. angle={math.degrees(angle):.1f}"

        # Extra local safety: next immediate cell should still be traversible.
        # Important: use the same inflation policy as the active planner.
        inflated = self._inflated_for_current_plan()

        if len(self.path) > 1 and not self._is_traversible(self.path[1], inflated):
            self.path_blocked_steps += 1

            # Do not instantly clear a valid target path because of one noisy scan.
            if self.path_blocked_steps < self.max_path_blocked_steps:
                if angle < 0:
                    self.sim_env.turn_right()
                    self.last_action = "turn_right"
                    side = "right"
                else:
                    self.sim_env.turn_left()
                    self.last_action = "turn_left"
                    side = "left"

                return (
                    f"[PATH-BLOCKED-SOFT] Next cell temporarily blocked under {self.plan_type}. "
                    f"blocked_steps={self.path_blocked_steps}/{self.max_path_blocked_steps}. "
                    f"Turning {side} and keeping path."
                )

            self.path = []
            self.path_goal = None
            self.steps_since_replan = 999
            self.path_blocked_steps = 0
            self.sim_env.turn_left()
            self.last_action = "turn_left"
            return (
                f"[PATH-BLOCKED-HARD] Next cell still non-traversible after repeated checks. "
                f"Replanning. plan={self.plan_type}"
            )

        self.path_blocked_steps = 0
        self.sim_env.move_forward()
        self.last_action = "move_forward"
        return f"[PATH] Moving forward along {self.plan_type}. remaining={len(self.path)}"

    # -----------------------------
    # Main step
    # -----------------------------

    def step(self):
        self.step_idx += 1

        obs = self.sim_env.get_observation()

        if self.done:
            return completion_message()

        if self.failed:
            return "[FAILED]"

        self._update_map_from_depth(obs)

        rgb = obs["rgb"][:, :, :3]
        detection = self.detector.detect(rgb, self.target)
        self.last_detection = detection

        if detection is not None:
            self._update_object_memory(obs, detection)
            # Force replan after confirming or updating target anchor.
            if self.confirmed_object is not None:
                self.path = []
                self.path_goal = None
                self.steps_since_replan = 999

        self.steps_since_replan += 1

        if self._need_replan():
            self._plan(obs)

        msg = self._follow_path(obs)

        free = int(np.sum((self.grid == FREE) | (self.grid == VISITED)))
        occ = int(np.sum(self.grid == OCCUPIED))
        memory = 1 if self.confirmed_object is not None else 0
        pending = 0 if self.pending is None else self.pending.count

        det_msg = "none"
        if detection is not None:
            det_msg = (
                f"{detection['label']} score={detection['score']:.2f} "
                f"bbox={detection['bbox']}"
            )

        return (
            f"{msg} | plan={self.plan_type}, det={det_msg}, "
            f"memory={memory}, pending={pending}, free={free}, occ={occ}, "
            f"instances={len(self.object_instances)}, "
            f"active={self.active_object_index}, "
            f"standoff_all={len(self.debug_standoff_all)}, "
            f"standoff_trav={len(self.debug_standoff_traversible)}, "
            f"standoff_reach={len(self.debug_standoff_reachable)}, "
            f"best_len={self.debug_best_path_len}, "
            f"reason={self.debug_planner_reason}"
        )

    # -----------------------------
    # Debug visualization helpers
    # -----------------------------

    def draw_map_overlay(self, size=180):
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        img[self.grid == UNKNOWN] = [30, 30, 30]
        img[self.grid == FREE] = [210, 210, 210]
        img[self.grid == VISITED] = [80, 180, 255]
        img[self.grid == OCCUPIED] = [255, 80, 80]

        # Debug: raw standoff ring candidates around object anchor.
        for gx, gy in self.debug_standoff_all:
            if self._in_bounds(gx, gy):
                img[gy, gx] = [180, 110, 40]

        # Debug: traversible standoff candidates.
        for gx, gy in self.debug_standoff_traversible:
            if self._in_bounds(gx, gy):
                img[gy, gx] = [0, 220, 220]

        # Debug: reachable standoff candidates.
        for gx, gy in self.debug_standoff_reachable:
            if self._in_bounds(gx, gy):
                img[gy, gx] = [80, 120, 255]

        # Path.
        for gx, gy in self.path:
            if self._in_bounds(gx, gy):
                img[gy, gx] = [80, 255, 80]

        # Robot.
        obs = self.sim_env.get_observation()
        rgx, rgy = self._current_cell(obs)
        if self._in_bounds(rgx, rgy):
            img[max(0, rgy - 2):min(self.grid_size, rgy + 3),
                max(0, rgx - 2):min(self.grid_size, rgx + 3)] = [255, 255, 0]

        # Object anchor.
        if self.confirmed_object is not None:
            ogx, ogy = self._world_to_grid(self.confirmed_object.x, self.confirmed_object.y)
            if self._in_bounds(ogx, ogy):
                img[max(0, ogy - 3):min(self.grid_size, ogy + 4),
                    max(0, ogx - 3):min(self.grid_size, ogx + 4)] = [255, 0, 255]

        pil = Image.fromarray(img).resize((size, size), Image.Resampling.NEAREST)
        return pil


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def draw_decision_frame(obs, agent, msg, out_path):
    rgb = obs["rgb"][:, :, :3]
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Header.
    header_h = 74
    canvas = Image.new("RGB", (img.width, img.height + header_h), (0, 0, 0))
    canvas.paste(img, (0, header_h))
    draw = ImageDraw.Draw(canvas)

    text = f"step={agent.step_idx} | {msg[:180]}"
    draw.text((8, 8), text, fill=(255, 255, 255))

    # Detection bbox.
    det = agent.last_detection
    if det is not None:
        x1, y1, x2, y2 = det["bbox"]
        yoff = header_h
        draw.rectangle([x1, y1 + yoff, x2, y2 + yoff], outline=(255, 0, 0), width=4)
        draw.rectangle([x1, y1 + yoff - 22, min(x2, x1 + 220), y1 + yoff], fill=(255, 0, 0))
        draw.text((x1 + 4, y1 + yoff - 20), f"{det['label']} {det['score']:.2f}", fill=(255, 255, 255))

    # Map overlay.
    overlay = agent.draw_map_overlay(size=180)
    canvas.paste(overlay, (canvas.width - 190, header_h + 10))

    canvas.save(out_path)


def save_evidence_frame(obs, detection, out_path):
    rgb = obs["rgb"][:, :, :3]
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = detection["bbox"]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    draw.text((x1 + 4, max(0, y1 - 20)), f"{detection['label']} {detection['score']:.2f}", fill=(255, 0, 0))

    img.save(out_path)


def build_detector(args):
    if args.detector == "debug":
        if DebugTargetDetector is None:
            raise RuntimeError("DebugTargetDetector is not available in perception.detector")
        return DebugTargetDetector()

    if DetrTargetDetector is None:
        raise RuntimeError("DetrTargetDetector is not available in perception.detector")

    return DetrTargetDetector(score_threshold=args.score_threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="请到桌子旁边")
    parser.add_argument("--out", type=str, default="demos/mapnav_test")
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--detector", type=str, default="detr", choices=["detr", "debug"])
    parser.add_argument("--score_threshold", type=float, default=0.60)
    args = parser.parse_args()

    safe_mkdir(args.out)

    frames_dir = os.path.join(args.out, "frames")
    decision_dir = os.path.join(args.out, "decision_frames")
    evidence_dir = os.path.join(args.out, "evidence")

    safe_mkdir(frames_dir)
    safe_mkdir(decision_dir)
    safe_mkdir(evidence_dir)

    detector = build_detector(args)
    env = HabitatEnvWrapper(output_dir=frames_dir)
    env.reset()

    agent = MapBasedObjectNavAgent(env, detector)

    print(f"User> {args.command}")
    reply = agent.receive_command(args.command)
    print(f"Agent> {reply}")

    logs = []

    for step in range(args.max_steps):
        msg = agent.step()
        obs = env.get_observation()

        evidence_path = None
        if agent.last_detection is not None:
            evidence_path = os.path.join(
                evidence_dir,
                f"step_{step:04d}_{agent.last_detection['label'].replace(' ', '_')}_{agent.last_detection['score']:.2f}.png",
            )
            save_evidence_frame(obs, agent.last_detection, evidence_path)
            print(f"Evidence> {evidence_path}")

        decision_path = os.path.join(decision_dir, f"decision_{step:04d}.png")
        draw_decision_frame(obs, agent, msg, decision_path)

        print(f"Agent> {msg}")

        logs.append({
            "step": step,
            "agent_message": msg,
            "plan": agent.plan_type,
            "detection": agent.last_detection,
            "evidence_frame": evidence_path,
            "done": agent.done,
            "debug": {
                "target_grid": agent.debug_target_grid,
                "standoff_all": len(agent.debug_standoff_all),
                "standoff_traversible": len(agent.debug_standoff_traversible),
                "standoff_reachable": len(agent.debug_standoff_reachable),
                "best_path_len": agent.debug_best_path_len,
                "planner_reason": agent.debug_planner_reason,
            },
        })

        if agent.done or agent.failed:
            break

    if not agent.done and not agent.failed:
        print("Agent> 任务未在最大步数内完成。")

    with open(os.path.join(args.out, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    env.close()

    print(f"Saved action frames to: {frames_dir}")
    print(f"Saved decision frames to: {decision_dir}")
    print(f"Saved evidence frames to: {evidence_dir}")
    print(f"Saved log to: {os.path.join(args.out, 'run_log.json')}")


if __name__ == "__main__":
    main()
