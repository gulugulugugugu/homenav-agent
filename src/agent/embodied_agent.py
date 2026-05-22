import heapq
import math

import numpy as np

from agent.state_machine import AgentState
from agent.dialogue import parse_target, completion_message


UNKNOWN = -1
FREE = 0
OCCUPIED = 1
VISITED = 2


class EmbodiedNavAgent:
    """
    VLFM-lite / SemExp-lite ObjectNav baseline.

    Idea:
    - Build a 2D occupancy map from depth using ray casting.
    - If the target is not confirmed, explore useful frontiers.
    - If the target is detected repeatedly, lock the first confirmed instance.
    - If the target is visible, use visual final approach.
    - If target is not visible but confirmed, plan to a free standoff cell near it.

    Used:
    - RGB observation
    - Depth observation
    - Robot odometry from executed actions

    Not used:
    - Simulator target coordinates
    - Semantic scene graph
    - Shortest-path oracle
    """

    def __init__(self, sim_env, detector):
        self.sim_env = sim_env
        self.detector = detector

        self.state = AgentState.IDLE
        self.target = None
        self.last_detection = None

        # Map config.
        self.map_size_m = 12.0
        self.resolution = 0.10
        self.grid_size = int(self.map_size_m / self.resolution)
        self.center = self.grid_size // 2
        self.grid = np.full((self.grid_size, self.grid_size), UNKNOWN, dtype=np.int8)

        # Robot footprint.
        self.robot_radius_m = 0.25
        self.safety_margin_m = 0.15
        self.inflation_radius_cells = int(
            math.ceil((self.robot_radius_m + self.safety_margin_m) / self.resolution)
        )

        # Camera assumptions.
        self.hfov_deg = 90.0
        self.max_depth_m = 5.0
        self.min_depth_m = 0.10

        # First-confirmed object instance memory.
        self.object_memory = {}
        self.pending_object_memory = {}
        self.confirm_required_count = 2
        self.confirm_cluster_radius_m = 0.85
        self.update_cluster_radius_m = 0.95

        # Planning.
        self.path = []
        self.path_goal = None
        self.current_plan_type = "none"
        self.replan_interval = 8
        self.steps_since_replan = 999
        self.exhausted_frontiers = []

        # Frontier selection.
        self.min_frontier_distance_m = 0.90
        self.max_frontier_candidates = 120

        # Final visual approach.
        self.target_standoff_m = 0.85
        self.standoff_tolerance_m = 0.30
        self.safe_front_dist_m = 0.65

        # Low-look traversability gate.
        # This uses down_depth to decide whether the ground corridor ahead is open.
        self.use_low_look = True
        self.low_look_min_p10 = 0.25
        self.low_look_min_median = 0.55
        self.low_look_max_close_ratio = 0.50

        self.target_visible_streak = 0
        self.target_lost_steps = 0
        self.target_lock_lost_limit = 18
        self.last_target_side = "center"

        # Wider visual centering threshold.
        # If target is roughly in the front view, move forward instead of over-turning.
        self.visual_left_bound = 0.30
        self.visual_right_bound = 0.78

        self.last_seen_distance = float("inf")
        self.last_seen_front = float("inf")
        self.visual_forward_steps = 0

        # If we saw the target once, pause frontier briefly to confirm it.
        self.candidate_lock_steps = 0
        self.max_candidate_lock_steps = 12

        self.target_path_forward_steps = 0

        self.step_count = 0

    # -----------------------------
    # Command interface
    # -----------------------------

    def receive_command(self, command: str):
        self.state = AgentState.PARSE_COMMAND
        self.target = parse_target(command)

        self.path = []
        self.path_goal = None
        self.current_plan_type = "none"
        self.steps_since_replan = 999
        self.exhausted_frontiers = []

        self.object_memory = {}
        self.pending_object_memory = {}

        self.last_detection = None
        self.target_visible_streak = 0
        self.target_lost_steps = 0
        self.last_target_side = "center"
        self.last_seen_distance = float("inf")
        self.last_seen_front = float("inf")
        self.visual_forward_steps = 0
        self.candidate_lock_steps = 0
        self.target_path_forward_steps = 0
        self.step_count = 0

        if self.target is None:
            self.state = AgentState.FAILED
            return (
                "我没识别出目标。你可以说：请到沙发旁边 / "
                "请到桌子旁边 / 请到椅子旁边。"
            )

        self.state = AgentState.SEARCH_TARGET
        return (
            f"收到。我会用 RGB-D 建图、探索 frontier，"
            f"并导航到 first-confirmed {self.target} 旁边。"
        )

    # -----------------------------
    # Coordinates
    # -----------------------------

    def _pose(self, obs):
        p = obs["robot_pose"]
        return p["x"], p["y"], p["yaw"]

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

    def _current_grid_cell(self, obs):
        x, y, _ = self._pose(obs)
        return self._world_to_grid(x, y)

    # -----------------------------
    # Mapping
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

    def _update_occupancy_from_depth(self, depth, pose):
        x, y, yaw = pose
        h, w = depth.shape[:2]

        robot_gx, robot_gy = self._world_to_grid(x, y)
        if self._in_bounds(robot_gx, robot_gy):
            self.grid[robot_gy, robot_gx] = VISITED

        cols = np.linspace(int(w * 0.06), int(w * 0.94), 55).astype(int)
        rows = [int(h * 0.45), int(h * 0.58), int(h * 0.70)]

        hfov = math.radians(self.hfov_deg)

        for col in cols:
            valid_depths = []

            for row in rows:
                d = float(depth[row, col])
                if np.isfinite(d) and self.min_depth_m < d < self.max_depth_m:
                    valid_depths.append(d)

            if not valid_depths:
                ray_dist = self.max_depth_m
                hit_obstacle = False
            else:
                ray_dist = min(valid_depths)
                hit_obstacle = ray_dist < self.max_depth_m * 0.98

            rel = col / max(1, w - 1) - 0.5
            angle = rel * hfov

            local_x = math.sin(angle) * ray_dist
            local_y = math.cos(angle) * ray_dist

            wx = x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
            wy = y + local_x * math.sin(yaw) + local_y * math.cos(yaw)

            end_gx, end_gy = self._world_to_grid(wx, wy)

            if not self._in_bounds(end_gx, end_gy):
                continue

            ray_cells = self._bresenham(robot_gx, robot_gy, end_gx, end_gy)

            for cgx, cgy in ray_cells[:-1]:
                self._mark_free(cgx, cgy)

            if hit_obstacle:
                self._mark_occupied(end_gx, end_gy)
            else:
                self._mark_free(end_gx, end_gy)

    def _inflated_obstacle_mask(self):
        occ = self.grid == OCCUPIED
        inflated = np.copy(occ)

        ys, xs = np.where(occ)
        r = self.inflation_radius_cells

        for y, x in zip(ys, xs):
            y0 = max(0, y - r)
            y1 = min(self.grid_size, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(self.grid_size, x + r + 1)
            inflated[y0:y1, x0:x1] = True

        return inflated

    def _is_traversable(self, gx, gy, inflated=None):
        if not self._in_bounds(gx, gy):
            return False

        if inflated is None:
            inflated = self._inflated_obstacle_mask()

        if inflated[gy, gx]:
            return False

        return self.grid[gy, gx] in (FREE, VISITED)

    # -----------------------------
    # Object memory: first-confirmed instance
    # -----------------------------

    def _estimate_object_position(self, depth, detection, pose):
        if detection is None:
            return None

        x, y, yaw = pose
        h, w = depth.shape[:2]

        x1, y1, x2, y2 = detection["bbox"]
        cx = int(max(0, min(w - 1, (x1 + x2) / 2)))
        cy = int(max(0, min(h - 1, (y1 + y2) / 2)))

        r = 10
        patch = depth[
            max(0, cy - r): min(h, cy + r + 1),
            max(0, cx - r): min(w, cx + r + 1),
        ]

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.min_depth_m) & (valid < self.max_depth_m)]

        if valid.size == 0:
            return None

        d = float(np.median(valid))

        hfov = math.radians(self.hfov_deg)
        rel = cx / max(1, w - 1) - 0.5
        angle = rel * hfov

        local_x = math.sin(angle) * d
        local_y = math.cos(angle) * d

        wx = x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
        wy = y + local_x * math.sin(yaw) + local_y * math.cos(yaw)

        return wx, wy

    def _get_target_memory_position(self):
        items = self.object_memory.get(self.target, [])
        if not items:
            return None

        scores = np.array([max(1e-3, it["score"]) for it in items], dtype=float)
        xs = np.array([it["x"] for it in items], dtype=float)
        ys = np.array([it["y"] for it in items], dtype=float)

        return (
            float(np.sum(xs * scores) / np.sum(scores)),
            float(np.sum(ys * scores) / np.sum(scores)),
        )

    def _update_object_memory(self, depth, detection, pose):
        if detection is None:
            return

        if detection.get("score", 0.0) < 0.60:
            return

        pos = self._estimate_object_position(depth, detection, pose)
        if pos is None:
            return

        wx, wy = pos

        # Already confirmed: only update if this detection belongs to same instance.
        if self.object_memory.get(self.target):
            current = self._get_target_memory_position()
            if current is None:
                return

            cx, cy = current
            dist = math.hypot(wx - cx, wy - cy)

            if dist <= self.update_cluster_radius_m:
                self.object_memory[self.target].append({
                    "x": wx,
                    "y": wy,
                    "score": float(detection["score"]),
                })
                self.object_memory[self.target] = self.object_memory[self.target][-8:]

            return

        # Not confirmed yet: build pending candidate.
        pending = self.pending_object_memory.get(self.target)

        if pending is None:
            self.pending_object_memory[self.target] = {
                "x": wx,
                "y": wy,
                "score": float(detection["score"]),
                "count": 1,
            }
            return

        dist = math.hypot(wx - pending["x"], wy - pending["y"])

        if dist <= self.confirm_cluster_radius_m:
            old_count = pending["count"]
            new_count = old_count + 1
            pending["x"] = (pending["x"] * old_count + wx) / new_count
            pending["y"] = (pending["y"] * old_count + wy) / new_count
            pending["score"] = max(float(pending["score"]), float(detection["score"]))
            pending["count"] = new_count

            if pending["count"] >= self.confirm_required_count:
                self.object_memory[self.target] = [{
                    "x": pending["x"],
                    "y": pending["y"],
                    "score": pending["score"],
                }]
                print(
                    f"[Memory] First-confirmed {self.target}: "
                    f"x={pending['x']:.2f}, y={pending['y']:.2f}, "
                    f"count={pending['count']}, score={pending['score']:.2f}"
                )

            return

        # Different instance before confirmation: reset pending.
        self.pending_object_memory[self.target] = {
            "x": wx,
            "y": wy,
            "score": float(detection["score"]),
            "count": 1,
        }

    # -----------------------------
    # Frontier and A*
    # -----------------------------

    def _neighbors4(self, cell):
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def _astar(self, start, goal):
        inflated = self._inflated_obstacle_mask()

        if not self._is_traversable(*start, inflated=inflated):
            return []

        if not self._is_traversable(*goal, inflated=inflated):
            return []

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(0, start)]
        came = {}
        gscore = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came:
                    current = came[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self._neighbors4(current):
                if not self._is_traversable(*nb, inflated=inflated):
                    continue

                ng = gscore[current] + 1

                if nb not in gscore or ng < gscore[nb]:
                    came[nb] = current
                    gscore[nb] = ng
                    heapq.heappush(open_set, (ng + h(nb, goal), nb))

        return []

    def _information_gain(self, gx, gy, radius=5):
        local = self.grid[
            max(0, gy - radius): min(self.grid_size, gy + radius + 1),
            max(0, gx - radius): min(self.grid_size, gx + radius + 1),
        ]
        return int(np.sum(local == UNKNOWN))

    def _is_exhausted_frontier(self, gx, gy):
        for ex, ey in self.exhausted_frontiers[-30:]:
            if abs(gx - ex) <= 4 and abs(gy - ey) <= 4:
                return True
        return False

    def _find_frontiers(self, start):
        sx, sy = start
        min_dist_cells = int(self.min_frontier_distance_m / self.resolution)

        inflated = self._inflated_obstacle_mask()
        frontiers = []

        for gy in range(1, self.grid_size - 1):
            for gx in range(1, self.grid_size - 1):
                if not self._is_traversable(gx, gy, inflated=inflated):
                    continue

                dist = abs(gx - sx) + abs(gy - sy)
                if dist < min_dist_cells:
                    continue

                if self._is_exhausted_frontier(gx, gy):
                    continue

                local = self.grid[gy - 1: gy + 2, gx - 1: gx + 2]
                if np.any(local == UNKNOWN):
                    frontiers.append((gx, gy))

        return frontiers

    def _plan_to_best_frontier(self, start):
        frontiers = self._find_frontiers(start)
        if not frontiers:
            return []

        # Downsample candidates for speed.
        stride = max(1, len(frontiers) // self.max_frontier_candidates)
        candidates = frontiers[::stride]

        best_path = []
        best_score = float("inf")

        for goal in candidates:
            gain = self._information_gain(*goal)
            if gain < 8:
                continue

            path = self._astar(start, goal)
            if not path:
                continue

            # Prefer information gain, but avoid very long paths.
            score = len(path) - 0.35 * gain

            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    def _plan_to_target_standoff(self, start):
        target_pos = self._get_target_memory_position()
        if target_pos is None:
            return []

        tx, ty = target_pos
        tgx, tgy = self._world_to_grid(tx, ty)

        if not self._in_bounds(tgx, tgy):
            return []

        inflated = self._inflated_obstacle_mask()

        min_r = int(0.55 / self.resolution)
        max_r = int(1.20 / self.resolution)

        best_path = []
        best_len = float("inf")

        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                r = math.sqrt(dx * dx + dy * dy)

                if r < min_r or r > max_r:
                    continue

                gx = tgx + dx
                gy = tgy + dy

                if not self._is_traversable(gx, gy, inflated=inflated):
                    continue

                path = self._astar(start, (gx, gy))
                if path and len(path) < best_len:
                    best_len = len(path)
                    best_path = path

        return best_path

    def _need_replan(self):
        if not self.path:
            return True
        if self.steps_since_replan >= self.replan_interval:
            return True
        return False

    def _plan(self, obs):
        start = self._current_grid_cell(obs)

        # GitHub-style hierarchy:
        # once a target instance is confirmed, it becomes the long-term goal.
        # Frontier exploration should not take control anymore.
        if self.object_memory.get(self.target):
            target_path = self._plan_to_target_standoff(start)
            if target_path:
                self.path = target_path
                self.path_goal = target_path[-1]
                self.current_plan_type = "target"
                self.target_path_forward_steps = 0
                self.steps_since_replan = 0
                return "target"

            self.path = []
            self.path_goal = None
            self.current_plan_type = "target_reacquire"
            return "target_reacquire"

        # Before target confirmation, explore frontiers.
        frontier_path = self._plan_to_best_frontier(start)
        if frontier_path:
            self.path = frontier_path
            self.path_goal = frontier_path[-1]
            self.current_plan_type = "frontier"
            self.steps_since_replan = 0
            return "frontier"

        self.path = []
        self.path_goal = None
        self.current_plan_type = "none"
        return "none"

    # -----------------------------
    # Visual final approach
    # -----------------------------

    def _front_clearance(self, depth):
        h, w = depth.shape[:2]
        patch = depth[int(h * 0.45): int(h * 0.88), int(w * 0.36): int(w * 0.64)]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 20.0)]
        if valid.size == 0:
            return float("inf")
        return float(np.percentile(valid, 12))

    def _ensure_floor_model_fields(self):
        if not hasattr(self, "floor_depth_model"):
            self.floor_depth_model = None
            self.floor_model_ready = False
            self.floor_model_updates = 0

            # Floor model parameters.
            self.floor_model_x0 = 0.18
            self.floor_model_x1 = 0.82
            self.floor_model_y0 = 0.18
            self.floor_model_y1 = 0.82

            # A pixel is floor-like if it is close to expected floor depth for that row.
            self.floor_abs_tol = 0.22
            self.floor_rel_tol = 0.22

            # If depth is much smaller than expected floor, something is blocking the floor.
            self.obstacle_rel_margin = 0.30

            # Corridor thresholds.
            self.floor_min_ratio = 0.28
            self.obstacle_max_ratio = 0.38
            self.valid_min_ratio = 0.35

    def _update_floor_depth_model(self, obs):
        """
        Learn a row-wise expected floor depth from down_depth.

        This is not simply "large depth = free".
        For each image row, floor should have a relatively stable expected depth.
        Later we judge whether a corridor looks like this expected floor pattern.
        """
        self._ensure_floor_model_fields()

        if "down_depth" not in obs:
            return

        depth = np.asarray(obs["down_depth"], dtype=np.float32)
        h, w = depth.shape[:2]

        x0 = int(w * self.floor_model_x0)
        x1 = int(w * self.floor_model_x1)
        y0 = int(h * self.floor_model_y0)
        y1 = int(h * self.floor_model_y1)

        band = depth[y0:y1, x0:x1]
        model = np.full(h, np.nan, dtype=np.float32)

        for yy in range(y0, y1):
            row = depth[yy, x0:x1]
            valid = row[np.isfinite(row)]
            valid = valid[(valid > 0.05) & (valid < 8.0)]

            if valid.size >= 20:
                model[yy] = float(np.median(valid))

        if self.floor_depth_model is None:
            self.floor_depth_model = model
            self.floor_model_ready = True
            self.floor_model_updates = 1
            return

        # Slow conservative update: only update rows that are close to old floor.
        old = self.floor_depth_model
        new = old.copy()

        for yy in range(h):
            if not np.isfinite(model[yy]):
                continue

            if not np.isfinite(old[yy]):
                new[yy] = model[yy]
                continue

            tol = max(self.floor_abs_tol, self.floor_rel_tol * old[yy])

            if abs(model[yy] - old[yy]) <= tol:
                new[yy] = 0.95 * old[yy] + 0.05 * model[yy]

        self.floor_depth_model = new
        self.floor_model_ready = True
        self.floor_model_updates += 1

    def _floor_corridor_stats(self, obs, x0_ratio, x1_ratio, y0_ratio=0.22, y1_ratio=0.82):
        """
        Return how much this down-looking corridor looks like floor.

        Important:
        - high depth alone is NOT treated as free
        - floor-like means close to row-wise expected floor depth
        - closer-than-floor means obstacle / furniture / wall
        """
        self._ensure_floor_model_fields()
        self._update_floor_depth_model(obs)

        if (not self.use_low_look) or ("down_depth" not in obs) or self.floor_depth_model is None:
            return {
                "ok": True,
                "score": 1.0,
                "floor_ratio": 1.0,
                "obstacle_ratio": 0.0,
                "valid_ratio": 1.0,
                "p50": 9.99,
                "reason": "disabled_or_missing",
            }

        depth = np.asarray(obs["down_depth"], dtype=np.float32)
        h, w = depth.shape[:2]

        x0 = int(w * x0_ratio)
        x1 = int(w * x1_ratio)
        y0 = int(h * y0_ratio)
        y1 = int(h * y1_ratio)

        patch = depth[y0:y1, x0:x1]
        total = max(1, patch.size)

        floor_count = 0
        obstacle_count = 0
        valid_count = 0
        vals = []

        for local_y, yy in enumerate(range(y0, y1)):
            expected = self.floor_depth_model[yy]
            if not np.isfinite(expected) or expected <= 0.05:
                continue

            row = patch[local_y, :]
            valid_mask = np.isfinite(row) & (row > 0.05) & (row < 8.0)
            valid = row[valid_mask]

            if valid.size == 0:
                continue

            valid_count += int(valid.size)
            vals.append(valid)

            tol = max(self.floor_abs_tol, self.floor_rel_tol * expected)

            # Floor-like: close to expected row-wise floor depth.
            floor_like = np.abs(valid - expected) <= tol
            floor_count += int(np.sum(floor_like))

            # Obstacle-like: significantly closer than expected floor.
            obstacle_like = valid < expected * (1.0 - self.obstacle_rel_margin)
            obstacle_count += int(np.sum(obstacle_like))

        valid_ratio = valid_count / total

        if valid_count == 0:
            return {
                "ok": False,
                "score": -1.0,
                "floor_ratio": 0.0,
                "obstacle_ratio": 1.0,
                "valid_ratio": 0.0,
                "p50": 0.0,
                "reason": "no_valid_depth",
            }

        floor_ratio = floor_count / valid_count
        obstacle_ratio = obstacle_count / valid_count

        all_vals = np.concatenate(vals) if vals else np.array([0.0])
        p50 = float(np.percentile(all_vals, 50))

        ok = (
            floor_ratio >= self.floor_min_ratio
            and obstacle_ratio <= self.obstacle_max_ratio
            and valid_ratio >= self.valid_min_ratio
        )

        score = floor_ratio - 0.9 * obstacle_ratio + 0.2 * valid_ratio

        return {
            "ok": bool(ok),
            "score": float(score),
            "floor_ratio": float(floor_ratio),
            "obstacle_ratio": float(obstacle_ratio),
            "valid_ratio": float(valid_ratio),
            "p50": p50,
            "reason": "floor_like" if ok else "not_floor_like",
        }

    def _floor_corridor_scores(self, obs):
        left = self._floor_corridor_stats(obs, 0.08, 0.35)
        center = self._floor_corridor_stats(obs, 0.36, 0.64)
        right = self._floor_corridor_stats(obs, 0.65, 0.92)
        return left, center, right

    def _floor_stats_msg(self, left, center, right):
        return (
            f"floorL={left['floor_ratio']:.2f}/obsL={left['obstacle_ratio']:.2f}/okL={left['ok']} "
            f"floorC={center['floor_ratio']:.2f}/obsC={center['obstacle_ratio']:.2f}/okC={center['ok']} "
            f"floorR={right['floor_ratio']:.2f}/obsR={right['obstacle_ratio']:.2f}/okR={right['ok']}"
        )

    def _best_floor_side(self, left, center, right):
        scores = {
            "left": left["score"],
            "center": center["score"],
            "right": right["score"],
        }
        return max(scores, key=scores.get)

    def _floor_corridor_policy(self, obs, target_side, distance, front):
        """
        Low-look local navigation policy.

        This is the part that finally makes down_depth control the behavior.
        It chooses a corridor, not just a forward brake.
        """
        left, center, right = self._floor_corridor_scores(obs)
        stats_msg = self._floor_stats_msg(left, center, right)

        # If center is floor-like, move forward. This means "the route ahead looks like floor".
        if center["ok"] and front > self.safe_front_dist_m:
            self.sim_env.move_forward()
            self.target_path_forward_steps += 1
            self.visual_forward_steps += 1
            return (
                f"[FLOOR-POLICY] Forward on center floor corridor. "
                f"target_side={target_side}, distance={distance:.2f}, front={front:.2f}, "
                f"visual_forward_steps={self.visual_forward_steps}, {stats_msg}"
            )

        # If target is on one side and that side has floor-like corridor, turn that way.
        if target_side == "left" and left["ok"]:
            self.sim_env.turn_left()
            return (
                f"[FLOOR-POLICY] Center blocked; target left and left floor exists. Turning left. "
                f"distance={distance:.2f}, front={front:.2f}, {stats_msg}"
            )

        if target_side == "right" and right["ok"]:
            self.sim_env.turn_right()
            return (
                f"[FLOOR-POLICY] Center blocked; target right and right floor exists. Turning right. "
                f"distance={distance:.2f}, front={front:.2f}, {stats_msg}"
            )

        # Otherwise choose best floor-like side.
        best = self._best_floor_side(left, center, right)

        if best == "left":
            self.sim_env.turn_left()
        elif best == "right":
            self.sim_env.turn_right()
        else:
            self.sim_env.turn_left()

        return (
            f"[FLOOR-POLICY] Center not safe; searching best floor side={best}. "
            f"target_side={target_side}, distance={distance:.2f}, front={front:.2f}, {stats_msg}"
        )

    def _candidate_keep_in_view(self, obs, detection):
        """
        Target was seen once but not confirmed yet.
        Do NOT continue frontier immediately; keep it in view to get a second confirmation.
        """
        depth = obs["depth"]
        center_x = float(detection["center_x"])
        distance = self.detector.estimate_distance(depth, detection)
        front = self._front_clearance(depth)

        self.last_seen_distance = distance
        self.last_seen_front = front
        self.candidate_lock_steps += 1

        if center_x < self.visual_left_bound:
            self.last_target_side = "left"
            self.sim_env.turn_left()
            return (
                f"[CANDIDATE] Saw possible {self.target}; turning left to confirm. "
                f"center_x={center_x:.2f}, distance={distance:.2f}, candidate_steps={self.candidate_lock_steps}"
            )

        if center_x > self.visual_right_bound:
            self.last_target_side = "right"
            self.sim_env.turn_right()
            return (
                f"[CANDIDATE] Saw possible {self.target}; turning right to confirm. "
                f"center_x={center_x:.2f}, distance={distance:.2f}, candidate_steps={self.candidate_lock_steps}"
            )

        self.last_target_side = "center"
        self.sim_env.stop()
        return (
            f"[CANDIDATE] Holding view to confirm first {self.target}. "
            f"center_x={center_x:.2f}, distance={distance:.2f}, candidate_steps={self.candidate_lock_steps}"
        )

    def _visual_final_approach(self, obs, detection):
        """
        Local policy after target lock.

        New behavior:
        target bbox tells us which side the object is on,
        down_depth floor model tells us where the walkable floor corridor is.
        """
        depth = obs["depth"]
        center_x = float(detection["center_x"])
        distance = self.detector.estimate_distance(depth, detection)
        front = self._front_clearance(depth)

        self.last_seen_distance = distance
        self.last_seen_front = front

        if center_x < self.visual_left_bound:
            target_side = "left"
        elif center_x > self.visual_right_bound:
            target_side = "right"
        else:
            target_side = "center"

        self.last_target_side = target_side

        distance_error = abs(distance - self.target_standoff_m)
        centered = self.visual_left_bound <= center_x <= self.visual_right_bound

        # Normal visual arrival.
        if centered and distance_error <= self.standoff_tolerance_m and self.target_visible_streak >= 2:
            self.sim_env.stop()
            self.state = AgentState.ARRIVED
            return (
                completion_message()
                + f" [VISUAL-ARRIVED] distance={distance:.2f}, error={distance_error:.2f}"
            )

        # If we are already near and low-look says the floor corridor is blocked,
        # treat it as furniture in front / beside us and stop.
        left, center, right = self._floor_corridor_scores(obs)
        stats_msg = self._floor_stats_msg(left, center, right)

        if self.visual_forward_steps >= 4 and distance < 1.75 and (not center["ok"] or front < self.safe_front_dist_m):
            self.sim_env.stop()
            self.state = AgentState.ARRIVED
            return (
                completion_message()
                + f" [FLOOR-SAFE-STOP] distance={distance:.2f}, front={front:.2f}, {stats_msg}"
            )

        # If still far, low-look floor policy decides movement.
        if distance > self.target_standoff_m + self.standoff_tolerance_m:
            return self._floor_corridor_policy(obs, target_side, distance, front)

        # Close enough or blocked.
        if self.target_visible_streak >= 2:
            self.sim_env.stop()
            self.state = AgentState.ARRIVED
            return (
                completion_message()
                + f" [VISUAL-SAFE-STOP] distance={distance:.2f}, front={front:.2f}, {stats_msg}"
            )

        # Fallback: small scan.
        self.sim_env.turn_left()
        return f"[VISUAL] Refining view. distance={distance:.2f}, front={front:.2f}, {stats_msg}"

    def _target_reacquire(self, obs):
        """
        Target is confirmed but temporarily not visible.

        If it was recently close, losing it is acceptable:
        the object may be too low / too near for the detector.
        """
        self.target_lost_steps += 1

        if (
            self.last_seen_distance < 1.75
            and self.visual_forward_steps >= 4
            and self.target_lost_steps >= 2
        ):
            self.sim_env.stop()
            self.state = AgentState.ARRIVED
            return (
                completion_message()
                + f" [LOST-NEAR-SAFE-STOP] last_distance={self.last_seen_distance:.2f}, "
                f"visual_forward_steps={self.visual_forward_steps}"
            )

        if self.target_lost_steps > self.target_lock_lost_limit:
            self.path = []
            self.path_goal = None
            self.steps_since_replan = 999
            self.current_plan_type = "target"
            return (
                f"[REACQUIRE-FAIL] Lost locked {self.target}; switching back to target-memory planning. "
                f"last_distance={self.last_seen_distance:.2f}"
            )

        if self.last_target_side == "right":
            self.sim_env.turn_right()
            return (
                f"[REACQUIRE] Locked {self.target} temporarily lost. "
                f"Searching right. lost_steps={self.target_lost_steps}"
            )

        if self.last_target_side == "left":
            self.sim_env.turn_left()
            return (
                f"[REACQUIRE] Locked {self.target} temporarily lost. "
                f"Searching left. lost_steps={self.target_lost_steps}"
            )

        self.sim_env.turn_left()
        return (
            f"[REACQUIRE] Locked {self.target} temporarily lost. "
            f"Small scan left. lost_steps={self.target_lost_steps}"
        )

    # -----------------------------
    # Path following
    # -----------------------------

    def _angle_to_cell(self, obs, cell):
        x, y, yaw = self._pose(obs)
        wx, wy = self._grid_to_world(*cell)

        dx = wx - x
        dy = wy - y

        # yaw=0 faces +y in our odom convention.
        desired = math.atan2(-dx, dy)
        diff = desired - yaw
        return math.atan2(math.sin(diff), math.cos(diff))

    def _follow_path(self, obs):
        if not self.path:
            self.sim_env.turn_left()
            return "[NO-PATH] No path available. Rotating to scan."

        start = self._current_grid_cell(obs)

        while len(self.path) > 1 and self.path[0] == start:
            self.path.pop(0)

        if len(self.path) <= 1:
            if self.current_plan_type == "target":
                if self.target_path_forward_steps >= 1:
                    self.sim_env.stop()
                    self.state = AgentState.ARRIVED
                    return completion_message()

                self.path = []
                self.steps_since_replan = 999
                self.sim_env.turn_left()
                return "[TARGET] Target path too short. Rotating to refine."

            if self.current_plan_type == "frontier":
                if self.path_goal is not None:
                    self.exhausted_frontiers.append(self.path_goal)

                self.path = []
                self.path_goal = None
                self.steps_since_replan = 999
                self.sim_env.turn_left()
                return "[FRONTIER] Reached useful frontier. Rotating to scan."

            self.path = []
            self.path_goal = None
            self.steps_since_replan = 999
            self.sim_env.turn_left()
            return "[NO-GOAL] No active goal. Rotating to scan."

        lookahead_idx = min(3, len(self.path) - 1)
        next_cell = self.path[lookahead_idx]

        angle_diff = self._angle_to_cell(obs, next_cell)

        if angle_diff > math.radians(12):
            self.sim_env.turn_left()
            return f"[PATH] Turning left toward path. angle={math.degrees(angle_diff):.1f}deg"

        if angle_diff < -math.radians(12):
            self.sim_env.turn_right()
            return f"[PATH] Turning right toward path. angle={math.degrees(angle_diff):.1f}deg"

        left, center, right = self._floor_corridor_scores(obs)
        stats_msg = self._floor_stats_msg(left, center, right)

        if not center["ok"]:
            self.path = []
            self.path_goal = None
            self.steps_since_replan = 999

            best = self._best_floor_side(left, center, right)

            if best == "right":
                self.sim_env.turn_right()
            else:
                self.sim_env.turn_left()

            return (
                f"[PATH-FLOOR-BLOCKED] Center corridor not floor-like; turning {best}. "
                f"plan={self.current_plan_type}, {stats_msg}"
            )

        self.sim_env.move_forward()

        if self.current_plan_type == "target":
            self.target_path_forward_steps += 1

        return (
            f"[PATH] Moving forward. plan={self.current_plan_type}, "
            f"remaining={len(self.path)}, target_forward_steps={self.target_path_forward_steps}, "
            f"{stats_msg}"
        )

    # -----------------------------
    # Main step
    # -----------------------------

    def step(self):
        self.step_count += 1

        obs = self.sim_env.get_observation()
        rgb = obs["rgb"]
        depth = obs["depth"]
        pose = self._pose(obs)

        if self.state == AgentState.ARRIVED:
            return completion_message()

        if self.state == AgentState.FAILED:
            return "[FAILED]"

        self._update_occupancy_from_depth(depth, pose)

        detection = self.detector.detect(rgb, self.target)
        self.last_detection = detection

        if detection is not None:
            self.target_visible_streak += 1
            self.target_lost_steps = 0

            self._update_object_memory(depth, detection, pose)

            target_confirmed = bool(self.object_memory.get(self.target))

            if target_confirmed:
                self.path = []
                self.path_goal = None
                self.current_plan_type = "visual"
                return self._with_stats(
                    self._visual_final_approach(obs, detection),
                    detection,
                    plan_type="visual",
                )

            # Candidate lock: saw it once, keep it in view for confirmation.
            if self.candidate_lock_steps < self.max_candidate_lock_steps:
                self.current_plan_type = "candidate"
                return self._with_stats(
                    self._candidate_keep_in_view(obs, detection),
                    detection,
                    plan_type="candidate",
                )

            # If candidate has not been confirmed after several steps, resume exploration.
            self.candidate_lock_steps = 0

        else:
            self.target_visible_streak = 0

            if self.object_memory.get(self.target):
                self.current_plan_type = "target_reacquire"
                return self._with_stats(
                    self._target_reacquire(obs),
                    detection,
                    plan_type="target_reacquire",
                )

            # If only pending candidate exists and we just lost it, briefly search last side.
            if self.pending_object_memory.get(self.target) and self.candidate_lock_steps > 0:
                self.candidate_lock_steps += 1

                if self.candidate_lock_steps <= self.max_candidate_lock_steps:
                    if self.last_target_side == "right":
                        self.sim_env.turn_right()
                        msg = f"[CANDIDATE-REACQUIRE] Searching right for second confirmation. steps={self.candidate_lock_steps}"
                    elif self.last_target_side == "left":
                        self.sim_env.turn_left()
                        msg = f"[CANDIDATE-REACQUIRE] Searching left for second confirmation. steps={self.candidate_lock_steps}"
                    else:
                        self.sim_env.turn_left()
                        msg = f"[CANDIDATE-REACQUIRE] Small scan for second confirmation. steps={self.candidate_lock_steps}"

                    return self._with_stats(msg, detection, plan_type="candidate_reacquire")

                self.candidate_lock_steps = 0

        self.steps_since_replan += 1

        if self._need_replan():
            plan_type = self._plan(obs)
        else:
            plan_type = self.current_plan_type

        msg = self._follow_path(obs)
        return self._with_stats(msg, detection, plan_type)

    def _with_stats(self, msg, detection, plan_type):
        memory_count = len(self.object_memory.get(self.target, []))
        pending = self.pending_object_memory.get(self.target, {})
        pending_count = pending.get("count", 0)

        known_free = int(np.sum((self.grid == FREE) | (self.grid == VISITED)))
        occupied = int(np.sum(self.grid == OCCUPIED))

        det_msg = "none"
        if detection is not None:
            det_msg = (
                f"{detection['label']} score={detection['score']:.2f} "
                f"bbox={detection['bbox']}"
            )

        return (
            f"{msg} | plan={plan_type}, det={det_msg}, "
            f"memory={memory_count}, pending={pending_count}, "
            f"free={known_free}, occ={occupied}"
        )
