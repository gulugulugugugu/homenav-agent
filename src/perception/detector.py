class DebugTargetDetector:
    """
    Debug detector for validating the navigation pipeline.

    This does not use simulator privileged object state.
    It only returns a fake visual detection after several search steps,
    so that the agent state machine and Habitat action loop can be tested.
    """

    def __init__(self):
        self.steps = 0

    def detect(self, rgb, target_name: str):
        self.steps += 1

        # Pretend the target becomes visible after several turns.
        if self.steps < 4:
            return None

        # center_x < 0.5 means target is left, > 0.5 means right.
        # Here we make it nearly centered.
        return {
            "label": target_name,
            "center_x": 0.52,
            "center_y": 0.50,
            "bbox": [260, 180, 380, 320],
            "score": 0.90,
        }

    def estimate_distance(self, depth, detection):
        x1, y1, x2, y2 = detection["bbox"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        try:
            d = float(depth[cy, cx])
            if d <= 0:
                return 2.0
            return d
        except Exception:
            return 2.0
