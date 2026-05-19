import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from trend_analysis import archive_vehicle
# =======================
# Vehicle Tracking
# =======================

MAX_LOST_FRAMES = 10    # frames before a tracker is dropped
IOU_THRESHOLD   = 0.25  # minimum IoU to accept a match
DT              = 1.0   # time step (1 frame)

# For easier tracking handling
class Vehicle:
    _id_counter = 0

    def __init__(self, bbox):
        Vehicle._id_counter += 1
        self.id          = Vehicle._id_counter
        self.bbox        = bbox          # last known (x, y, w, h)
        self.lost_frames = 0
        self.trajectory  = []
        self.color       = tuple(int(c) for c in np.random.randint(50, 255, 3))

        cx, cy = self._centre(bbox)
        self.kalman = create_kalman(cx, cy)
        self.trajectory.append((cx, cy))

    @staticmethod
    def _centre(bbox):
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

    def predict(self):
        """
        Kalman prediction step.
        Returns the predicted (cx, cy) for this frame.
        """
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])

    def update(self, bbox):
        """
        Kalman correction step using the matched detection.
        """
        cx, cy = self._centre(bbox)
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self.kalman.correct(measurement)

        self.bbox = bbox
        self.lost_frames = 0
        self.trajectory.append((cx, cy))

    def predicted_bbox(self):
        """
        Build a synthetic bbox around the Kalman-predicted centre,
        keeping the last known width and height.
        """
        cx, cy   = self.predict()
        _, _, w, h = self.bbox
        return cx - w // 2, cy - h // 2, w, h

# Kalman fileter generation
def create_kalman(cx, cy):
    """
    Create and initialise a Kalman filter for a single vehicle.

    State vector  x = [u, v, u̇, v̇]ᵀ
    Measurement   z = [u, v]ᵀ   (centre of bounding box)
    """
    kf = cv2.KalmanFilter(4, 2)   # 4 state dims, 2 measurement dims

    # State transition matrix F  (constant velocity model)
    # u_t  = u_{t-1} + dt * u̇_{t-1}
    # v_t  = v_{t-1} + dt * v̇_{t-1}
    kf.transitionMatrix = np.array([
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ], dtype=np.float32)

    # Observation matrix H  (we only measure position, not velocity)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Process noise covariance Q
    # Higher values = allow more sudden changes in motion
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    # Measurement noise covariance R
    # Higher values = trust the detector less
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    # Initial state covariance P
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

    # Seed the filter with the first detected position
    kf.statePost = np.array(
        [[cx], [cy], [0.], [0.]], dtype=np.float32
    )

    return kf

# Calculate the IoU matches
def compute_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

# Build the cost matrix
def build_cost_matrix(vehicles, detections):
    """
    Cost matrix C where C[i][j] = 1 - IoU(predicted_bbox_i, detection_j).
    Shape: (num_vehicles, num_detections)
    """
    n, m = len(vehicles), len(detections)
    C = np.ones((n, m), dtype=np.float32)
    for i, vehicle in enumerate(vehicles):
        pred_bbox = vehicle.predicted_bbox()
        for j, det in enumerate(detections):
            C[i, j] = 1.0 - compute_iou(pred_bbox, det)
    return C

# Match based on hungarian algorithm
def hungarian_match(vehicles, detections):
    """
    Run the Hungarian algorithm on the IoU cost matrix.
    Returns:
      matched   : list of (vehicle_idx, detection_idx) pairs
      unmatched_vehicles   : list of vehicle indices with no match
      unmatched_detections : list of detection indices with no match
    """
    if not vehicles or not detections:
        return [], list(range(len(vehicles))), list(range(len(detections)))

    C = build_cost_matrix(vehicles, detections)
    row_ind, col_ind = linear_sum_assignment(C)

    matched, unmatched_v, unmatched_d = [], [], []
    matched_det = set()

    for r, c in zip(row_ind, col_ind):
        if C[r, c] < (1.0 - IOU_THRESHOLD):   # IoU > threshold → valid match
            matched.append((r, c))
            matched_det.add(c)
        else:
            unmatched_v.append(r)

    unmatched_d = [j for j in range(len(detections)) if j not in matched_det]

    # Vehicles not touched by linear_sum_assignment at all
    matched_v = {r for r, _ in matched} | set(unmatched_v)
    unmatched_v += [i for i in range(len(vehicles)) if i not in matched_v]

    return matched, unmatched_v, unmatched_d

def update_trackers(frame, detections, vehicles, archive):
    """
    Call once per frame after detect_vehicles().

    Args:
      frame      : raw BGR frame (for visualisation only)
      detections : list of (x, y, w, h) from detect_vehicles()

    Returns:
      vehicles   : updated list of active Vehicle objects
      debug_frame: annotated BGR frame
    """
    debug_frame = frame.copy()

    # ── Step 1: Kalman prediction for all active trackers ──────────
    for vehicle in vehicles:
        vehicle.predict()

    # ── Step 2: Hungarian algorithm — match trackers to detections ─
    matched, unmatched_v, unmatched_d = hungarian_match(vehicles, detections)

    # ── Step 3: Update matched trackers with Kalman correction ─────
    for v_idx, d_idx in matched:
        vehicles[v_idx].update(detections[d_idx])

    # ── Step 4: Mark unmatched trackers as lost ────────────────────
    for v_idx in unmatched_v:
        vehicles[v_idx].lost_frames += 1

    # ── Step 5: Spawn new trackers for unmatched detections ────────
    for d_idx in unmatched_d:
        vehicles.append(Vehicle(detections[d_idx]))

    # ── Step 6: Remove stale trackers ─────────────────────────────
    still_active = []
    for v in vehicles:
        if v.lost_frames > MAX_LOST_FRAMES:
            archive = archive_vehicle(v, archive)       # save before dropping
        else:
            still_active.append(v)
    vehicles = still_active

    # ── Step 7: Visualisation ─────────────────────────────────────
    for vehicle in vehicles:
        x, y, w, h = vehicle.bbox
        color = vehicle.color

        # Bounding box
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)

        # ID label
        cv2.putText(
            debug_frame, f"ID {vehicle.id}",
            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )

        # Trajectory polyline
        if len(vehicle.trajectory) > 1:
            pts = np.array(vehicle.trajectory, dtype=np.int32)
            cv2.polylines(debug_frame, [pts], isClosed=False,
                          color=color, thickness=2)

        # Kalman predicted centre
        pcx, pcy = vehicle.predict()
        cv2.circle(debug_frame, (pcx, pcy), 4, (0, 0, 255), -1)

    return vehicles, archive, debug_frame