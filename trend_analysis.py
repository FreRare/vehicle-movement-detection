import numpy as np
from sklearn.cluster import KMeans
import cv2


# =======================
# Trend analysis
# =======================

N_SAMPLE_POINTS  = 20     # each trajectory resampled to this many points
MIN_TRAJ_LENGTH  = 10     # discard trajectories shorter than this
K_MAX            = 10     # maximum number of clusters to evaluate
K_MIN            = 3      # minimum number of clusters to evaluate
WCSS_SMOOTH      = True   # smooth the WCSS curve before elbow detection
FRAME_WIDTH  = None   # set once at startup
FRAME_HEIGHT = None
POSITION_WEIGHT     = 4.0   # weight of position components
DISPLACEMENT_WEIGHT = 2.0   # higher = direction matters more than position

# Needed so trend analysis work with all trajectories and not just seen ones
class ArchivedVehicle:
    """
    Lightweight stand-in for a full Vehicle object.
    Holds only what trend analysis needs.
    """
    def __init__(self, record):
        self.id         = record["id"]
        self.trajectory = record["trajectory"]
        self.color      = record["color"]

def collect_all_vehicles(active_vehicles, archive):
    """
    Merge active trackers with the trajectory archive into one list.
    Deduplicates by vehicle ID so no trajectory is counted twice.
    """
    seen_ids = {v.id for v in active_vehicles}
    archived = [
        ArchivedVehicle(r) for r in archive
        if r["id"] not in seen_ids
    ]
    return active_vehicles + archived

# Step1 normalization
def resample_trajectory(trajectory, n=N_SAMPLE_POINTS):
    """
    Linearly interpolate a variable-length trajectory to exactly
    n equally spaced points.

    Args:
      trajectory : list of (cx, cy) tuples
      n          : target number of points

    Returns:
      numpy array of shape (n, 2)
    """
    traj  = np.array(trajectory, dtype=np.float32)
    old_t = np.linspace(0, 1, len(traj))
    new_t = np.linspace(0, 1, n)

    resampled_x = np.interp(new_t, old_t, traj[:, 0])
    resampled_y = np.interp(new_t, old_t, traj[:, 1])

    return np.stack([resampled_x, resampled_y], axis=1)

def set_frame_dimensions(w, h):
    global FRAME_WIDTH, FRAME_HEIGHT
    FRAME_WIDTH  = w
    FRAME_HEIGHT = h


def trajectory_to_feature_vector(trajectory):
    """
    Feature vector with positions normalised to frame dimensions,
    preserving absolute spatial location (lane position).
    """
    if FRAME_WIDTH is None or FRAME_HEIGHT is None:
        raise RuntimeError(
            "Call set_frame_dimensions(w, h) before trend analysis."
        )

    resampled = resample_trajectory(trajectory)   # (N, 2) in pixels

    # ── Position: normalise by frame size, NOT trajectory bounds ───
    positions = resampled.copy().astype(np.float32)
    positions[:, 0] /= FRAME_WIDTH    # x in [0, 1] relative to frame
    positions[:, 1] /= FRAME_HEIGHT   # y in [0, 1] relative to frame

    # ── Displacement: encodes direction of travel ──────────────────
    displacements = np.diff(resampled, axis=0).astype(np.float32)
    max_disp = np.linalg.norm(displacements, axis=1).max()
    if max_disp > 0:
        displacements /= max_disp

    pos_flat  = positions.flatten()     * POSITION_WEIGHT
    disp_flat = displacements.flatten() * DISPLACEMENT_WEIGHT

    return np.concatenate([pos_flat, disp_flat])


def build_feature_matrix(vehicles):
    """
    Build the feature matrix from all vehicles with
    sufficiently long trajectories.

    Returns:
      feature_matrix : np.array of shape (M, 2*N)
      valid_vehicles : list of Vehicle objects used
    """
    valid_vehicles = [
        v for v in vehicles if len(v.trajectory) >= MIN_TRAJ_LENGTH
    ]

    if not valid_vehicles:
        return None, []

    feature_matrix = np.array([
        trajectory_to_feature_vector(v.trajectory)
        for v in valid_vehicles
    ], dtype=np.float32)

    return feature_matrix, valid_vehicles

# Step2 Optimal K using elbow method
def compute_wcss(feature_matrix, k_range):
    """
    Run K-means for each K in k_range and record the
    Within-Cluster Sum of Squares (WCSS).
    """
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(feature_matrix)
        wcss.append(km.inertia_)
    return np.array(wcss)


def find_elbow(wcss_values):
    """
    Detect the elbow point using the second derivative of the
    WCSS curve (maximum curvature).
    Returns the optimal index into the k_range list.
    """
    if len(wcss_values) < 3:
        # Not enough points for a second derivative — return the middle
        print("[TrendAnalysis] Too few WCSS values for elbow detection, using first K")
        return 0

    if WCSS_SMOOTH:
        kernel      = np.array([1/3, 1/3, 1/3])
        wcss_values = np.convolve(wcss_values, kernel, mode='valid')

    if len(wcss_values) < 3:
        print("[TrendAnalysis] WCSS too short after smoothing, skipping elbow")
        return 0

    second_diff = np.diff(wcss_values, n=2)

    if len(second_diff) == 0:
        print("[TrendAnalysis] second_diff is empty, defaulting to first K")
        return 0

    elbow_idx = np.argmax(second_diff) + 1
    return elbow_idx


def select_optimal_k(feature_matrix):
    n_samples = len(feature_matrix)

    # Need at least K_MIN + 1 trajectories to compare cluster counts
    if n_samples <= K_MIN:
        print(f"[TrendAnalysis] Only {n_samples} trajectories — defaulting to K=1")
        return 1, np.array([]), []

    max_k   = min(K_MAX, n_samples - 1)
    k_range = range(K_MIN, max_k + 1)

    # Sanity check: range must be non-empty
    if len(list(k_range)) == 0:
        print(f"[TrendAnalysis] k_range is empty (max_k={max_k}) — defaulting to K={K_MIN}")
        return K_MIN, np.array([]), []

    wcss      = compute_wcss(feature_matrix, k_range)
    elbow     = find_elbow(wcss)
    optimal_k = list(k_range)[elbow]

    print(f"[TrendAnalysis] WCSS values: {np.round(wcss, 1)}")
    print(f"[TrendAnalysis] Optimal K = {optimal_k}")

    return optimal_k, wcss, list(k_range)

# Step3 Cluster and compute metadata
def compute_average_speed(trajectory):
    """
    Mean Euclidean distance between consecutive trajectory points,
    in pixels per frame.
    """
    traj  = np.array(trajectory, dtype=np.float32)
    diffs = np.diff(traj, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    return float(np.mean(speeds))


def compute_heading(trajectory):
    """
    Overall heading angle of the trajectory in degrees,
    measured from the first to the last point.
    """
    traj  = np.array(trajectory, dtype=np.float32)
    delta = traj[-1] - traj[0]
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def cluster_trajectories(feature_matrix, valid_vehicles, k):
    """
    Run K-means with the chosen K and compute per-cluster metadata.

    Returns:
      labels        : cluster label per vehicle
      cluster_info  : list of dicts, one per cluster, containing:
                        centroid, covariance, avg_speed, heading,
                        member_count, member_ids
    """
    if k == 1:
        labels = np.zeros(len(feature_matrix), dtype=int)
        cluster_info = []
    else:
        km     = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(feature_matrix)

        cluster_info = []

        for j in range(k):
            members_idx = np.where(labels == j)[0]
            members_fv  = feature_matrix[members_idx]
            members_v   = [valid_vehicles[i] for i in members_idx]

            centroid    = km.cluster_centers_[j]

            # Covariance matrix of feature vectors in this cluster
            # Used later by the anomaly detection module
            if len(members_fv) > 1:
                covariance = np.cov(members_fv, rowvar=False).astype(np.float64)
            else:
                covariance = np.eye(members_fv.shape[1], dtype=np.float64)

            avg_speeds = [compute_average_speed(v.trajectory) for v in members_v]
            headings   = [compute_heading(v.trajectory) for v in members_v]

            cluster_info.append({
                "id"           : j,
                "centroid"     : centroid,
                "covariance"   : covariance,
                "mean_speed"   : float(np.mean(avg_speeds)),
                "std_speed"    : float(np.std(avg_speeds)),
                "mean_heading" : float(np.mean(headings)),
                "member_count" : len(members_idx),
                "member_ids"   : [v.id for v in members_v],
            })

            print(
                f"  Cluster {j}: {len(members_idx)} vehicles | "
                f"heading {cluster_info[-1]['mean_heading']:.1f}° | "
                f"speed {cluster_info[-1]['mean_speed']:.1f} px/frame"
            )
    return labels, cluster_info

# Step4 visualization
CLUSTER_COLOURS = [
    (255, 60,  60),   (60,  255, 60),  (60,  60,  255),
    (255, 255, 60),   (255, 60,  255), (60,  255, 255),
    (255, 165, 0),    (180, 0,   255), (0,   200, 100),
    (200, 100, 0),
]

def draw_cluster_trajectories(frame, valid_vehicles, labels):
    """
    Redraw each vehicle's trajectory coloured by its cluster label.
    """
    overlay = frame.copy()

    for vehicle, label in zip(valid_vehicles, labels):
        color = CLUSTER_COLOURS[label % len(CLUSTER_COLOURS)]
        traj  = np.array(vehicle.trajectory, dtype=np.int32)

        if len(traj) > 1:
            cv2.polylines(overlay, [traj], isClosed=False, color=color, thickness=2)

        # Mark end point
        cv2.circle(overlay, tuple(traj[-1]), 5, color, -1)

        # Cluster label near end point
        cv2.putText(
            overlay, f"C{label}",
            tuple(traj[-1] + np.array([6, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )

    return overlay

def run_trend_analysis(vehicles, frame=None):
    """
    Full trend analysis pipeline.

    Args:
      vehicles : list of Vehicle objects from the tracking module
      frame    : optional BGR frame for visualisation

    Returns:
      labels        : np.array — cluster label per valid vehicle
      cluster_info  : list of cluster metadata dicts
      valid_vehicles: list of Vehicle objects that were clustered
      overlay       : annotated frame (or None if no frame provided)
    """
    print("[TrendAnalysis] Building feature matrix...")
    feature_matrix, valid_vehicles = build_feature_matrix(vehicles)

    if feature_matrix is None:
        print("[TrendAnalysis] Not enough trajectories yet.")
        return None, [], [], None

    print(f"[TrendAnalysis] {len(valid_vehicles)} trajectories available.")

    optimal_k, _, _ = select_optimal_k(feature_matrix)
    labels, cluster_info = cluster_trajectories(
        feature_matrix, valid_vehicles, optimal_k
    )

    overlay = None
    if frame is not None:
        overlay = draw_cluster_trajectories(frame, valid_vehicles, labels)

    return labels, cluster_info, valid_vehicles, overlay

def compute_pixel_centroid(vehicles_in_cluster):
    """
    Compute the mean trajectory in pixel space by resampling all
    member trajectories to N_SAMPLE_POINTS and averaging them.
    This avoids the per-trajectory normalisation problem entirely.
    """
    resampled = np.array([
        resample_trajectory(v.trajectory)
        for v in vehicles_in_cluster
    ])                          # shape: (n_members, N_SAMPLE_POINTS, 2)
    return np.mean(resampled, axis=0).astype(np.int32)   # (N_SAMPLE_POINTS, 2)


def draw_trend_summary(frame, cluster_info, valid_vehicles, labels):
    summary = frame.copy()
    overlay = frame.copy()

    for cluster in cluster_info:
        cid   = cluster["id"]
        color = CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]

        # Collect members of this cluster
        members = [
            v for v, lbl in zip(valid_vehicles, labels) if lbl == cid
        ]

        if not members:
            continue

        # ── Member trajectories (drawn on overlay, blended later) ──
        for vehicle in members:
            traj = np.array(vehicle.trajectory, dtype=np.int32)
            if len(traj) > 1:
                cv2.polylines(
                    overlay, [traj], isClosed=False,
                    color=color, thickness=1
                )

        # ── Pixel-space centroid path ──────────────────────────────
        centroid_px = compute_pixel_centroid(members)   # (N, 2) in pixels

        cv2.polylines(
            summary, [centroid_px], isClosed=False,
            color=color, thickness=4
        )

        # Directional arrow at the end
        if len(centroid_px) >= 2:
            cv2.arrowedLine(
                summary,
                tuple(centroid_px[-2]),
                tuple(centroid_px[-1]),
                color=color, thickness=4, tipLength=0.4
            )

        # ── Label box at the midpoint of the centroid path ─────────
        mid      = centroid_px[len(centroid_px) // 2]
        label_text = (
            f"C{cid} | n={cluster['member_count']} | "
            f"{cluster['mean_speed']:.1f}px/f | "
            f"{cluster['mean_heading']:.0f}deg"
        )
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            summary,
            (mid[0], mid[1] - th - 6),
            (mid[0] + tw + 4, mid[1] + 2),
            (0, 0, 0), -1
        )
        cv2.putText(
            summary, label_text, tuple(mid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    # ── Blend member trajectories semi-transparently ───────────────
    cv2.addWeighted(overlay, 0.3, summary, 0.7, 0, summary)

    # ── Legend ─────────────────────────────────────────────────────
    legend_h = 30 + 22 * len(cluster_info)
    cv2.rectangle(summary, (8, 8), (320, legend_h), (0, 0, 0), -1)
    cv2.putText(
        summary, f"Detected trends: {len(cluster_info)}",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    for cluster in cluster_info:
        cid   = cluster["id"]
        color = CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]
        y     = 28 + 22 * (cid + 1)
        cv2.circle(summary, (20, y - 4), 6, color, -1)
        cv2.putText(
            summary,
            f"Cluster {cid}: {cluster['member_count']} vehicles, "
            f"heading {cluster['mean_heading']:.0f}deg",
            (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

    return summary

def archive_vehicle(vehicle, archive):
    """
    Save a vehicle's trajectory to the archive before dropping it.
    Only worth keeping if the trajectory is long enough to be useful.
    """
    if len(vehicle.trajectory) >= MIN_TRAJ_LENGTH:
        archive.append({
            "id"         : vehicle.id,
            "trajectory" : vehicle.trajectory.copy(),
            "color"      : vehicle.color,
        })
        print(f"[Archive] Vehicle {vehicle.id} archived "
              f"({len(vehicle.trajectory)} points)")
    return archive