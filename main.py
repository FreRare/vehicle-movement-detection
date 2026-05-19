import cv2
from detection import preprocess_frame, detect_vehicles
from tracking import update_trackers
from trend_analysis import run_trend_analysis, draw_trend_summary, collect_all_vehicles, set_frame_dimensions
import time

VIDEO_PATH = "test_data/test_vid_short.mp4"

ANALYSIS_INTERVAL = 200 # Trend analysis on 200 frame interval

SHOW_DEBUG_FRAMES = True # Show debug frames if True
SLOW_DEBUG_VIDEO = False # Slows video frame processing to 10ms/frame

def main():
    # Main loop
    frame_idx = 0
    # Tracked vehicles
    vehicles = []
    # Stores completed trajectories of vehicles that have left the frame.
    # Survives the entire session — never cleared.
    trajectory_archive = []   # list of (vehicle_id, trajectory)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")
    
    # set up frame size
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    set_frame_dimensions(w, h)

    # Last frame needed for finalized trend display
    last_frame = None

    while cap.isOpened():
        if SLOW_DEBUG_VIDEO:
            time.sleep(0.01)
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()

        preprocessed          = preprocess_frame(frame)
        detections, debug_img = detect_vehicles(frame, preprocessed)
        vehicles, trajectory_archive, debug_img = update_trackers(frame, detections, vehicles, trajectory_archive)

        if frame_idx % ANALYSIS_INTERVAL == 0 and frame_idx > 0:
            all_vehicles = collect_all_vehicles(vehicles, trajectory_archive)
            labels, cluster_info, valid_v, overlay = run_trend_analysis(all_vehicles, frame)
            if overlay is not None:
                cv2.imshow("Trend analysis", overlay)
        
        if SHOW_DEBUG_FRAMES:
            cv2.imshow("Tracking", debug_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Final analysis
    labels, cluster_info, valid_v, overlay = run_trend_analysis(all_vehicles, last_frame)
    if overlay is not None:
        cv2.imshow("Trend analysis", overlay)
    if cluster_info:
        summary = draw_trend_summary(last_frame, cluster_info, valid_v, labels)
        cv2.imshow("Trend Summary", summary)
        cv2.imwrite("trend_summary.png", summary)   # save to disk
        cv2.waitKey(0)


if __name__ == "__main__":
    main()