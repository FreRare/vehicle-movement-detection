import cv2

# =======================
# Preprocessing 
# =======================

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA  = 1.5

def preprocess_frame(frame):
    """
    Convert a raw BGR frame to a noise-reduced grayscale image.
    Steps:
      1. BGR → Grayscale
      2. Gaussian blur for noise reduction
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    return blurred

# =======================
# Vehicle Detection 
# =======================

MIN_CONTOUR_AREA = 1000   # px² — filters out small noise blobs
MOG2_HISTORY     = 500    # frames used to build the background model
MOG2_THRESHOLD   = 20     # sensitivity: lower = more detections

mog2 = cv2.createBackgroundSubtractorMOG2(
    history      = MOG2_HISTORY,
    varThreshold = MOG2_THRESHOLD,
    detectShadows= True           # shadows labelled as 127, foreground as 255
)

# Morphological kernels
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_vehicles(frame, preprocessed):
    """
    Detect vehicles in a frame using MOG2 background subtraction.
    Returns:
      detections : list of (x, y, w, h) bounding boxes
      debug_frame : annotated BGR frame for visualisation
    """
    # ── Background subtraction ────────────────────────────────────
    fg_mask = mog2.apply(preprocessed)

    # Remove shadow pixels (value 127), keep only foreground (255)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # ── Morphological cleanup ─────────────────────────────────────
    # CLOSE: fills holes inside detected blobs
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    # OPEN:  removes small isolated noise blobs
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)

    # ── Contour detection ─────────────────────────────────────────
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections  = []
    debug_frame = frame.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue                        # skip noise

        x, y, w, h = cv2.boundingRect(contour)

        # Basic aspect-ratio filter: vehicles are wider than tall typically
        aspect_ratio = w / float(h)
        if not (0.3 < aspect_ratio < 5.0):
            continue

        detections.append((x, y, w, h))

        # Draw bounding box + area label on debug frame
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_frame, f"{int(area)}px",
            (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    return detections, debug_frame