import cv2
import time
from typing import Callable, List, Tuple, Optional
import mediapipe as mp

# Types for the callback
Landmarks = List[Tuple[float, float]]  # list of (x_norm, y_norm) in [0,1]
FrameResult = dict  # {"landmarks": [Landmarks], "boxes": [Tuple[int,int,int,int]], "fps": float}

def run_face_tracking(
    camera_index: int = 0,
    max_num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    draw: bool = True,
    window_name: str = "MediaPipe Face Tracking",
    callback: Optional[Callable[[FrameResult], None]] = None,
) -> None:
    """
    Open a webcam stream and track faces in real-time using MediaPipe Face Mesh.

    Press 'q' to quit the window.

    Args:
        camera_index: OpenCV camera index (0 = default webcam).
        max_num_faces: Maximum faces to track.
        min_detection_confidence: Face detection confidence threshold.
        min_tracking_confidence: Landmark tracking confidence threshold.
        draw: If True, draw landmarks/overlays on the frame.
        window_name: Title of the display window.
        callback: Optional function called every frame with a dict:
            {
              "landmarks": [ [(x_norm,y_norm), ...] for each face ],
              "boxes": [ (x, y, w, h) for each face in pixel coords ],
              "fps": float
            }
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    # Improve latency a bit
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # FaceMesh does detection + tracking internally
    with mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=True,  # enables iris landmarks
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        prev_t = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            h, w = frame.shape[:2]
            all_landmarks: List[Landmarks] = []
            boxes: List[Tuple[int, int, int, int]] = []

            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    # Collect normalized landmarks
                    lm = [(p.x, p.y) for p in face_lms.landmark]
                    all_landmarks.append(lm)

                    # Compute a rough bounding box from landmarks
                    xs = [int(p.x * w) for p in face_lms.landmark]
                    ys = [int(p.y * h) for p in face_lms.landmark]
                    x0, x1 = max(min(xs), 0), min(max(xs), w - 1)
                    y0, y1 = max(min(ys), 0), min(max(ys), h - 1)
                    boxes.append((x0, y0, x1 - x0, y1 - y0))

                    if draw:
                        # Draw landmarks/edges
                        mp_drawing.draw_landmarks(
                            frame,
                            face_lms,
                            mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            frame,
                            face_lms,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                        )
                        mp_drawing.draw_landmarks(
                            frame,
                            face_lms,
                            mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
                        )
                        # Draw bbox
                        for (bx, by, bw, bh) in boxes[-1:]:
                            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            if draw:
                cv2.putText(frame, f"Faces: {len(all_landmarks)}  FPS: {fps:.1f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)

            if callback is not None:
                callback({"landmarks": all_landmarks, "boxes": boxes, "fps": fps})

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Demo usage: run this file directly to test
if __name__ == "__main__":
    def print_simple(res: FrameResult):
        # Example: print FPS and first face bbox
        if res["boxes"]:
            (x, y, w, h) = res["boxes"][0]
            print(f"FPS {res['fps']:.1f} | First box: ({x},{y},{w},{h})", end="\r")

    run_face_tracking(max_num_faces=2, callback=None)  # set callback=print_simple to log
