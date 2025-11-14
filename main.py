import cv2
import dlib
import numpy as np
import torch
import intrinsic_model
import time

def extract_from_video_dlib(
    video_path: str,
    predictor_path: str,
    resize_width: int | None = 640,
    frame_stride: int = 1,
    max_frames: int | None = None,
    upsample_num_times: int = 0,
    device: str = "cpu",
):
    """
    Dlib-only face landmark extraction from a video.

    Returns:
        pts_torch: torch.Tensor of shape (F, N, 2)
                   F = number of frames with a detected face (after stride/max_frames)
                   N = number of landmarks from the predictor (e.g. 68)
        idx_torch: torch.LongTensor of shape (F,)
                   frame indices in the original video (0-based).
    """

    # HOG face detector
    detector = dlib.get_frontal_face_detector()
    # Landmark predictor (e.g. 68-point)
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    all_landmarks = []
    frame_indices = []

    frame_idx = 0
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # frame stride
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]

        # optional resize for speed
        if resize_width is not None:
            scale = resize_width / w
            frame_proc = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            scale = 1.0
            frame_proc = frame

        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = detector(gray, upsample_num_times)
        if len(faces) == 0:
            frame_idx += 1
            continue

        # pick the largest face
        areas = [(f.right() - f.left()) * (f.bottom() - f.top()) for f in faces]
        best_idx = int(np.argmax(areas))
        rect = faces[best_idx]

        # run landmark predictor on resized frame
        shape = predictor(gray, rect)

        # number of landmarks N (e.g. 68)
        num_points = shape.num_parts
        pts = np.empty((num_points, 2), dtype=np.float32)
        for i in range(num_points):
            pts[i, 0] = shape.part(i).x  # x
            pts[i, 1] = shape.part(i).y  # y
        H, W = frame_proc.shape[:2]

        pts[:, 0] = (pts[:, 0] - W * 0.5) / W   # x_norm
        pts[:, 1] = (pts[:, 1] - H * 0.5) / H   # y_norm
        # store landmarks (in resized frame coordinates)
        all_landmarks.append(pts)
        frame_indices.append(frame_idx)

        count += 1
        if max_frames is not None and count >= max_frames:
            break

        frame_idx += 1

    cap.release()

    # stack to [F, N, 2]
    if len(all_landmarks) > 0:
        pts_np = np.stack(all_landmarks, axis=0)  # (F, N, 2)
    else:
        # no faces found at all
        pts_np = np.zeros((0, 0, 2), dtype=np.float32)
    F, N, _ = pts_np.shape
    pts_np = pts_np.reshape(1, F*N, 2)
    pts_torch = torch.from_numpy(pts_np).to(device)
    idx_torch = torch.tensor(frame_indices, dtype=torch.long, device=device)
    return pts_torch, idx_torch
def predict_intrinsic(video_path: str, model_path: str, device: str = "cpu"):
    pts, idx = extract_from_video_dlib(
        video_path=video_path,
        predictor_path="Models/shape_predictor_68_face_landmarks.dat",
        resize_width=2160,
        frame_stride=2,
        max_frames=None,
        upsample_num_times=0,
        device="cpu",
    )
    device = torch.device(device)
    model = intrinsic_model.PointNet()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(intrinsic_model.predict(model, pts, device))
if __name__ == "__main__":
    start_time = time.perf_counter()
    result = predict_intrinsic(
                video_path="Data/Sample/XR001_08100015_C015.mov",
                model_path="D:/Calibration/Models/testModel.pth"
            )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    

            
