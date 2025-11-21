import cv2
import dlib
import numpy as np
import torch
import model
import time
import matplotlib.pyplot as plt
def extract_from_video_dlib(
    video_path: str,
    predictor_path: str,
    frame_stride: int = 1,
    max_frames: int | None = None,
    upsample_num_times: int = 0,
    device: str = "cpu",
):

    detector = dlib.get_frontal_face_detector()
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

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, upsample_num_times)
        if len(faces) == 0:
            frame_idx += 1
            continue

        areas = [(f.right() - f.left()) * (f.bottom() - f.top()) for f in faces]
        best_idx = int(np.argmax(areas))
        rect = faces[best_idx]
    
        shape = predictor(gray, rect)

        num_points = shape.num_parts
        pts = np.empty((num_points, 2), dtype=np.float32)
        for i in range(num_points):
            pts[i, 0] = shape.part(i).x  # x
            pts[i, 1] = shape.part(i).y  # y
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
    pts_torch = torch.from_numpy(pts_np).to(device)
    idx_torch = torch.tensor(frame_indices, dtype=torch.long, device=device)
    return pts_torch, idx_torch, W, H
def resample_to_fixed_frames(pts, target_f=60):
    F = pts.shape[0]
    if F == target_f:
        return pts
    
    old_indices = np.linspace(0, F - 1, F)
    new_indices = np.linspace(0, F - 1, target_f)
    
    if pts.ndim == 1:
        pts_resampled = np.interp(new_indices, old_indices, pts)
    else:
        pts_resampled = np.zeros((target_f,) + pts.shape[1:])
        pts_flat = pts.reshape(F, -1)
        for i in range(pts_flat.shape[1]):
            pts_resampled.reshape(target_f, -1)[:, i] = np.interp(
                new_indices, old_indices, pts_flat[:, i]
            )
    return pts_resampled

def predict_intrinsic(video_path: str, model_path: str, device: str = "cpu"):
    pts, idx, W, H = extract_from_video_dlib(
        video_path=video_path,
        predictor_path="Models/shape_predictor_68_face_landmarks.dat",
        frame_stride=5,
        max_frames=10,
        upsample_num_times=0,
        device=device,
    )
    F, N, _ = pts.shape
    device = torch.device(device)
    pts = pts.to(device, dtype=torch.float32)
    for pt in pts:
        plt.clf()  # clear previous frame
        plt.scatter(pt[:, 0], pt[:, 1], s=30)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(0, W)
        # plt.ylim(0, H)
        plt.xlabel("u")
        plt.ylabel("v")
        plt.tight_layout()
        plt.pause(1.0/10)
    pts = pts.reshape(1, F * N, 2)
    
    W_t = torch.tensor(float(W), dtype=pts.dtype, device=device)
    H_t = torch.tensor(float(H), dtype=pts.dtype, device=device)
    pts[..., 0] = (pts[..., 0] - W_t * 0.5) / W_t
    pts[..., 1] = (pts[..., 1] - H_t * 0.5) / H_t
    model = model.PointNet()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model.predict(model, pts, device)[0]
    fx = (pred[0] * W_t).item()
    fy = (pred[1] * H_t).item()
    cx = (pred[2] * W_t).item()
    cy = (pred[3] * H_t).item()
    return [fx, fy, cx, cy, float(W), float(H)]
if __name__ == "__main__":
    start_time = time.perf_counter()
    result = predict_intrinsic(
                video_path="Data/Sample/XR001_08100015_C015.mov",
                model_path="testModel.pth"
            )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    print(result)   

            
