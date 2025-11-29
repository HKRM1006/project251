import cv2
import dlib
import numpy as np
import torch
from PointNet2D import PointNet
from model import Model
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
            pts[i, 1] = H - shape.part(i).y
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
    pts_np = resample_to_fixed_frames(pts_np, 100)
    pts_torch = torch.from_numpy(pts_np).to(device)
    idx_torch = torch.tensor(frame_indices, dtype=torch.long, device=device)
    return pts_torch, idx_torch, W, H

def resample_to_fixed_frames(pts, target_f=60):
    pts = np.asarray(pts)
    F = pts.shape[0]

    if F == target_f:
        return pts.copy()

    old_idx = np.arange(F)
    new_idx = np.linspace(0, F - 1, target_f)

    interp_func = interp1d(old_idx, pts, axis=0, kind='linear', fill_value="extrapolate")
    pts_resampled = interp_func(new_idx)

    return pts_resampled

def predict_intrinsic(video_path: str, model_name: str, token: str , device: str = "cpu"):
    pts, idx, W, H = extract_from_video_dlib(
        video_path=video_path,
        predictor_path="Models/shape_predictor_68_face_landmarks.dat",
        frame_stride=4,
        max_frames=100,
        upsample_num_times=0,
        device=device,
    )
    F, N, _ = pts.shape
    device = torch.device(device)
    pts = pts.to(device, dtype=torch.float32)
    
    pt = pts.reshape(F*N,2)
    plt.clf()  # clear previous frame
    plt.scatter(pt[:, 0], pt[:, 1], s=30)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.xlabel("u")
    plt.ylabel("v")
    plt.tight_layout()
    plt.show()

    pts = pts.reshape(F,2,N)
    W_t = torch.tensor(float(W), dtype=pts.dtype, device=device)
    H_t = torch.tensor(float(H), dtype=pts.dtype, device=device)
    model = Model(torch.tensor([1600/2,896/2,1]))
    model.load(model_name, token)
    model.set_eval()
    print(model.predict_intrinsic(pts).mean(0))
    _, pred, _, _ = model.alternating_optimize(pts)
    pred = pred.mean(0)
    return pred, W, H

def render_video_with_landmarks(
    video_path: str,
    predictor_path: str,
    output_path: str | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    upsample_num_times: int = 0,
    device: str = "cpu",
    draw_indices: bool = False,
    circle_radius: int = 2,
    thickness: int = 1,
    circle_color=(0, 255, 0),
    text_color=(0, 255, 255),
    wait_ms: int = 1,
):
    """
    Trích landmark rồi chiếu video với landmark được vẽ lên.
    Nếu output_path != None thì sẽ lưu video đã vẽ (MP4).
    Trả về output_path hoặc None (nếu không lưu).
    """
    # Lấy landmarks (dạng torch) và chỉ số frame đã trích
    pts_torch, idx_torch, W, H = extract_from_video_dlib(
        video_path=video_path,
        predictor_path=predictor_path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        upsample_num_times=upsample_num_times,
        device=device,
    )

    # Nếu không có landmarks, vẫn chiếu video gốc hoặc thông báo
    if pts_torch.numel() == 0:
        print("Không tìm thấy face / landmark nào trong video.")
        # Mở video chỉ để hiển thị hoặc sao chép nếu output_path được yêu cầu
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None
        if output_path is not None:
            writer = cv2.VideoWriter(output_path, fourcc, fps, (int(W), int(H)))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Video (no landmarks)", frame)
            if writer is not None:
                writer.write(frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        return output_path if output_path is not None else None

    # Chuyển về numpy (cpu) cho dễ dùng
    pts_np = pts_torch.cpu().numpy()  # (F_extracted, N, 2)
    frame_indices = idx_torch.cpu().numpy().tolist()  # danh sách chỉ số frame gốc

    # Tạo dict map frame_index -> landmarks (N,2)
    frame_to_pts = {int(fr_idx): pts_np[i] for i, fr_idx in enumerate(frame_indices)}

    # Mở video gốc để render
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = None
    if output_path is not None:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Nếu frame này có landmarks thì vẽ
        if frame_idx in frame_to_pts:
            landmarks = frame_to_pts[frame_idx]  # (N,2) -- lưu y dưới dạng H - y
            # chuyển y về toạ độ OpenCV (origin top-left)
            xs = landmarks[:, 0]
            ys = H - landmarks[:, 1]  # chuyển lại
            # vẽ từng điểm
            for i, (x, y) in enumerate(zip(xs, ys)):
                # round to int
                xi = int(round(float(x)))
                yi = int(round(float(y)))
                cv2.circle(frame, (xi, yi), circle_radius, circle_color, thickness, lineType=cv2.LINE_AA)
                if draw_indices:
                    cv2.putText(frame, str(i), (xi + 3, yi - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)

        # Hiển thị frame
        cv2.imshow("Video with Landmarks", frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return output_path if output_path is not None else None


if __name__ == "__main__":
    start_time = time.perf_counter()
    result, W, H = predict_intrinsic(
                video_path="Data/Sample/2025-11-25 14-53-04 cam2.mp4",
                model_name="1600_lowrange",
                token="05_"
            )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    print(result)

    # start_time = time.perf_counter()
    # render_video_with_landmarks(
    #     video_path="Data/Sample/2025-11-25 14-53-04 cam4.mp4",
    #     predictor_path="Models/shape_predictor_68_face_landmarks.dat",
    #     output_path="Data/Sample/2025-11-25 14-53-04 cam4_landmark.mp4",
    #     frame_stride=2,
    #     max_frames=None
    #     upsample_num_times=0,
    #     device="cpu",
    #     draw_indices=False,
    #     circle_radius=3,
    #     thickness=-1,      
    #     wait_ms=10)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Execution time: {elapsed_time:.6f} seconds")
            
