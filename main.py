import cv2
import cv2.aruco as aruco
import dlib
import numpy as np
import torch
from model import Model
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional
import collections
import itertools
import os
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
        frame_stride=2,
        max_frames=100,
        upsample_num_times=0,
        device=device,
    )
    print(pts.shape)
    F, N, _ = pts.shape
    device = torch.device(device)
    pts = pts.to(device, dtype=torch.float32)

    pts = pts.reshape(F,2,N)

    model = Model(torch.tensor([1600/2,896/2,1]))
    model.load(model_name, token)
    model.set_eval()
    print(model.predict_intrinsic(pts).mean(0))
    _, pred, R, T = model.alternating_optimize(pts)
    print(R.shape, T.shape)
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
    
def invert_relative(R_ji: np.ndarray, T_ji: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given relative transform j_from_i (R_ji, T_ji) such that X_j = R_ji @ X_i + T_ji,
    return the inverse transform i_from_j: (R_ij, T_ij) such that X_i = R_ij @ X_j + T_ij.
    """
    R_ij = R_ji.T
    T_ij = - R_ij @ T_ji
    return R_ij, T_ij

def project_to_so3(M: np.ndarray) -> np.ndarray:
    """
    Project a 3x3 matrix M to the nearest rotation matrix using SVD (orthogonal Procrustes).
    """
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Fix possible reflection (determinant = -1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def compute_global_poses_from_pairwise(
    pairwise_results: List[Dict],
    ref_camera: Optional[int] = 0
) -> Dict[int, Dict]:
    """
    Input:
      pairwise_results: list of dicts, each dict must contain keys:
         - "cam_i" (int), "cam_j" (int),
         - "R_j_from_i" (3x3 numpy array or None),
         - "T_j_from_i" (3-vector numpy array or None)
      (This is the output format from pairwise_camera_calibration.)

    Output:
      A dict mapping camera_index -> {
         "R": 3x3 numpy rotation (global->cam),
         "T": 3-vector translation (global->cam),
         "estimates": list of (R,T) estimates used to fuse,
         "reachable": bool
      }
    Notes:
      - Poses are returned in the convention X_cam = R @ X_global + T.
      - Poses are determined up to the global reference frame (we set ref_camera pose to identity).
      - For cameras in disconnected components, they get their own identity-rooted pose if ref_camera is in same comp; otherwise they are marked unreachable.
    """
    # Build adjacency with relative transforms
    graph = collections.defaultdict(list)  # node -> list of (neighbor, R_neighbor_from_node, T_neighbor_from_node)
    cameras = set()
    for p in pairwise_results:
        i = int(p["cam_i"])
        j = int(p["cam_j"])
        cameras.add(i); cameras.add(j)
        Rji = p.get("R_j_from_i", None)
        Tji = p.get("T_j_from_i", None)
        if Rji is not None and Tji is not None:
            Rji = np.asarray(Rji)
            Tji = np.asarray(Tji).reshape(3,)
            graph[i].append((j, Rji, Tji))
            # add inverse edge j->i
            Rij, Tij = invert_relative(Rji, Tji)
            graph[j].append((i, Rij, Tij))
        # if relative missing, skip edge

    if len(cameras) == 0:
        return {}

    # choose reference camera: default 0 if present, else choose camera with largest degree
    if ref_camera is None or ref_camera not in cameras:
        # pick camera with highest degree (most connectivity) as reference
        degrees = {c: len(graph[c]) for c in cameras}
        ref_camera = max(degrees.items(), key=lambda kv: kv[1])[0]

    # We'll collect multiple estimates per camera and fuse them
    pose_estimates = {c: [] for c in cameras}  # camera -> list of (R, T)
    reachable = {c: False for c in cameras}

    # BFS from reference to get initial poses (single-path)
    from collections import deque
    q = deque()
    # ref pose = identity
    pose_estimates[ref_camera].append((np.eye(3), np.zeros(3)))
    reachable[ref_camera] = True
    q.append(ref_camera)
    visited = set([ref_camera])

    while q:
        src = q.popleft()
        # take the average of current estimates for src as the "source pose" to propagate from
        Rs = [est[0] for est in pose_estimates[src]]
        Ts = [est[1] for est in pose_estimates[src]]
        # simple fusion for source to get single representative pose for propagation:
        R_src = project_to_so3(np.sum(Rs, axis=0))
        T_src = np.mean(Ts, axis=0)
        for (nbr, R_n_from_s, T_n_from_s) in graph[src]:
            # compute pose of neighbor using the relation:
            # X_n = R_n_from_s @ X_s + T_n_from_s
            R_n = R_n_from_s @ R_src
            T_n = R_n_from_s @ T_src + T_n_from_s
            pose_estimates[nbr].append((R_n, T_n))
            reachable[nbr] = True
            if nbr not in visited:
                visited.add(nbr)
                q.append(nbr)

    # After BFS we may have multiple estimates per camera (if multiple incoming edges).
    # Fuse them now: average translations and sum rotation matrices then project to SO(3).
    fused = {}
    for c in cameras:
        ests = pose_estimates[c]
        if len(ests) == 0:
            fused[c] = {
                "R": None,
                "T": None,
                "estimates": [],
                "reachable": False
            }
            continue
        Rs = np.stack([e[0] for e in ests], axis=0)  # (k,3,3)
        Ts = np.stack([e[1] for e in ests], axis=0)  # (k,3)
        R_sum = np.sum(Rs, axis=0)
        R_fused = project_to_so3(R_sum)
        T_fused = np.mean(Ts, axis=0)
        fused[c] = {
            "R": R_fused,
            "T": T_fused,
            "estimates": ests,
            "reachable": reachable[c]
        }

    # Find disconnected components and report which cameras are not connected to ref
    disconnected = [c for c in cameras if not fused[c]["reachable"]]
    if len(disconnected) > 0:
        print(f"Warning: {len(disconnected)} cameras not reachable from reference camera {ref_camera}: {disconnected}")

    return fused

def calibrate_and_compute_globals(
    folder_path: str,
    model_name: str,
    token: str,
    device: str = "cpu",
    min_common_frames: int = 5,
    try_fallback: bool = True,
    ref_camera: Optional[int] = 0
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], List[Dict]]:
    """
    Runs pairwise camera calibration then computes global camera poses.

    Returns:
      - K_map: dict camera_index -> 3x3 intrinsic matrix K (np.ndarray)
      - global_R: dict camera_index -> 3x3 rotation (R: global -> cam)
      - global_t: dict camera_index -> 3-vector translation (T: global -> cam)
      - pairwise_results: the list of pairwise dicts (for debugging)
    Notes / assumptions:
      - uses Model(center).alternating_optimize(...) as in your code; attempts to extract intrinsics
        from pred_out or shape_out if present; otherwise uses fallback focal length.
      - pose convention: X_cam = R @ X_world + T
    """
    # --- load videos and extract landmarks (uses your extract_from_video_dlib) ---
    videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if len(videos) == 0:
        raise ValueError("No .mp4 files found in folder_path")

    video_landmark: List[Tuple[torch.Tensor, np.ndarray, int, int]] = []
    W, H = None, None
    for video in videos:
        pts, idx, W, H = extract_from_video_dlib(
            video_path=os.path.join(folder_path, video),
            predictor_path="Models/shape_predictor_68_face_landmarks.dat",
            frame_stride=1,
            max_frames=100,
            upsample_num_times=0,
            device=device,
        )
        idx_arr = np.asarray(idx, dtype=np.int64)
        video_landmark.append((pts, idx_arr, W, H))

    if W is None or H is None:
        raise RuntimeError("Failed to get W,H from video extraction.")
    center = torch.tensor([W/2, H/2, 1.0])
    model = Model(center)
    model.load(model_name, token)
    model.set_eval()
    device_torch = torch.device(device)

    # per-camera intrinsics storage (we aim to fill this)
    per_camera_K: Dict[int, np.ndarray] = {}

    def run_opt_for_camera_and_K(pts_tensor: torch.Tensor, cam_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Runs alternating_optimize and tries to return (R_np, T_np, K_np_or_None)
        """
        pts_t = pts_tensor.to(device_torch, dtype=torch.float32)
        if pts_t.dim() != 3:
            raise RuntimeError(f"pts must be 3-D tensor, got dim {pts_t.dim()}")
        F, a, b = pts_t.shape
        if a == 2:
            pts_for_model = pts_t
        elif b == 2:
            pts_for_model = pts_t.permute(0, 2, 1).contiguous()
        else:
            pts_for_model = pts_t.reshape(F, 2, -1)

        # call model
        shape_out, pred_out, R_out, T_out = model.alternating_optimize(pts_for_model)

        if R_out is None or T_out is None:
            raise RuntimeError("Model returned None for R or T")

        # convert to numpy
        R_np = R_out.detach().cpu().numpy() if isinstance(R_out, torch.Tensor) else np.asarray(R_out)
        T_np = T_out.detach().cpu().numpy() if isinstance(T_out, torch.Tensor) else np.asarray(T_out)

        # attempt to get intrinsics K
        K_np = None
        try:
            # Case 1: pred_out is a dict-like with 'K' or 'intrinsics'
            if isinstance(pred_out, dict):
                if "K" in pred_out:
                    k = pred_out["K"]
                    K_np = k.detach().cpu().numpy() if isinstance(k, torch.Tensor) else np.asarray(k)
                elif "intrinsics" in pred_out:
                    k = pred_out["intrinsics"]
                    K_np = k.detach().cpu().numpy() if isinstance(k, torch.Tensor) else np.asarray(k)
                elif "f" in pred_out or "fx" in pred_out:
                    # try common formats: fx, fy, cx, cy
                    fx = pred_out.get("fx", pred_out.get("f", None))
                    fy = pred_out.get("fy", fx)
                    cx = pred_out.get("cx", W/2)
                    cy = pred_out.get("cy", H/2)
                    if fx is not None:
                        fx_v = fx.detach().cpu().item() if isinstance(fx, torch.Tensor) else float(fx)
                        fy_v = fy.detach().cpu().item() if isinstance(fy, torch.Tensor) else float(fy)
                        cx_v = cx.detach().cpu().item() if isinstance(cx, torch.Tensor) else float(cx)
                        cy_v = cy.detach().cpu().item() if isinstance(cy, torch.Tensor) else float(cy)
                        K_np = np.array([[fx_v, 0.0, cx_v],
                                         [0.0, fy_v, cy_v],
                                         [0.0, 0.0, 1.0]])
            # Case 2: pred_out is a tensor / array of [fx, fy, cx, cy] or [f, cx, cy]
            elif isinstance(pred_out, (torch.Tensor, np.ndarray, list, tuple)):
                arr = pred_out.detach().cpu().numpy() if isinstance(pred_out, torch.Tensor) else np.asarray(pred_out)
                arr = arr.ravel()
                if arr.size >= 4:
                    fx_v, fy_v, cx_v, cy_v = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                    K_np = np.array([[fx_v, 0.0, cx_v],
                                     [0.0, fy_v, cy_v],
                                     [0.0, 0.0, 1.0]])
                elif arr.size == 3:
                    # assume [f, cx, cy] and fx=fy=f
                    f_v, cx_v, cy_v = float(arr[0]), float(arr[1]), float(arr[2])
                    K_np = np.array([[f_v, 0.0, cx_v],
                                     [0.0, f_v, cy_v],
                                     [0.0, 0.0, 1.0]])
            # Case 3: try shape_out if that encodes focal
            if K_np is None and shape_out is not None:
                # sometimes shape_out contains camera scale; attempt best-effort extraction:
                if isinstance(shape_out, (torch.Tensor, np.ndarray)):
                    s = shape_out.detach().cpu().numpy().ravel() if isinstance(shape_out, torch.Tensor) else np.asarray(shape_out).ravel()
                    if s.size >= 1:
                        # heuristic: convert a scale to focal
                        f_guess = float(np.abs(s[0])) * max(W, H)
                        K_np = np.array([[f_guess, 0.0, W/2.0],
                                         [0.0, f_guess, H/2.0],
                                         [0.0, 0.0, 1.0]])
        except Exception:
            K_np = None

        # final fallback if no K found: pinhole approximate
        if K_np is None:
            f_guess = 1.2 * max(W, H)
            K_np = np.array([[f_guess, 0.0, W/2.0],
                             [0.0, f_guess, H/2.0],
                             [0.0, 0.0, 1.0]])

        # store in per_camera_K (caller may overwrite later with a better estimate)
        per_camera_K[cam_idx] = K_np
        return R_np, T_np, K_np

    # --- pairwise calibration loop (mostly copied + adapted from your original function) ---
    results = []
    n_vid = len(video_landmark)
    for i, j in itertools.combinations(range(n_vid), 2):
        pts_i, idx_i, _, _ = video_landmark[i]
        pts_j, idx_j, _, _ = video_landmark[j]

        set_i = set(idx_i.tolist())
        set_j = set(idx_j.tolist())
        common = sorted(list(set_i.intersection(set_j)))

        used_frames = []
        status = "failed"
        note = ""
        Ri = Ti = Rj = Tj = None
        R_j_from_i = T_j_from_i = None

        if len(common) >= min_common_frames:
            pos_i_map = {int(fr): p for p, fr in enumerate(idx_i.tolist())}
            pos_j_map = {int(fr): p for p, fr in enumerate(idx_j.tolist())}
            positions_i = [pos_i_map[fr] for fr in common if fr in pos_i_map]
            positions_j = [pos_j_map[fr] for fr in common if fr in pos_j_map]

            if len(positions_i) == len(positions_j) and len(positions_i) >= min_common_frames:
                try:
                    R_i_np, T_i_np, K_i = run_opt_for_camera_and_K(pts_i[positions_i].to(device_torch, dtype=torch.float32), cam_idx=i)
                    R_j_np, T_j_np, K_j = run_opt_for_camera_and_K(pts_j[positions_j].to(device_torch, dtype=torch.float32), cam_idx=j)
                    Ri, Ti, Rj, Tj = R_i_np, T_i_np, R_j_np, T_j_np
                    used_frames = common
                    status = "ok"
                    note = f"used {len(common)} synchronized frames"
                except Exception as e:
                    status = "failed"
                    note = f"alternating_optimize failed on synchronized frames: {e}"
            else:
                status = "failed"
                note = "mapping positions mismatch (shouldn't happen)"
        else:
            note = f"not enough common frames ({len(common)} < {min_common_frames})"

        if status != "ok" and try_fallback:
            try:
                pos_all_i = list(range(pts_i.shape[0]))
                pos_all_j = list(range(pts_j.shape[0]))
                R_i_np, T_i_np, K_i = run_opt_for_camera_and_K(pts_i[pos_all_i].to(device_torch, dtype=torch.float32), cam_idx=i)
                R_j_np, T_j_np, K_j = run_opt_for_camera_and_K(pts_j[pos_all_j].to(device_torch, dtype=torch.float32), cam_idx=j)
                Ri, Ti, Rj, Tj = R_i_np, T_i_np, R_j_np, T_j_np
                used_frames = common
                status = "fallback"
                note = "used per-camera own frames as fallback (no/insufficient synchronized frames)"
            except Exception as e:
                status = "failed"
                note = f"fallback optimization failed: {e}"

        # If we have per-camera extrinsics, compute relative transform R_j_from_i and T_j_from_i
        if Ri is not None and Ti is not None and Rj is not None and Tj is not None:
            try:
                Ri = np.asarray(Ri)
                Rj = np.asarray(Rj)
                Ti = np.asarray(Ti).reshape(3,)
                Tj = np.asarray(Tj).reshape(3,)
                R_j_from_i = Rj @ Ri.T
                T_j_from_i = Tj - (R_j_from_i @ Ti)
            except Exception as e:
                note += f" | relative transform compute failed: {e}"

        results.append({
            "cam_i": i,
            "cam_j": j,
            "used_frame_indices": used_frames,
            "Ri": Ri,
            "Ti": Ti,
            "Rj": Rj,
            "Tj": Tj,
            "R_j_from_i": R_j_from_i,
            "T_j_from_i": T_j_from_i,
            "status": status,
            "note": note
        })

    # --- compute global poses from pairwise results ---
    # The compute_global_poses_from_pairwise implementation must be available (copy from your code)
    fused = compute_global_poses_from_pairwise(results, ref_camera=ref_camera)

    # Build K, global_R, global_t outputs
    K_map: Dict[int, np.ndarray] = {}
    global_R: Dict[int, np.ndarray] = {}
    global_t: Dict[int, np.ndarray] = {}
    for cam_idx in fused.keys():
        entry = fused[cam_idx]
        R_fused = entry["R"]
        T_fused = entry["T"]
        global_R[cam_idx] = R_fused
        global_t[cam_idx] = T_fused
        # If we computed per_camera_K earlier, use it; otherwise fallback to identity-like K
        if cam_idx in per_camera_K:
            K_map[cam_idx] = per_camera_K[cam_idx]
        else:
            f_guess = 1.2 * max(W, H)
            K_map[cam_idx] = np.array([[f_guess, 0.0, W/2.0],
                                       [0.0, f_guess, H/2.0],
                                       [0.0, 0.0, 1.0]])

    return K_map, global_R, global_t, results

if __name__ == "__main__":
    # while True:
    #     mode = input("Mode (Single/Folder/Exit): ")
    #     if mode == "Single":
    #         start_time = time.perf_counter()
    #         video_path = input("Video path: ")
    #         model_name = input("Model name: ")
    #         checkpoint = int(input("Checkpoint: "))
    #         result, W, H = predict_intrinsic(
    #                     video_path=video_path,
    #                     model_name=model_name,
    #                     token=f"{checkpoint:02d}_"
    #                 )
    #         end_time = time.perf_counter()
    #         elapsed_time = end_time - start_time
    #         print(f"Execution time: {elapsed_time:.6f} seconds")
    #         print(result)
    #     elif mode == "Folder":
    #         start_time = time.perf_counter()
    #         folder_path = input("Folder path: ")
    #         model_name = input("Model name: ")
    #         checkpoint = int(input("Checkpoint: "))
    #         result = calibrate_and_compute_globals(
    #             folder_path=folder_path,
    #             model_name=model_name,
    #             token=f"{checkpoint:02d}_",
    #         )
    #         end_time = time.perf_counter()
    #         elapsed_time = end_time - start_time
    #         print(f"Execution time: {elapsed_time:.6f} seconds")
    #         # print(result)
    #     elif mode == "Exit":
    #         break
    #     else:
    #         continue
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    board = aruco.CharucoBoard(
        size=(7, 5),
        squareLength=0.04,
        markerLength=0.02,
        dictionary=dictionary
    )

    img = board.generateImage((2000, 1400))
    cv2.imwrite("charuco.png", img)
        
