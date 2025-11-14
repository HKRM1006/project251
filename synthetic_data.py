import os, pickle, numpy as np, torch
from smplx import FLAME
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from typing import Tuple, List
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"D:/Calibration/Models"
STATIC_EMB = os.path.join(MODEL_DIR, "flame_static_embedding.pkl")

flame = FLAME(model_path=MODEL_DIR, use_face_contour=False).to(device)
def load_static_embedding(path): 
    with open(path, "rb") as f: 
        emb = pickle.load(f, encoding="latin1") 
        key_pairs = [ ("lmk_face_idx", "lmk_bary_coords"), 
                     ("static_lmk_faces_idx", "static_lmk_bary_coords"), 
                     ("lmk_face_idx", "lmk_b_coords"), ] 
        for k_faces, k_bary in key_pairs: 
            if k_faces in emb and k_bary in emb: 
                face_idx = np.asarray(emb[k_faces]).reshape(-1) 
                bary = np.asarray(emb[k_bary]).reshape(-1, 3) 
                return face_idx, bary 
        raise KeyError(f"Cannot find expected keys in {path}. Available keys: {list(emb.keys())}")
lmk_face_idx, lmk_bary = load_static_embedding(STATIC_EMB)

#Rotation around axis
def Rx(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float32)

def Ry(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32)

def Rz(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32)

# get dimension according to flame version 
def get_pose_dim(flame):
    return (getattr(flame, "num_pose_params", None)
            or (getattr(flame, "pose_shape", [1, -1])[1] if hasattr(flame, "pose_shape") else 12))

# --- Smooth stochastic driver (Ornstein–Uhlenbeck) ---
class OU:
    def __init__(self, mu=0.0, theta=0.8, sigma=8.0, dt=1/30, x0=0.0, clamp=None):
        """
        dX = theta*(mu - X)*dt + sigma*sqrt(dt)*N(0,1). Values in 'degrees' if you use deg amplitudes.
        clamp: (-amax, +amax) if you want hard bounds.
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x = x0
        self.clamp = clamp

    def step(self):
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.x += dx
        if self.clamp is not None:
            lo, hi = self.clamp
            self.x = float(np.clip(self.x, lo, hi))
        return self.x

def generate_landmark_3d_sequence(
    num_frames: int,
    betas_scale: float = 2.0,
    expr_base_scale: float = 0.15,
    expr_jitter: float = 0.04,
    expr_smooth: float = 0.98,
    expr_cap_norm: float | None = 0.6,  
    jaw_max_deg: float = 12.0,          
    head_max_yaw: float = 15.0,
    head_max_pitch: float = 8.0,
    head_max_roll: float = 6.0,
    fps: float = 30.0
) -> np.ndarray:
    dt = 1.0 / max(1.0, fps)
    pose_dim = get_pose_dim(flame)

    # mild OU drivers for head/jaw
    ou_jaw   = OU(mu=0.0, theta=1.8, sigma=30.0, dt=dt, x0=0.0, clamp=(-jaw_max_deg, jaw_max_deg))
    ou_yaw   = OU(mu=0.0, theta=1.2, sigma=20.0, dt=dt, x0=0.0, clamp=(-head_max_yaw, head_max_yaw))
    ou_pitch = OU(mu=0.0, theta=1.2, sigma=15.0, dt=dt, x0=0.0, clamp=(-head_max_pitch, head_max_pitch))
    ou_roll  = OU(mu=0.0, theta=1.2, sigma=12.0, dt=dt, x0=0.0, clamp=(-head_max_roll, head_max_roll))

    with torch.no_grad():
        betas = torch.randn(1, flame.num_betas, device=device) * betas_scale
        expr  = torch.randn(1, flame.num_expression_coeffs, device=device) * expr_base_scale

        # optional per-coeff damping
        w = torch.ones(flame.num_expression_coeffs, device=device) * 0.7
        w[:10] *= 0.5  # damp top modes a bit more

        lm_seq = []
        for _ in range(num_frames):

            expr = expr_smooth * expr + (1.0 - expr_smooth) * (torch.randn_like(expr) * expr_jitter)

            if expr_cap_norm is not None:
                n = torch.linalg.norm(expr)
                if float(n) > expr_cap_norm:
                    expr *= (expr_cap_norm / (n + 1e-8))

            jaw_deg   = ou_jaw.step() # jaw
            yaw_deg   = ou_yaw.step() # head turn
            pitch_deg = ou_pitch.step() # head look up/down
            roll_deg  = ou_roll.step() # tilting

            jaw_pose = torch.tensor([[math.radians(jaw_deg), 0.0, 0.0]], dtype=torch.float32, device=device)

            pose_vec = torch.zeros(1, pose_dim, device=device)
            if pose_dim >= 6:
                pose_vec[:, 3:6] = jaw_pose

            expr_use = expr * w  # apply damping

            try:
                out = flame(betas=betas, expression=expr_use, jaw_pose=jaw_pose)
            except TypeError:
                out = flame(betas=betas, expression=expr_use, pose=pose_vec)

            verts = out.vertices[0].cpu().numpy()

            R_head = Rz(roll_deg) @ Ry(yaw_deg) @ Rx(pitch_deg)
            verts_dyn = (R_head @ verts.T).T

            tri = flame.faces[lmk_face_idx]
            v0, v1, v2 = verts_dyn[tri[:,0]], verts_dyn[tri[:,1]], verts_dyn[tri[:,2]]
            lm3d = (lmk_bary[:, [0]] * v0 +
                    lmk_bary[:, [1]] * v1 +
                    lmk_bary[:, [2]] * v2).astype(np.float32)
            lm_seq.append(lm3d)

    return np.stack(lm_seq, 0).astype(np.float32)

# spherical coordinates to cartesian coordinates
def sph_to_cart(az_deg, el_deg, dist):
    az = np.radians(az_deg); el = np.radians(el_deg)
    x = dist * np.cos(el) * np.sin(az)
    y = dist * np.sin(el)
    z = dist * np.cos(el) * np.cos(az)
    return np.array([x, y, z], np.float32)


def sample_intrinsics(W, H):
    fx_norm = np.random.uniform(0.35, 1.2)
    fy_norm = fx_norm + np.random.uniform(-0.05, 0.05)

    cx_norm = np.random.normal(0.5, 0.02)
    cy_norm = np.random.normal(0.5, 0.02)

    fx = fx_norm * W
    fy = fy_norm * H
    cx = cx_norm * W
    cy = cy_norm * H

    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0,  1]
    ], dtype=np.float32)


def look_at_Rt_target(C, T, up=np.array([0,1,0], np.float32)):
    z = (T - C); z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0).astype(np.float32)
    t = (-R @ C.astype(np.float32)).astype(np.float32)
    return R, t

def project_onecam_video(
    Xw_seq, W=1280, H=960, K=None,
    base_az=10.0, base_el=-5.0, base_dist=0.9,
    right_per_frame=0.02,   # meters along camera-right each frame
    up_per_frame=0.00,      # meters along camera-up each frame
    roll_per_frame_deg=0.0
):
    """
    Moves the camera laterally (strafe) on a sphere of constant radius around the origin.
    That keeps distance constant -> no zooming/size drift.
    """
    F, N, _ = Xw_seq.shape
    if K is None:
        K = sample_intrinsics(W, H)

    # start on the viewing sphere
    C = sph_to_cart(base_az, base_el, base_dist).astype(np.float32)
    T = np.zeros(3, np.float32)

    all_uv = []
    for f in range(F):
        # Look-at to get camera axes in world coords
        R, t = look_at_Rt_target(C, T)
        if roll_per_frame_deg != 0.0:
            R = Rz(f * roll_per_frame_deg) @ R

        # R rows are [x_world; y_world; z_world]
        x_world = R[0]  # camera right in world coords
        y_world = R[1]  # camera up in world coords

        # Strafe in tangent plane (no radial component)
        C = C + right_per_frame * x_world + up_per_frame * y_world

        # Re-project back to the sphere to keep constant distance
        C = (base_dist / (np.linalg.norm(C) + 1e-8)) * C

        # Recompute extrinsics after moving C
        R, t = look_at_Rt_target(C, T)
        if roll_per_frame_deg != 0.0:
            R = Rz(f * roll_per_frame_deg) @ R

        # Project
        Xw = Xw_seq[f]
        Xc = (R @ Xw.T).T + t
        Z  = Xc[:, 2:3].copy(); Z[Z < 1e-6] = 1e-6
        u = K[0,0]*(Xc[:,0:1]/Z) + K[0,2]
        v = K[1,1]*(Xc[:,1:2]/Z) + K[1,2]
        all_uv.append(np.concatenate([u, v], 1).astype(np.float32))

    return torch.from_numpy(np.stack(all_uv, 0)).to(device), K
if __name__ == "__main__":
    x, y = [], []
    ws, hs = [], []

    resolutions = [
        # (1280, 960),
        # (1920, 1080),
        # (2560, 1440),
        (3840, 2160)
    ]

    NUM_SAMPLES = 1
    FRAMES_PER_SAMPLE = 120

    for (W, H) in resolutions:
        for _ in range(NUM_SAMPLES):

            # ---- 3D landmarks ----
            lm3d = generate_landmark_3d_sequence(FRAMES_PER_SAMPLE)

            # ---- project to chosen resolution ----
            pts_2ds, K = project_onecam_video(lm3d, W, H)
            pts_2ds = pts_2ds.cpu().numpy()   # (F,68,2)

            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]

            # ---- normalize 2D landmarks ----
            pts_norm = pts_2ds.copy()
            pts_norm[..., 0] = (pts_2ds[..., 0] - W * 0.5) / W
            pts_norm[..., 1] = (pts_2ds[..., 1] - H * 0.5) / H

            # ---- normalize intrinsics ----
            y_norm = np.array([
                fx / W,
                fy / H,
                cx / W,
                cy / H
            ], dtype=np.float32)

            x.append(pts_norm)
            y.append(y_norm)
            ws.append(W)
            hs.append(H)

    x = torch.tensor(np.array(x, dtype=np.float32))     # (N,F,68,2)
    y = torch.tensor(np.array(y, dtype=np.float32))     # (N,4)
    ws = torch.tensor(ws)
    hs = torch.tensor(hs)

    os.makedirs("dataset", exist_ok=True)
    torch.save({
        "x": x,
        "y": y,
        "W": ws,
        "H": hs,
    }, "dataset/synthetic_face_dataset_test.pt")


    # colors = ["red", "green", "blue", "orange"]
    # plt.figure(figsize=(12.8, 9.6), dpi=100)  # Optional: match image resolution
    # for i, pts in enumerate(pts_2ds):
    #     plt.scatter(pts[:, 0], pts[:, 1], s=25, color=colors[i], label=f"Cam {i}")

    # plt.xlim(0, 1280)  # Fix x-axis to image width
    # plt.ylim(0, 960)   # Fix y-axis to image height (invert to match image coordinates)

    # plt.legend()
    # plt.gca().set_aspect("equal")
    # plt.xlabel("u")
    # plt.ylabel("v")
    # plt.title("All cameras overlaid")
    # plt.tight_layout()
    # plt.show()

    # for pts_2d in pts_2ds:
    #     plt.clf()  # clear previous frame
    #     plt.scatter(pts_2d[:, 0], pts_2d[:, 1], s=30)
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.xlim(0, 1280)
    #     plt.ylim(0, 960)
    #     plt.xlabel("u")
    #     plt.ylabel("v")
    #     plt.tight_layout()
    #     plt.pause(1.0/30)  # <-- show frame for ~100 ms
    #plt.close()
