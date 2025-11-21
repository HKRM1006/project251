import os, pickle, numpy as np, torch
from smplx import FLAME
import math
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"Models"
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
    betas_scale: float = 3.0,
    expr_base_scale: float = 0.18,
    expr_jitter: float = 0.03,
    expr_smooth: float = 0.90,
    expr_cap_norm: float | None = 0.2,  
    jaw_max_deg: float = 10.0,          
    head_max_yaw: float = 30.0,
    head_max_pitch: float = 25.0,
    head_max_roll: float = 25.0,
    trans_xy_max: float = 0.12,
    trans_z_max: float = 0.18,    
    fps: float = 30.0
) -> np.ndarray:
    dt = 1.0 / max(1.0, fps)
    pose_dim = get_pose_dim(flame)

    # mild OU drivers for head/jaw
    ou_jaw   = OU(mu=0.0, theta=1.8, sigma=30.0, dt=dt, x0=0.0, clamp=(-jaw_max_deg, jaw_max_deg))
    ou_yaw   = OU(mu=0.0, theta=3.0, sigma=8.0,  dt=dt, x0=0.0, clamp=(-head_max_yaw, head_max_yaw))
    ou_pitch = OU(mu=0.0, theta=1.2, sigma=15.0, dt=dt, x0=0.0, clamp=(-head_max_pitch, head_max_pitch))
    ou_roll  = OU(mu=0.0, theta=1.2, sigma=12.0, dt=dt, x0=0.0, clamp=(-head_max_roll, head_max_roll))
    ou_tx = OU(mu=0.0, theta=1.8, sigma=0.20, dt=dt,
               x0=0.0, clamp=(-trans_xy_max, trans_xy_max))
    ou_ty = OU(mu=0.0, theta=1.8, sigma=0.20, dt=dt,
               x0=0.0, clamp=(-trans_xy_max, trans_xy_max))
    ou_tz = OU(mu=0.0, theta=1.8, sigma=0.18, dt=dt,
               x0=0.0, clamp=(-trans_z_max, trans_z_max))

    with torch.no_grad():
        betas = (torch.empty(1, flame.num_betas, device=device)
                    .uniform_(-betas_scale, betas_scale))  # 3DMM α ~ U(-3,3)
        expr  = (torch.empty(1, flame.num_expression_coeffs, device=device)
                    .uniform_(-betas_scale, betas_scale) * expr_base_scale)

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

            jaw_deg   = ou_jaw.step()
            yaw_deg   = ou_yaw.step()
            pitch_deg = ou_pitch.step()
            roll_deg  = ou_roll.step()

            jaw_pose = torch.tensor([[math.radians(jaw_deg), 0.0, 0.0]],
                                    dtype=torch.float32, device=device)

            pose_vec = torch.zeros(1, pose_dim, device=device)
            if pose_dim >= 6:
                pose_vec[:, 3:6] = jaw_pose

            expr_use = expr * w  # apply damping

            try:
                out = flame(betas=betas, expression=expr_use, jaw_pose=jaw_pose)
            except TypeError:
                out = flame(betas=betas, expression=expr_use, pose=pose_vec)

            verts = out.vertices[0].cpu().numpy()

            # head rotation
            R_head = Rz(roll_deg) @ Ry(yaw_deg) @ Rx(pitch_deg)

            tx = ou_tx.step()
            ty = ou_ty.step()
            tz = ou_tz.step()
            T_head = np.array([tx, ty, tz], np.float32)

            verts_dyn = (R_head @ verts.T).T + T_head[None, :]

            tri = flame.faces[lmk_face_idx]
            v0, v1, v2 = verts_dyn[tri[:,0]], verts_dyn[tri[:,1]], verts_dyn[tri[:,2]]
            lm3d = (lmk_bary[:, [0]] * v0 +
                    lmk_bary[:, [1]] * v1 +
                    lmk_bary[:, [2]] * v2).astype(np.float32)
            lm_seq.append(lm3d)
    betas = betas.squeeze(0)
    return np.stack(lm_seq, 0).astype(np.float32), betas


# spherical coordinates to cartesian coordinates
def sph_to_cart(az_deg, el_deg, dist):
    az = np.radians(az_deg); el = np.radians(el_deg)
    x = dist * np.cos(el) * np.sin(az)
    y = dist * np.sin(el)
    z = dist * np.cos(el) * np.cos(az)
    return np.array([x, y, z], np.float32)


def sample_intrinsics(W, H):
    fx_norm = np.random.uniform(0.45, 0.9)
    cx_norm = np.clip(np.random.normal(0.5, 0.01), 0.47, 0.53)
    cy_norm = np.clip(np.random.normal(0.5, 0.01), 0.47, 0.53)

    fx = fx_norm * W
    fy = fx * np.random.uniform(0.98, 1.02)
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
    Xw_seq, W=1280, H=960,
    az_deg=10.0, el_deg=-5.0, dist=0.9,
    flip_y: bool = True
):
    F, N, _ = Xw_seq.shape
    K = sample_intrinsics(W, H)

    C = sph_to_cart(az_deg, el_deg, dist).astype(np.float32)
    T = np.zeros(3, np.float32)

    R, t = look_at_Rt_target(C, T)

    all_uv = []
    for f in range(F):
        Xw = Xw_seq[f] 

        Xc = (R @ Xw.T).T + t
        Z  = Xc[:, 2:3].copy()
        Z[Z < 1e-6] = 1e-6

        u = K[0,0]*(Xc[:,0:1]/Z) + K[0,2]
        v = K[1,1]*(Xc[:,1:2]/Z) + K[1,2]

        if flip_y:
            v_img = H - v
        else:
            v_img = v

        u_img = u

        all_uv.append(np.concatenate([u_img, v_img], 1).astype(np.float32))

    pts_2d = torch.from_numpy(np.stack(all_uv, 0)).to(device)  # (F,N,2)
    return pts_2d, K
if __name__ == "__main__":
    x, y = [], []
    ws, hs = [], []
    betas_list = []
    resolutions = [
        (640, 480),
        (1920, 1080),
        (3840, 2160)
    ]

    NUM_SAMPLES = int(input("Number of sample (each resolution): "))
    FRAMES_PER_SAMPLE = int(input("Frame per sample: "))
    filename = input("Dataset name: ")
    for (W, H) in resolutions:
        for _ in range(NUM_SAMPLES):
            dist = np.random.uniform(1.5, 2.5)
            # ---- 3D landmarks ----
            lm3d, beta = generate_landmark_3d_sequence(FRAMES_PER_SAMPLE)

            # ---- project to chosen resolution ----
            pts_2ds, K = project_onecam_video(lm3d, W=W, H=H, dist = dist)
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
            betas_list.append(beta.cpu().numpy().astype(np.float32))

    x = torch.tensor(np.array(x, dtype=np.float32))     # (N,F,68,2)
    y = torch.tensor(np.array(y, dtype=np.float32))     # (N,4)
    ws = torch.tensor(ws)
    hs = torch.tensor(hs)
    betas = torch.tensor(np.stack(betas_list, axis=0), dtype=torch.float32)
    
    os.makedirs("dataset", exist_ok=True)
    torch.save({
        "x": x,
        "y": y,
        "W": ws,
        "H": hs,
        "Beta": betas,
    }, "dataset/"+ filename + ".pt")



