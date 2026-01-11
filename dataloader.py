from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
import numpy as np
import os
from smplx import FLAME
import pickle
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = r"Models"
STATIC_EMB = os.path.join(MODEL_DIR, "flame_static_embedding.pkl")
flame = FLAME(model_path=MODEL_DIR, use_face_contour=False).to(device)
faces = None
if hasattr(flame, "faces"):
    faces = np.array(flame.faces.astype(np.int64))
elif hasattr(flame, "faces_tensor"):
    faces = flame.faces_tensor.detach().cpu().numpy()
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
def get_flame_shape_and_expression_space(flame_model):
    if hasattr(flame_model, "v_template"):
        v_template = flame_model.v_template.detach().cpu().numpy()
    else:
        raise RuntimeError("FLAME model has no attribute 'v_template'")
    shapedirs = None  
    for name in ("shapedirs", "shapedirs_tensor", "shape_dirs"):  
        if hasattr(flame_model, name):  
            shapedirs_attr = getattr(flame_model, name)  
            if isinstance(shapedirs_attr, torch.Tensor):  
                shapedirs = shapedirs_attr.detach().cpu().numpy()  
            else:  
                shapedirs = np.array(shapedirs_attr)  
            break  
    if shapedirs is None:  
        raise RuntimeError("FLAME model has no shapedirs (principal components) attribute")  
    if shapedirs.ndim == 3:  
        if shapedirs.shape[0] == v_template.shape[0] and shapedirs.shape[1] == 3:  
            shapedirs_np = shapedirs  
        elif shapedirs.shape[0] == shapedirs.shape[2] and shapedirs.shape[1] == v_template.shape[0]:  
            shapedirs_np = np.transpose(shapedirs, (1,2,0))  
        else:  
            shapedirs_np = np.transpose(shapedirs, (1,2,0)) if shapedirs.shape[0] < shapedirs.shape[2] else shapedirs  
    else:  
        raise RuntimeError(f"Unexpected shapedirs shape: {shapedirs.shape}")  
    return v_template, shapedirs_np


class SyntheticLoader(Dataset):
    def __init__(self):
        self.mu_lm, self.lm_eigenvec = get_flame_shape_and_expression_space(flame)
        n_shape = self.lm_eigenvec.shape[2]
        self.sigma = torch.ones(n_shape, device=device)

        # video sequence length
        self.M = 100
        self.N = 68
        self.res = 1600

        # extra boundaries on camera coordinates
        self.w = 1600
        self.h = 896
        self.minz = 1.5
        self.maxz = 3.5
        self.max_rx = 20
        self.max_ry = 20 
        self.max_rz = 20

    def sample_intrinsics(self):
        fx = np.random.uniform(0.6, 1.2) * self.w
        cx = np.clip(np.random.normal(0.5, 0.01), 0.47, 0.53) * self.w
        cy = np.clip(np.random.normal(0.5, 0.01), 0.47, 0.53) * self.h
        fy = fx * np.random.uniform(0.98, 1.02)

        return np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,   0,  1]
        ], dtype=np.float32)
    
    def get_shape(self, betas):
        B = betas.shape[0]
        device = betas.device
        dtype = betas.dtype

        mu_s = torch.from_numpy(self.mu_lm).to(device=device, dtype=dtype)
        shapedirs = torch.from_numpy(self.lm_eigenvec).to(device=device, dtype=dtype)
        shapedirs = shapedirs * self.sigma.view(1,1,-1)

        offsets = torch.einsum('vcn,bn->bvc', shapedirs, betas)
        vertices = mu_s.unsqueeze(0) + offsets
        #vertices = vertices - vertices.mean(dim=1, keepdim=True)

        face_idx = torch.from_numpy(lmk_face_idx).long().to(device)
        bary_coords = torch.from_numpy(lmk_bary).to(device, dtype=dtype)

        f = faces[face_idx]
        v0 = vertices[:, f[:,0], :]
        v1 = vertices[:, f[:,1], :]
        v2 = vertices[:, f[:,2], :]

        bary_coords = bary_coords.unsqueeze(0)
        landmarks = bary_coords[:,:,0:1]*v0 + bary_coords[:,:,1:2]*v1 + bary_coords[:,:,2:3]*v2  # (B, N_lmk, 3)

        return landmarks

    
    def generate_random_face_from_flame(self, device=device):
        n_verts = self.mu_lm.shape[0]
        n_shape = self.lm_eigenvec.shape[2]
        shapedirs_flat = torch.from_numpy(self.lm_eigenvec[:, :, :n_shape].reshape(n_verts*3, n_shape)).to(device)
        mu_s = torch.from_numpy(self.mu_lm.reshape(n_verts*3)).to(device)
        sigma_vec = torch.ones(n_shape, device=device)
        alphas = (torch.rand(n_shape, device=device) * 2 - 1) * 2.0
        coeffs = sigma_vec * alphas
        delta = shapedirs_flat @ coeffs
        vertices = (mu_s + delta).reshape(n_verts, 3)
        return vertices, alphas, coeffs
    
    def __getitem__(self,idx):
        # data holders
        M = self.M
        N = self.N
        x_w = np.zeros((N,3))
        x_cam = np.zeros((M,N,3))
        x_img = np.zeros((M,N,2))
        x_img_true = np.zeros((M,N,2))

        K_norm = torch.tensor(self.sample_intrinsics(), device=device)
        K = K_norm.clone()

        vertices, alpha, coeffs = self.generate_random_face_from_flame()
        x_w = self.landmarks_from_bary(vertices, faces, lmk_face_idx, lmk_bary)


        while True:
            r_init, q_init = self.sample_rot()
            r_final, q_final = self.sample_rot()

            t_init = self.sample_translation(K)
            t_final = self.sample_translation(K)
            ximg_init = self.project2d(r_init,t_init,K,x_w)
            ximg_final = self.project2d(r_final,t_final,K,x_w)

            if torch.any(torch.min(ximg_init, dim=0).values < 0): continue
            if torch.any(torch.min(ximg_final, dim=0).values < 0): continue
            if torch.any(torch.min(ximg_init, dim=1).values < 0): continue
            if torch.any(torch.min(ximg_final, dim=1).values < 0): continue

            init  = torch.max(ximg_init, dim=0).values
            final = torch.max(ximg_final, dim=0).values

            if init[0]  > self.w: continue
            if final[0] > self.w: continue
            if init[1]  > self.h: continue
            if final[1] > self.h: continue
            break

        qs = np.stack((q_init,q_final))
        Rs = R.from_quat(qs)
        times = np.linspace(0,1,M)
        slerper = Slerp([0,1],Rs)
        rotations = slerper(times)
        matrices = rotations.as_matrix()

        T = np.stack((np.linspace(t_init[0],t_final[0],M),
                np.linspace(t_init[1],t_final[1],M),
                np.linspace(t_init[2],t_final[2],M))).T

        xw_3n = x_w.T[np.newaxis, :, :]
        xw_3n = np.broadcast_to(xw_3n, (M, 3, N)) 
        x_cam = np.matmul(matrices, xw_3n) + T[:, :, np.newaxis]

        K_batch = np.broadcast_to(K[np.newaxis, :, :], (M, 3, 3))
        proj = np.matmul(K_batch, x_cam)

        proj = proj.transpose(0, 2, 1)
        x_img_true = proj[:, :, :2] / proj[:,:,2:]
        x_img = x_img_true + 0.5*np.random.randn(M, N, 2)

        sample = {}
        sample['alpha_gt'] = alpha
        sample['x_w_gt'] = torch.from_numpy(x_w).float()
        sample['x_cam_gt'] = torch.from_numpy(x_cam).float()
        sample['x_img'] = torch.from_numpy(x_img).float().permute(0,2,1)
        sample['x_img_gt'] = torch.from_numpy(x_img_true).float().permute(0,2,1)
        sample['K_gt'] = K_norm
        sample['T_gt'] = torch.from_numpy(T).float()
        sample['R_gt'] = torch.from_numpy(matrices).float()
        sample['f_gt'] = torch.Tensor(K[0][0])
        return sample

    def sample_rot(self):
        rx = (np.random.rand()*2 - 1) * np.deg2rad(self.max_rx)
        ry = (np.random.rand()*2 - 1) * np.deg2rad(self.max_ry)
        rz = (np.random.rand()*2 - 1) * np.deg2rad(self.max_rz)
        rot = R.from_euler('zyx', [rz, ry, rx])
        return rot.as_matrix(), rot.as_quat()

    def sample_translation(self, K):
        K_inv = np.linalg.inv(K)
        vx = K_inv @ np.array([self.w, self.w/2, 1])
        vy = K_inv @ np.array([self.h/2, self.h, 1])
        vz = np.array([0,0,1])
        thetax = np.arctan2(np.linalg.norm(np.cross(vz, vy)), np.dot(vz, vy))
        thetay = np.arctan2(np.linalg.norm(np.cross(vz, vx)), np.dot(vz, vx))
        tz = np.random.rand() * (self.maxz - self.minz) + self.minz
        maxx = tz * np.tan(thetax)
        maxy = tz * np.tan(thetay)
        tx = np.random.rand()*2*maxx - maxx
        ty = np.random.rand()*2*maxy - maxy
        t = np.array([tx, ty, tz])
        return t

    def landmarks_from_bary(self, vertices, faces, face_idx, bary_coords):
        face_idx = np.asarray(face_idx, dtype=np.int64).reshape(-1)
        bary = np.asarray(bary_coords, dtype=np.float32).reshape(-1,3)
        lmk_coords = np.zeros((face_idx.shape[0], 3), dtype=np.float32)
        for i, fi in enumerate(face_idx):
            f = faces[fi]
            v0 = vertices[f[0]]
            v1 = vertices[f[1]]
            v2 = vertices[f[2]]
            b = bary[i]
            lmk_coords[i] = b[0]*v0 + b[1]*v1 + b[2]*v2
        return lmk_coords
    
    def project2d(self,R,t,K,x_w):

        r_mat = torch.tensor(R, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        xc = (r_mat @ x_w.T + t.unsqueeze(1)).T

        proj = K @ xc.T
        proj = proj / proj[2,:]
        ximg = proj[:2,:].T

        return ximg
    




def main():
    def make_intrinsics_range_fn(w, h, f_min, f_max, rng_seed=None):
        rng = np.random.RandomState(rng_seed)

        fx = rng.uniform(f_min, f_max)
        fy = fx * rng.uniform(0.98, 1.02)

        cx = w * np.clip(np.random.normal(0.5, 0.01), 0.47, 0.53)
        cy = h * np.clip(np.random.normal(0.5, 0.01), 0.47, 0.53)

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        def _fn():
            return K.copy()

        return _fn
    
    def round100(x):
        return int(round(x / 100)) * 100
    
    output_file = "dataset/test1.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    loader = SyntheticLoader()
    w, h = loader.w, loader.h

    f_min_global = round100(0.6 * w)
    f_max_global = round100(1.2 * w)
    f_values = list(range(f_min_global, f_max_global - 99, 100))

    all_data = []
    global_idx = 0

    print("Generating f values:", f_values)

    for f_pix in f_values:
        print(f"\n=== Generating for base f = {f_pix} px ===")

        # range for fx
        low_fx = f_pix
        high_fx = f_pix + 100

        for i in range(10):
            # deterministic seed per (f_pix, i)
            seed = (f_pix * 1000003) ^ (i * 9176)

            loader.sample_intrinsics = make_intrinsics_range_fn(
                w, h,
                low_fx, high_fx,
                rng_seed=seed
            )

            # get one sample
            sample = loader.sample() if hasattr(loader, "sample") else loader[0]

            all_data.append({
                "global_index": global_idx,
                "f_range": (low_fx, high_fx),
                "f_base": f_pix,
                "sub_index": i,
                "data": sample,
            })

            print(f"  Generated video {i} (global {global_idx}) | fx∈[{low_fx}, {high_fx}]")
            global_idx += 1

    torch.save(all_data, output_file)
    print("\nDONE. Saved to", output_file)


if __name__ == "__main__":
    main()