import torch
from torch.utils.data import DataLoader, TensorDataset
import kornia.geometry.calibration as kp
def load_dataset_pt(path="dataset/synthetic_face_dataset.pt"):
    data = torch.load(path)
    x = data["x"].float()
    K = data["y"].float()
    W = data["W"].float()
    H = data["H"].float()
    betas = data["Beta"].float()
    N, F, C, _ = x.shape   # C = 68
    x = x.reshape(N, F * C, 2)
    return x, K, W, H, betas

def split_dataset(x, y, betas, val_ratio=0.2, shuffle=True):
    N = x.shape[0]
    idx = torch.arange(N)
    if shuffle:
        idx = idx[torch.randperm(N)]
    val_size = int(N * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val,   y_val   = x[val_idx],   y[val_idx]
    betas_train, betas_val =  betas[train_idx], betas[val_idx]
    return x_train, y_train, x_val, y_val, betas_train, betas_val

def build_dataloaders(
        x_train, y_train, betas_train,
        x_val, y_val, betas_val,
        batch_size=16, shuffle_train=True):

    train_ds = TensorDataset(x_train, y_train, betas_train)
    val_ds   = TensorDataset(x_val,   y_val,   betas_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader

def create_K(param):
    M = param.shape[0]
    K = torch.zeros((M,3,3)).float().to(param.device)
    K[:,0,0] = param[:,0]
    K[:,1,1] = param[:,1]
    K[:,0,2] = param[:,2]
    K[:,1,2] = param[:,3]
    K[:,2,2] = 1
    return K


def PnP(x_i,x_w,K):
    B, N, _ = x_i.shape

    x_i = x_i.float()
    x_w = x_w.float()
    K = K.float()

    A = kp.solve_pnp_dlt(x_w, x_i, K)
    R = A[:, :, :3]
    T = A[:, :, 3]

    Xc = torch.bmm(R, x_w.transpose(1, 2)) + T.unsqueeze(2)
    Xc = Xc.transpose(1, 2)      

    return Xc, R, T

