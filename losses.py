import torch
def projection_loss(pimg,pw,R,T,A):
    M = pimg.shape[0]
    N = pimg.shape[2]
    if len(pw.shape) == 3:
        pc = torch.bmm(R,pw.permute(0,2,1))
    else:
        pc = torch.bmm(R,torch.stack(M*[pw]).permute(0,2,1))
    pct = pc + T.unsqueeze(2)
    if len(A.shape) == 3:
        proj = torch.bmm(A,pct)
    else:
        proj = torch.bmm(torch.stack([A]*M),pct)
    proj_img = proj / proj[:,-1,:].unsqueeze(1)
    pimg_pred = proj_img[:,:2,:]
    loss = torch.norm(pimg - pimg_pred,dim=1)
    return loss

def motion_loss(Pc):
    M = Pc.shape[0]
    c1 = Pc[:M-1]
    c2 = Pc[1:]
    diff = c1 - c2
    return torch.norm(diff,dim=-1)

def principal_loss(K,p_hat):
    p = K[:,:,2]
    diff = p - p_hat
    return torch.mean(torch.norm(diff,dim=1))

def compute_reprojection_loss(x, S, R, T, A):
    pc = torch.bmm(S, R.permute(0, 2, 1)) + T.unsqueeze(1)
    proj = torch.bmm(pc, A.permute(0, 2, 1))
    pimg_pred = proj / proj[:, :, -1].unsqueeze(-1)
    diff = x - pimg_pred[:, :, :2]
    loss = torch.norm(diff, dim=2).mean()
    return loss



def shape_loss(betas, x, S_pred, S_gt, R_gt, T_gt, k_gt, epoch, logging = False):
    B = x.shape[0]
    repro_loss = compute_reprojection_loss(
        x.permute(0, 2, 1), S_pred, R_gt, T_gt, 
        k_gt.unsqueeze(0).repeat(B, 1, 1)
    )

    diff = S_pred - S_gt.unsqueeze(0).repeat(B, 1, 1)
    distances = torch.norm(diff, dim=2)
    distance_loss = distances.mean()

    beta_reg = torch.mean(betas ** 2)
    beta_std = betas.std(0).mean()
    diversity_loss = torch.exp(-beta_std * 10)
    
    w_repro = 1.0
    w_distance = min(0.1 * (epoch / 100), 0.5)
    w_beta_reg = 0.001
    w_diversity = max(0.5 * (1 - epoch / 500), 0.05)
    
    total_loss = (
        w_repro * repro_loss +
        w_distance * distance_loss +
        w_beta_reg * beta_reg +
        w_diversity * diversity_loss
    )
    if logging:
        losses = {
            'total': total_loss,
            'repro': repro_loss,
            'distance': distance_loss,
            'beta_reg': beta_reg,
            'diversity': diversity_loss,
        }
        return total_loss, losses
    else:
        return total_loss
