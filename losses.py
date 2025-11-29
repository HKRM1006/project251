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