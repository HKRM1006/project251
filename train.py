from model import Model
import losses
import dataloader
import torch
import itertools
import utils
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def compute_reprojection_loss(x,S,A):
    Xc, R, T = utils.PnP(x,S,A)
    pc = torch.bmm(S,R.permute(0,2,1)) + T.unsqueeze(1)
    proj = torch.bmm(pc,A.permute(0,2,1))
    pimg_pred = proj / proj[:,:,-1].unsqueeze(-1)
    diff = x - pimg_pred[:,:,:2]
    loss = torch.norm(diff,dim=2).mean()
    return loss

def train(model_name:str):
    loader = dataloader.SyntheticLoader()
    center = torch.tensor([loader.w/2,loader.h/2,1])
    model = Model(center,gt=None)
    model.to_device(device)
    model.shape_opt = torch.optim.Adam(model.shape_net.parameters(),lr=1e-4)
    model.calib_opt = torch.optim.Adam(model.calib_net.parameters(),lr=1e-3)
    acc_steps = 50
    model.shape_opt.zero_grad()
    model.calib_opt.zero_grad()
    for epoch in itertools.count():
        for i in range(10000):
            batch = loader[i]
            x = batch['x_img'].float().to(device)
            k_true = batch['K_gt'].float().to(device)
            
            K = model.predict_intrinsic(x)
            f_loss = torch.norm(K - k_true, dim=(1,2)).mean() / acc_steps
            f_loss.backward()
            if (i+1) % acc_steps == 0:
                model.calib_opt.step()
                model.calib_opt.zero_grad()
            
            try:    
                S = model.get_shape(x)
                K_detached = K.detach()
                s_loss = compute_reprojection_loss(x.permute(0,2,1), S, K_detached) / acc_steps
            except:
                print(model.shape_net(x).mean(0))
                print("PnP fail, skip this video!")
                continue
            
            s_loss.backward()
            if (i+1) % acc_steps == 0:
                model.shape_opt.step()
                model.shape_opt.zero_grad()


            print(
                f"epoch: {epoch} | iter: {i} "
                f"| f_error: {f_loss.item()*acc_steps:.3f} "
                f"| fx: {K.mean(0)[0,0].item():.2f}/{k_true[0,0]:.2f} "
                f"| fy: {K.mean(0)[1,1].item():.2f}/{k_true[1,1]:.2f} "
                f"| cx: {K.mean(0)[0,2].item():.2f}/{k_true[0,2]:.2f} "
                f"| cy: {K.mean(0)[1,2].item():.2f}/{k_true[1,2]:.2f} "
                f"| S_err: {s_loss.item()*acc_steps:.3f}"
            )

        model.save(model_name, f"{epoch:02d}_")

if __name__ == '__main__':
    name = input("Model name: ")
    train(name)