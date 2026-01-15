import utils
import losses
import dataloader
import torch
import numpy as np
from PointNet2D import PointNet
import os
import matplotlib.pyplot as plt
class Model():
    def __init__(self, center = None, shape_net = None, calib_net = None, gt = None, name:str = None):
        #Neural Net
        if shape_net is None or calib_net is None:
            self.new_model()
        else:
            self.shape_net = shape_net
            self.calib_net = calib_net
        self.center = center
        self.gt = gt
        self.w, self.h = center[0]*2, center[1]*2

        #Dataloader
        self.loader = dataloader.SyntheticLoader()

        #optim
        self.shape_opt = torch.optim.Adam(self.shape_net.parameters(),lr=5e-1)
        self.calib_opt = torch.optim.Adam(self.calib_net.parameters(),lr=1e-4)
    
    def new_model(self):
        self.shape_net = PointNet(10)
        self.calib_net = PointNet(4)
    
    def to_device(self, device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calib_net = self.calib_net.to(device)
        self.shape_net = self.shape_net.to(device)

    def set_eval(self):
        self.shape_net.eval()
        self.calib_net.eval()
    
    def set_train(self):
        self.shape_net.train()
        self.calib_net.train()
    
    def get_shape(self, frames):
        b = frames.shape[0]
        betas = self.shape_net(frames).mean(0).unsqueeze(0).repeat(b,1)
        S = self.loader.get_shape(betas)
        return S

    def predict_intrinsic(self, frames):
        k = self.calib_net(frames)
        return utils.create_K(k)
    
    def shape_optimize(self, frames, K, max_iter=5):
        for i in range(max_iter):
            self.shape_opt.zero_grad()

            S = self.get_shape(frames)
            Xc, R, T = utils.PnP(frames.permute(0,2,1),S,K)
            error2d = torch.log(losses.projection_loss(frames,S,R,T,K).mean())
            error_p = losses.principal_loss(K,self.center.unsqueeze(0)).mean()

            loss = error2d
            loss.backward()
            self.shape_opt.step()

            fx = torch.mean(K[:,0,0])
            fy = torch.mean(K[:,1,1])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])
            pred = {'iter': i, 'error': loss, 'e_pr': error_p,'fx': fx, 'fy': fy,
                    'e2d': error2d,'px': px, 'py': py}

        pred['S'] = S.detach()
        return pred
    
    def calib_optimize(self,frames,S,max_iter=5):
        b = frames.shape[0]
        for i in range(max_iter):
            self.calib_opt.zero_grad()
            K = self.predict_intrinsic(frames).mean(0).repeat(b,1,1)
            Xc, R, T = utils.PnP(frames.permute(0,2,1),S,K)
            error2d = torch.log(losses.projection_loss(frames,S,R,T,K).mean())
            error_p = losses.principal_loss(K,self.center.unsqueeze(0)).mean()


            loss = error2d + error_p*1e-2
            loss.backward()
            self.calib_opt.step()

            fx = torch.mean(K[:,0,0])
            fy = torch.mean(K[:,1,1])
            px = torch.mean(K[:,0,2])
            py = torch.mean(K[:,1,2])
            pred = {'iter': i, 'error': loss, 'e_pr': error_p,'fx': fx, 'fy': fy,
                    'e2d': error2d,'px': px, 'py': py}


        pred['K'] = K.detach()
        return pred
    
    def alternating_optimize(self, x, max_iter=5):
        b = x.shape[0]
        K = self.predict_intrinsic(x).detach()
        S = self.get_shape(x).detach()
        pred = {'S': S, 'K': K}

        for _ in range(max_iter):
            pred = self.calib_optimize(x, pred['S'].mean(0).unsqueeze(0).repeat(b,1,1),max_iter=7)
            pred = self.shape_optimize(x, pred['K'].mean(0).unsqueeze(0).repeat(b,1,1),max_iter=7)

        S = self.get_shape(x).mean(0).unsqueeze(0).repeat(b,1,1)
        K = self.predict_intrinsic(x).mean(0).unsqueeze(0).repeat(b,1,1)
        _, R, T = utils.PnP(x.permute(0,2,1),S,K)
        return S, K, R, T
    
    def save(self, dir: str = "", token = ""):
        path = os.path.join("Models", dir)
        os.makedirs(path, exist_ok=True)
        torch.save(self.calib_net.state_dict(), os.path.join(path, token + "calib_net.pt"))
        torch.save(self.shape_net.state_dict(), os.path.join(path, token + "shape_net.pt"))

    def load(self, dir: str = "", token = ""):
        path = os.path.join("Models", dir)
        self.calib_net.load_state_dict(torch.load(os.path.join(path, token + "calib_net.pt"), weights_only=True))
        self.shape_net.load_state_dict(torch.load(os.path.join(path, token + "shape_net.pt"), weights_only=True))
        self.shape_opt = torch.optim.Adam(self.shape_net.parameters(),lr=1e-5)
        self.calib_opt = torch.optim.Adam(self.calib_net.parameters(),lr=1e-5)

