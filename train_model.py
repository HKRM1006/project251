import torch, torch.nn as nn
import torch.nn.functional as F
import os
import data_loader as data
from torch.utils.data import DataLoader, TensorDataset

class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )


    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)            
        
        x_global = torch.max(x, dim=2)[0]       
                     
        return x_global

class PointNetK(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, landmarks_2d):
        B = landmarks_2d.size(0)
        feat = self.encoder(landmarks_2d)
        out = self.fc(feat)
        return out

class PointNetShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, landmarks_2d):
        B = landmarks_2d.size(0)
        feat = self.encoder(landmarks_2d)
        out = self.fc(feat)
        return out
    

def init_model(name: str):
    intrinsic_model = PointNetK()
    shape_model = PointNetShape()
    if not os.path.exists("Models"):
        os.makedirs("Models")
    torch.save(intrinsic_model.state_dict(), "Models/" + name + "Intrinsic" + ".pth")
    torch.save(shape_model.state_dict(), "Models/" + name + "Shape" + ".pth")

def fit(shape_model, intrinsic_model, dataloader, criterion_K, criterion_beta,
        opt_K, opt_beta, device, epochs=10):
    
    shape_model.to(device)
    intrinsic_model.to(device)

    for epoch in range(1, epochs+1):
        shape_model.train()
        intrinsic_model.train()

        lossK_sum, lossB_sum, total = 0, 0, 0

        for xb, yK, yBeta in dataloader:
            xb = xb.to(device)
            yK = yK.to(device)
            yBeta = yBeta.to(device)

            # ====== Train intrinsic model ======
            opt_K.zero_grad()
            predK = intrinsic_model(xb)
            lossK = criterion_K(predK, yK)
            lossK.backward()
            opt_K.step()

            # ====== Train shape model ======
            opt_beta.zero_grad()
            predB = shape_model(xb)
            lossB = criterion_beta(predB, yBeta)
            lossB.backward()
            opt_beta.step()

            lossK_sum += lossK.item() * xb.size(0)
            lossB_sum += lossB.item() * xb.size(0)
            total += xb.size(0)

        print(f"Epoch {epoch:03d} | K_loss={lossK_sum/total:.6f} | Beta_loss={lossB_sum/total:.6f}")

def evaluate(shape_model, intrinsic_model, dataloader, criterion_K, criterion_beta, device):
    shape_model.to(device)
    intrinsic_model.to(device)
    shape_model.eval()
    intrinsic_model.eval()
    with torch.no_grad():
        lossK_sum, lossB_sum, total = 0, 0, 0
        for xb, yK, yBeta in dataloader:
            xb = xb.to(device)
            yK = yK.to(device)
            yBeta = yBeta.to(device)

            # ====== Eval intrinsic model ======

            predK = intrinsic_model(xb)
            lossK = criterion_K(predK, yK)


            # ====== Eval shape model ======
            predB = shape_model(xb)
            lossB = criterion_beta(predB, yBeta)

            lossK_sum += lossK.item() * xb.size(0)
            lossB_sum += lossB.item() * xb.size(0)
            total += xb.size(0)

        print(f"Evaluate | K_loss={lossK_sum/total:.6f} | Beta_loss={lossB_sum/total:.6f}")
def predict(intrinsic_model: nn.Module, data, device):
    intrinsic_model.eval()
    intrinsic_model.to(device)
    return intrinsic_model(data)

def intrinsic_loss(pred, target):
    return F.mse_loss(pred, target)

def beta_loss(pred, target):
    return F.mse_loss(pred, target)

def test(model, loader, device):
    model.eval()
    total_fx_err = 0.0
    total_fy_err = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yK in loader:
            xb = xb.to(device)
            yK = yK.to(device)
            pred = model(xb)

            fx_hat = pred[:,0]
            fy_hat = pred[:,1]
            fx = yK[:,0]
            fy = yK[:,1]

            fx_err_sum += (torch.abs(fx_hat - fx)/(fx+1e-8)).sum().item()
            fy_err_sum += (torch.abs(fy_hat - fy)/(fy+1e-8)).sum().item()
            n += xb.size(0)
    print(f"efx: {total_fx_err / total_samples:.6f}   efy: {total_fy_err / total_samples:.6f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = input("Model name: ")
    # Load dataset
    x, yK, _, _, betas = data.load_dataset_pt(input("Train dataset: "))
    xT, yT, _, _, bT = data.load_dataset_pt(input("Test dataset: "))

    # Split
    x_train, yK_train, x_val, yK_val, b_train, b_val = data.split_dataset(x, yK, betas, val_ratio=0.1)

    # Loader
    train_loader, val_loader = data.build_dataloaders(
        x_train, yK_train, b_train,
        x_val, yK_val, b_val,
        batch_size=64
    )

    # Models
    intrinsic_model = PointNetK()
    shape_model = PointNetShape()

    # Optimizers
    optK = torch.optim.AdamW(intrinsic_model.parameters(), lr=1e-4)
    optB = torch.optim.AdamW(shape_model.parameters(), lr=1e-4)

    # Train
    fit(shape_model, intrinsic_model,
        train_loader,
        intrinsic_loss, beta_loss,
        optK, optB,
        device, epochs=50)

    # Test intrinsic model
    test_loader = DataLoader(TensorDataset(xT, yT), batch_size=16)
    test(intrinsic_model, test_loader, device)

    torch.save(intrinsic_model.state_dict(), "Models/" + name + "Intrinsic" + ".pth")
    torch.save(shape_model.state_dict(), "Models/" + name + "Shape" + ".pth")