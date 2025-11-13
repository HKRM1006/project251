import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import os
import process_data as data
from torch.utils.data import DataLoader, TensorDataset
class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
    def forward(self, x):
        B = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        I = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        I = I.repeat(B, 1)
        x = x + I
        x = x.view(-1, self.k, self.k)
        return x
class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet1 = TNet(2)
        self.mlp1 = nn.Sequential(nn.Conv1d(2, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1),  nn.BatchNorm1d(64),   nn.ReLU())
        self.tnet2 = TNet(64)
        self.mlp3 = nn.Sequential(nn.Conv1d(64, 64, 1),  nn.BatchNorm1d(64),   nn.ReLU())
        self.mlp4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128),  nn.ReLU())
        self.mlp5 = nn.Sequential(nn.Conv1d(128, 1024,1),nn.BatchNorm1d(1024), nn.ReLU())
        self.fc1 = nn.Linear(1024, 512);  
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256);   
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 4)
    def forward(self, x):
        B = x.size(0)
        x = x.transpose(1, 2)
        T1 = self.tnet1(x)
        x = torch.bmm(T1, x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        T2 = self.tnet2(x)
        x = torch.bmm(T2, x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x) 
        return out
def init_model(name: str):
    model = PointNet()
    if not os.path.exists("Models"):
        os.makedirs("Models")
    torch.save(model.state_dict(), "Models/" + name + ".pth")

def fit(model, dataloader, criterion, optimizer, device,
        epochs=10, val_loader=None, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau =None):
    """
    scheduler:
        - For StepLR / CosineAnnealingLR: scheduler.step() each epoch
        - For ReduceLROnPlateau: scheduler.step(val_loss)
    """
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        running_loss = 0.0
        total = 0

        for xb, yb in dataloader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        avg_train = running_loss / total

        # -------------------- VALIDATE --------------------
        avg_val = None
        if val_loader is not None:
            model.eval()
            vloss = 0.0
            vtotal = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device, dtype=torch.float32)
                    vy = vy.to(device, dtype=torch.float32)

                    preds = model(vx)
                    loss = criterion(preds, vy)
                    vloss += loss.item() * vx.size(0)
                    vtotal += vx.size(0)
            avg_val = vloss / vtotal
            print(f"Epoch {epoch:03d} | train={avg_train:.6f} | val={avg_val:.6f}")
        else:
            print(f"Epoch {epoch:03d} | train={avg_train:.6f}")

        # -------------------- LR SCHEDULER --------------------
        if scheduler is not None:
            scheduler.step(avg_val if avg_val is not None else avg_train)

def predict(model: nn.Module, data, device):
    model.eval()
    model.to(device)
    return model(data)
def intrinsic_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()
def test(model, val_loader, device):
    model.eval()
    model.to(device)
    total_fx_err = 0.0
    total_fy_err = 0.0
    total_samples = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device, dtype=torch.float32)
            vy = vy.to(device, dtype=torch.float32)
            preds = model(vx)
            fx_hat = preds[:, 0]
            fy_hat = preds[:, 1]
            fx_true = vy[:, 0]
            fy_true = vy[:, 1]
            fx_err = torch.abs(fx_hat - fx_true) / (fx_true + 1e-8)
            fy_err = torch.abs(fy_hat - fy_true) / (fy_true + 1e-8)
            total_fx_err += fx_err.sum().item()
            total_fy_err += fy_err.sum().item()
            total_samples += vx.size(0)
    print(f"efx: {total_fx_err / total_samples:.6f}   efy: {total_fy_err / total_samples:.6f}")

def main():
    #init_model("testModel")
    device = torch.device("cpu")
    model = PointNet()
    state = torch.load("Models/testModel.pth", map_location=device)
    model.load_state_dict(state)
    
    
    # x, y = data.load_dataset_pt("dataset/synthetic_face_dataset.pt")
    # x_train, y_train, x_val, y_val = data.split_dataset(x, y, val_ratio=0.1)
    # train_loader, val_loader = data.build_dataloaders(
    #     x_train, y_train, x_val, y_val,
    #     batch_size = 32
    # )
    # criterion = intrinsic_loss
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # fit(model, train_loader, criterion, optimizer, device, 30, val_loader, scheduler)
    # torch.save(model.state_dict(), "Models/testModel.pth")


    x, y = data.load_dataset_pt("dataset/synthetic_face_dataset_test.pt")
    test_dataset = TensorDataset(x, y)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test(model, test_loader, device)
if __name__ == "__main__":
    main()