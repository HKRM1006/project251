


import argparse
import itertools
import torch
import matplotlib.pyplot as plt
import os
from model import Model
import dataloader
import utils
import losses
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self, center, device, train_loader=None):
        self.device = device
        self.loader = train_loader if train_loader is not None else dataloader.SyntheticLoader()
        self.center = center

    def train_and_eval(self, out_dir, val_dataset_path=None, epochs=40, acc_steps=50, iters_per_epoch=500):
        os.makedirs(out_dir, exist_ok=True)
        model = Model(self.center, gt=None)
        model.to_device(self.device)
        model.shape_opt = torch.optim.Adam(model.shape_net.parameters(), lr=1e-4)
        model.calib_opt = torch.optim.Adam(model.calib_net.parameters(), lr=1e-5)
        model.shape_opt.zero_grad()
        model.calib_opt.zero_grad()

        train_losses = []     
        train_s_losses = []   
        val_losses = []       
        val_s_losses = []     
        ef_totals = []

        for epoch in range(epochs):
            print(f"==== Start epoch {epoch} ====")
            running_f_loss = 0.0
            running_s_loss = 0.0
            steps_counted = 0
            
            for i in range(iters_per_epoch):
                batch = self.loader[i]
                x = batch['x_img'].float().to(self.device)
                B = x.shape[0]
                k_gt = batch['K_gt'].float().to(self.device)
                R_gt = batch['R_gt'].float().to(self.device)
                T_gt = batch['T_gt'].float().to(self.device)
                S_gt = batch['x_w_gt'].float().to(device)
                
                K = model.predict_intrinsic(x)
                betas = model.shape_net(x)
                S_pred = self.loader.get_shape(betas)
                
                f_loss = torch.norm(K.mean(0) - k_gt) 
                (f_loss / acc_steps).backward()
                if (i + 1) % acc_steps == 0:
                    model.calib_opt.step()
                    model.calib_opt.zero_grad()
                
                s_loss = losses.shape_loss(
                    betas, x, S_pred, S_gt, 
                    R_gt, T_gt, k_gt, epoch
                )
                (s_loss / acc_steps).backward()
                if (i + 1) % acc_steps == 0:
                    model.shape_opt.step()
                    model.shape_opt.zero_grad()
                    

                running_f_loss += f_loss.item()
                running_s_loss += s_loss.item()
                steps_counted += 1

            
            avg_f_loss = (running_f_loss / max(1, steps_counted))
            avg_s_loss = (running_s_loss / max(1, steps_counted))
            train_losses.append(avg_f_loss)
            train_s_losses.append(avg_s_loss)
            print(f"Epoch {epoch} Training: f_loss={avg_f_loss:.6f} | s_loss={avg_s_loss:.6f}")
            
            model.save(os.path.join(out_dir, 'model'), f"{epoch:02d}_")

            
            if val_dataset_path is not None:
                ef_total, val_f_loss, val_s_loss = self.eval_on_dataset(model, val_dataset_path, epoch)
                val_losses.append(val_f_loss)
                val_s_losses.append(val_s_loss)
                ef_totals.append(ef_total)
                print(f"Epoch {epoch} VALIDATION: val_f_loss={val_f_loss:.6f} | val_s_loss={val_s_loss:.6f} | ef_total={ef_total:.6f}")

            
            metrics = {
                'train_f_loss': train_losses,
                'train_s_loss': train_s_losses,
                'val_f_loss': val_losses,
                'val_s_loss': val_s_losses,
                'ef_total': ef_totals
            }
            torch.save(metrics, os.path.join(out_dir, 'metrics.pt'))

            
            self.plot_metrics(train_losses, train_s_losses, val_losses, val_s_losses, ef_totals, out_dir)

        print("Training finished. Final plots saved to:", out_dir)

    def eval_on_dataset(self, model, dataset_path, epoch):
        dataset = torch.load(dataset_path, weights_only=False)
        model.set_eval()
        model.to_device(self.device)

        total_samples = 0
        sum_rel_err = 0.0
        sum_val_f_loss = 0.0
        sum_val_s_loss = 0.0

        with torch.no_grad():
            for data in dataset:
                total_samples += 1
                sample = data['data']
                K_gt = sample['K_gt'].to(self.device)
                x_img = sample['x_img'].to(self.device)

                K_pred = model.predict_intrinsic(x_img)                
                K_pred = K_pred.mean(0)
                rel = torch.abs(K_pred - K_gt) / (K_gt + 1e-8)
                
                rel_vals = torch.tensor([rel[0, 0].item(), rel[1, 1].item(), rel[0, 2].item(), rel[1, 2].item()])
                sum_rel_err += rel_vals.mean().item()

                
                try:
                    if x_img.dim() == 3:
                        K_batch = model.predict_intrinsic(x_img)
                        k_true = K_gt
                        val_f = torch.norm(K_batch - k_true, dim=(1, 2)).mean().item()
                    else:
                        K_batch = K_pred.unsqueeze(0)
                        k_true = K_gt.unsqueeze(0)
                        val_f = torch.norm(K_batch - k_true, dim=(1, 2)).mean().item()
                except Exception:
                    val_f = 0.0

                sum_val_f_loss += val_f

                try:
                    if 'alpha_gt' in sample:
                        alpha_gt = sample['alpha_gt'].to(self.device)
                        alpha_pred = model.shape_net(x_img)
                        S_pred = self.loader.get_shape(alpha_pred)
                        k_gt = sample['K_gt'].float().to(self.device)
                        R_gt = sample['R_gt'].float().to(self.device)
                        T_gt = sample['T_gt'].float().to(self.device)
                        S_gt = sample['x_w_gt'].float().to(device)
                        val_s = losses.shape_loss(
                            alpha_pred, x_img, S_pred, S_gt, 
                            R_gt, T_gt, k_gt, epoch
                        )
                        sum_val_s_loss += val_s
                except Exception:
                    pass

        ef_total = sum_rel_err / max(1, total_samples)
        avg_val_f_loss = sum_val_f_loss / max(1, total_samples)
        avg_val_s_loss = sum_val_s_loss / max(1, total_samples)
        return ef_total, avg_val_f_loss, avg_val_s_loss

    def plot_metrics(self, train_f_losses, train_s_losses, val_f_losses, val_s_losses, ef_totals, out_dir):
        epochs = list(range(len(train_f_losses)))

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_f_losses, label='train_loss', marker='o')
        if len(val_f_losses) > 0:
            plt.plot(epochs[:len(val_f_losses)], val_f_losses, label='eval_loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('F Loss')
        plt.title('Intrinsic Loss (F Loss)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'f_loss_curve.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_s_losses, label='train_loss', marker='o')
        if len(val_s_losses) > 0:
            plt.plot(epochs[:len(val_s_losses)], val_s_losses, label='eval_loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('S Loss (Shape)')
        plt.title('Shape Loss (S Loss)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 's_loss_curve.png'))
        plt.close()

        if len(ef_totals) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs[:len(ef_totals)], ef_totals, label='ef', marker='o', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Relative Intrinsic Error')
            plt.title('Relative Intrinsic Error')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'ef_total_curve.png'))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dataset', type=str, default='dataset/test1.pt', help='path to validation dataset (torch file)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--out', type=str, default='./train_eval_out')
    parser.add_argument('--iters_per_epoch', type=int, default=500)
    parser.add_argument('--acc_steps', type=int, default=50)
    args = parser.parse_args()

    center = torch.tensor([1600/2, 896/2, 1])
    train_loader = dataloader.SyntheticLoader()

    trainer = Trainer(center, device, train_loader)
    trainer.train_and_eval(out_dir=args.out, val_dataset_path=args.val_dataset, epochs=args.epochs, acc_steps=args.acc_steps, iters_per_epoch=args.iters_per_epoch)
    
    

