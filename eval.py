from model import Model
import losses
import dataloader
import torch
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def eval(dataset_path: str, model_name: str, checkpoint:str):
    dataset = torch.load(dataset_path, weights_only=False)
    center = torch.tensor([1600/2,896/2,1])
    model = Model(center,gt=None)
    model.load(model_name, checkpoint)
    model.set_eval()
    model.to_device(device)
    total = 0
    efx, efy, ecx, ecy = 0.0, 0.0, 0.0, 0.0
    efx_total, efy_total, ecx_total, ecy_total = 0.0, 0.0, 0.0, 0.0    
    for data in dataset:
        total += 1
        sample = data["data"]
        K = sample["K_gt"].to(device)
        x_w = sample['x_w_gt'].to(device)
        x_cam = sample['x_cam_gt'].to(device)
        x_img = sample['x_img'].to(device)
        x_img_true = sample['x_img_gt'].to(device)
        # _, K_pred, _, _ = model.alternating_optimize(x_img, max_iter=7)
        # model.load(model_name, checkpoint)
        K_pred = model.predict_intrinsic(x_img)
        K_pred = K_pred.mean(0)
        error = torch.abs(K_pred - K)/K
        efx += error[0,0].item()
        efy += error[1,1].item()
        ecx += error[0,2].item()
        ecy += error[1,2].item()
        print(f"Index {data['global_index']}: efx: {error[0,0].item():.5f} | efy: {error[1,1]:.5f} | ecx: {error[0,2]:.5f} | ecy: {error[1,2]:.5f}")
        if data['sub_index'] == 19:
            efx_total += efx
            efy_total += efy
            ecx_total += ecx
            ecy_total += ecy
            efx /= 20
            efy /= 20
            ecx /= 20
            ecy /= 20
            print(f"Block {data['f_range']}: efx: {efx:.5f} | efy: {efy:.5f} | ecx: {ecx:.5f} | ecy: {ecy:.5f}\n\n")
            efx, efy, ecx, ecy = 0.0, 0.0, 0.0, 0.0
    print(f"Total efx: {efx_total/total} | efy: {efy_total/total:.5f} | ecx: {ecx_total/total:.5f} | ecy: {ecy_total/total:.5f}")
if __name__ == '__main__':
    center = torch.tensor([1600/2, 896/2, 1])
    model = Model(center)
    history_proj = []
    history_ecx = []
    history_ecy = []
    dataset = torch.load("dataset/test1.pt", weights_only=False)
    for i in range(1):
        total_samples = 0
        proj_loss = 0.0
        sum_ecx = 0.0
        sum_ecy = 0.0
        sum_reproj_error = 0.0

        model.load(r"pretrain", f"{0:02d}_")
        model.to_device(device)
        model.set_eval()

        for data in dataset:
            total_samples += 1
            sample = data['data']

            # get gt and inputs
            K_gt = sample['K_gt'].to(device)           # (3,3) or (1,3,3)
            x_img = sample['x_img'].to(device)         # expected shape (B,2,N) or (1,2,68) etc.
            S_pred, K_pred_batch, R_pred, T_pred = model.alternating_optimize(x_img, 3)
            model.load(r"train_eval_out\model", f"{i:02d}_")
            B = x_img.shape[0]
            # Get additional data for reprojection
            S = sample['x_w_gt'].to(device).unsqueeze(0).expand(B, -1, -1)              # 3D points
            R = sample['R_gt'].to(device)                 # Rotation
            T = sample['T_gt'].to(device)                 # Translation

            #K_pred_batch = model.predict_intrinsic(x_img) # expect (B,3,3)

            # --- compute reprojection loss ---
            # Convert x_img from (B,2,N) to (B,N,2) if needed
            if x_img.dim() == 3 and x_img.shape[1] == 2:
                x_img_transposed = x_img.permute(0, 2, 1)  # (B,2,N) -> (B,N,2)
            else:
                x_img_transposed = x_img
            
            # Compute reprojection error for each frame in batch
            reproj_error = losses.compute_reprojection_loss(
                x_img_transposed, 
                S_pred, 
                R_pred, 
                T_pred, 
                K_pred_batch
            )
            sum_reproj_error += reproj_error.item()

            # --- compute cx, cy errors ---
            # reduce predicted intrinsics over the batch frames to a single matrix (mean)
            # and compare with ground-truth K_gt (which may be single matrix)
            # K_pred_batch: (B,3,3) -> mean over B
            K_pred_mean = K_pred_batch.mean(dim=0)  # (3,3)

            # ensure K_gt is (3,3)
            if K_gt.dim() == 3 and K_gt.shape[0] == 1:
                K_gt_mat = K_gt.squeeze(0)
            else:
                K_gt_mat = K_gt
            print(K_gt)
            print(K_pred_mean)
            # move to device
            K_gt_mat = K_gt_mat.to(device)

            cx_gt = K_gt_mat[0, 2].item()
            cy_gt = K_gt_mat[1, 2].item()

            cx_pred = K_pred_mean[0, 2].item()
            cy_pred = K_pred_mean[1, 2].item()

            # relative errors (absolute relative)
            ecx_sample = abs(cx_pred - cx_gt) / (abs(cx_gt))
            ecy_sample = abs(cy_pred - cy_gt) / (abs(cy_gt))

            sum_ecx += ecx_sample
            sum_ecy += ecy_sample

            # end dataset loop

        # normalize by number of samples
        if total_samples == 0:
            avg_ecx = 0.0
            avg_ecy = 0.0
            avg_reproj = 0.0
        else:
            avg_ecx = sum_ecx / total_samples
            avg_ecy = sum_ecy / total_samples
            avg_reproj = sum_reproj_error / total_samples

        print(f"Epoch ckpt {i:02d} | ecx={avg_ecx:.6f} | ecy={avg_ecy:.6f} | reproj={avg_reproj:.6f}")

        history_ecx.append(avg_ecx)
        history_ecy.append(avg_ecy)
        history_proj.append(avg_reproj)

    # plot ecx & ecy together
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(history_ecx)), history_ecx, label='ecx (rel)', marker='o')
    plt.plot(range(len(history_ecy)), history_ecy, label='ecy (rel)', marker='s')
    plt.xlabel('Checkpoint (epoch)')
    plt.ylabel('Relative error')
    plt.title('Relative errors for cx and cy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('5x5/cx_cy_relative_error.png')
    plt.close()

    # plot reprojection error
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(history_proj)), history_proj, label='Reprojection Error', marker='o', color='green')
    plt.xlabel('Checkpoint (epoch)')
    plt.ylabel('Reprojection Error')
    plt.title('Reprojection Error over checkpoints')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('5x5/reprojection_error.png')
    plt.close()