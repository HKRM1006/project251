from model import Model
import torch
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
    dataset_path = input("Dataset path: ")
    model_name = input("Model name: ")
    checkpoint = int(input("Checkpoint: "))
    eval(dataset_path, model_name, f"{checkpoint:02d}_")