# Camera Intrinsic Calibration with PointNet (Alternating Optimization)

A PyTorch project for estimating **camera intrinsic parameters** using a **PointNet-based neural network** combined with **alternating optimization** between a learned 3D shape representation and camera intrinsics.  
This method improves accuracy by iteratively refining both the predicted shape and the calibration parameters.

---

## 🔧 Key Features

- **PointNet2D architecture** (`PointNet2D.py`) for processing unordered 2D keypoints.
- **Alternating optimization** between:
  - Shape network (predicts 3D shape or latent shape parameters)
  - Calibration network (predicts the camera intrinsic matrix)

## Needed model
Download needed model and add it to Models folder: https://drive.google.com/drive/u/1/folders/1h2uCnsndk--n6u68Evt604oklvdxcUus
