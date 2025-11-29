# Camera Intrinsic Calibration with PointNet (Alternating Optimization)

A PyTorch project for estimating **camera intrinsic parameters** using a **PointNet-based neural network** combined with **alternating optimization** between a learned 3D shape representation and camera intrinsics.  
This method improves accuracy by iteratively refining both the predicted shape and the calibration parameters.

---

## 🔧 Key Features

- **PointNet2D architecture** (`PointNet2D.py`) for processing unordered 2D keypoints.
- **Alternating optimization** between:
  - Shape network (predicts 3D shape or latent shape parameters)
  - Calibration network (predicts the camera intrinsic matrix)
## Recommended Environment

This project was developed and tested using the following environment (Conda):

- **Python:** 3.11  
- **PyTorch:** 2.5.1 (CPU-only or CUDA version depending on your system)  
- **TorchVision:** 0.20.1  
- **TorchAudio:** 2.5.1  
- **NumPy:** 2.3.5  
- **SciPy:** 1.16.3  
- **OpenCV:** 4.11.0 (`opencv-python` from conda-forge)  
- **Matplotlib:** 3.10.8  
- **Kornia:** 0.8.2  
- **dlib:** 20.0.0  
- **PyYAML:** 6.0.3  
- **Sympy:** 1.14.0  
- **SMPLX (optional):** 0.1.28  

## Needed model
Download needed model and add it to Models folder: https://drive.google.com/drive/u/1/folders/1h2uCnsndk--n6u68Evt604oklvdxcUus
