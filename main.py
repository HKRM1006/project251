import cv2
import dlib
import numpy as np
import torch
import os
from model import Model
from typing import Dict, List, Tuple
import argparse
import json
def extract_face_landmarks(
    video_path: str,
    predictor_path: str,
    frame_stride: int = 2,
    max_frames: int = 100,
    device: str = "cpu"
) -> Tuple[torch.Tensor, int, int]:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    landmarks = []
    frame_idx, count = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 0)
        if len(faces) == 0:
            frame_idx += 1
            continue

        rect = max(faces, key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
        shape = predictor(gray, rect)

        pts = np.array(
            [[shape.part(i).x, H - shape.part(i).y] for i in range(shape.num_parts)],
            dtype=np.float32
        )

        landmarks.append(pts)
        count += 1
        if count >= max_frames:
            break

        frame_idx += 1

    cap.release()

    if len(landmarks) == 0:
        return torch.empty(0, 0, 2), W, H

    pts = torch.from_numpy(np.stack(landmarks)).to(device)
    return pts, W, H


def calibrate_intrinsic(
    video_path: str,
    model_name: str,
    token: str,
    predictor_path: str,
    device: str = "cpu"
) -> np.ndarray:

    pts, W, H = extract_face_landmarks(
        video_path, predictor_path, device=device
    )

    if pts.numel() == 0:
        raise RuntimeError(f"No face detected in {video_path}")

    F, N, _ = pts.shape
    pts = pts.reshape(F, 2, N).float().to(device)

    model = Model(torch.tensor([W / 2, H / 2, 1.0]))
    model.load(model_name, token)
    model.set_eval()

    _, pred, _, _ = model.alternating_optimize(pts)

    pred = pred.mean(0).detach().cpu().numpy()

    return pred


def calibrate_intrinsic_from_folder(
    folder_path: str,
    model_name: str,
    token: str,
    predictor_path: str,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:

    videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    if not videos:
        raise ValueError("No video found")

    K_results = {}

    for video in videos:
        video_path = os.path.join(folder_path, video)
        print(f"Calibrating: {video}")

        try:
            K = calibrate_intrinsic(
                video_path,
                model_name,
                token,
                predictor_path,
                device
            )
            K_results[video] = K
        except Exception as e:
            print(f"  Failed: {e}")

    return K_results

def save_intrinsics_to_json(K_results: dict, output_path: str):
    json_dict = {
        name: K.tolist() for name, K in K_results.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2)

    print(f"[OK] Saved intrinsic results to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Face-based intrinsic calibration from videos"
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Folder containing input videos (.mp4)"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )

    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Checkpoint token, e.g. 02_"
    )

    parser.add_argument(
        "--predictor",
        type=str,
        default="Models/shape_predictor_68_face_landmarks.dat",
        help="Dlib facial landmark predictor"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="intrinsic_results.json",
        help="Output JS file"
    )

    args = parser.parse_args()

    K_results = calibrate_intrinsic_from_folder(
        folder_path=args.video_dir,
        model_name=args.model,
        token=args.token,
        predictor_path=args.predictor,
        device=args.device
    )

    save_intrinsics_to_json(K_results, args.output)

if __name__ == "__main__":
    main()