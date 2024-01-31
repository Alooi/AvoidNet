import cv2
import numpy as np
import time
from avoid_net import get_model
import argparse
import torch
from dataset import SUIM
from PIL import Image
import numpy as np


def run_model(arc, run_name, source, video_path=None):
    model = get_model(arc)
    model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}...")
    model.to(device).eval()

    dataset = SUIM("/media/ali/New Volume/Datasets/TEST")
    image_transform = dataset.get_transform()
    mask_transform = dataset.get_mask_transform()

    # read from source using opencv
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video":
        cap = cv2.VideoCapture(video_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # prepare the frame for inference
        frame_tensor = Image.fromarray(frame)
        frame_tensor = image_transform(frame_tensor).to(device).unsqueeze(0)
        outputs = model(frame_tensor)
        print(outputs)
        # move the output tensors to cpu for visualization
        outputs = outputs.detach().cpu().squeeze(0)
        outputs = outputs.permute(1, 2, 0)
        outputs = np.array(outputs)  # Convert to numpy array
        # show the output
        cv2.imshow("frame", frame)
        heatmap = cv2.applyColorMap(np.uint8(outputs * 255), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        cv2.imshow("heatmap", heatmap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc",
        type=str,
        default="unet",
        help="Architecture to be used for training",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="unet_1",
        help="Name of the run to be used for training",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Source to be used for inference",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="Path to the video to be used for inference",
    )
    args = parser.parse_args()

    run_model(args.arc, args.run_name, args.source, args.video_path)
