import cv2
import numpy as np
import time
from avoid_net import get_model
import argparse
import torch
from dataset import SUIM, SUIM_grayscale
from PIL import Image
import numpy as np
from draw_obsticle import draw_red_squares


def run_model(arc, run_name, source, video_path=None, use_gpu=False, save_video=False):
    model = get_model(arc)
    model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Running on: {device}...")
    model.to(device).eval()

    dataset = SUIM_grayscale("/media/ali/New Volume/Datasets/TEST")
    image_transform = dataset.get_transform()
    mask_transform = dataset.get_mask_transform()

    # read from source using opencv
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video":
        cap = cv2.VideoCapture(video_path)
        
    if save_video:
        out = cv2.VideoWriter(
            "results/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # prepare the frame for inference
        frame_tensor = Image.fromarray(frame)
        frame_tensor = image_transform(frame_tensor).to(device).unsqueeze(0)
        outputs = model(frame_tensor)

        # move the output tensors to cpu for visualization
        outputs = outputs.detach().cpu()
        outputs = outputs[0].permute(1, 2, 0)
        outputs = np.array(outputs)  # Convert to numpy array
        # show the output
        frame = draw_red_squares(frame, outputs, 0.5)
        # cv2.imshow("frame", frame)
        if save_video:
            frame = cv2.resize(frame, (640, 480))
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            out.release()
            break
    out.release()


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
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Use GPU for inference",
    )
    parser.add_argument(
        "--save_video",
        type=bool,
        default=False,
        help="Save the output video",
    )
    
    args = parser.parse_args()

    run_model(args.arc, args.run_name, args.source, args.video_path, args.use_gpu, args.save_video)

# example usage:
# python run_model.py --arc ImageReducer_bounded_grayscale --run_name run_2 --source video --video_path samples/underwater_drone_sample.mp4 --use_gpu True --save_video True