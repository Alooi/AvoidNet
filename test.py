import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torch import nn, optim
import matplotlib.pyplot as plt
from dataset import SUIM
import time
from avoid_net import get_model
import argparse


def test(batch_size, num_avg, arc, run_name, use_gpu=True):

    model = get_model(arc)
    model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth"))

    # Prepare your own dataset
    dataset = SUIM("/media/ali/New Volume/Datasets/TEST")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    model.to(device)
    model.eval()

    # load images and masks
    images, masks = next(iter(dataloader))
    images = images.to(device)

    # forward pass
    tic = time.time()
    for i in range(num_avg):
        outputs = model(images)
        i += 1
    toc = time.time()
    print(f"-- Inference time: ", ((toc - tic) / num_avg) / batch_size)

    # forward pass
    outputs = model(images)

    # move the output tensors to cpu for visualization
    outputs = outputs.detach().cpu()
    images = images.cpu()

    # denormalize the images and masks
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    # calculate mask predictions vs ground truth
    MSE = nn.MSELoss()
    MSE_loss = MSE(outputs, masks)
    print(f"-- MSE loss: {MSE_loss}")

    # # show the first batch of images and masks and predictions side by side on the same figure

    # show all of the batch images and masks on the same figure
    fig, ax = plt.subplots(batch_size, 3, figsize=(15, 15))
    for i in range(batch_size):
        ax[i, 0].imshow(images[i].permute(1, 2, 0))
        ax[i, 0].set_title("Image")
        ax[i, 1].imshow(masks[i].permute(1, 2, 0))
        ax[i, 1].set_title("Mask")
        ax[i, 2].imshow(outputs[i].permute(1, 2, 0))
        ax[i, 2].set_title("Prediction")
    plt.show()


if __name__ == "__main__":
    # get args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_average",
        type=int,
        default=10,
        help="Number of average runs for speed calculation",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="test",
        help="Name of the run",
    )
    parser.add_argument(
        "--arc",
        type=str,
        default="ImageReducer",
        help="Name of the model architecture",
    )
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Use GPU for testing",
    )
    args = parser.parse_args()

    # test the model
    test(args.batch_size, args.num_average, args.arc, args.run_name, args.use_gpu)
