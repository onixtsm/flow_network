#!/bin/python
import pandas as pd
import numpy as np
import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def save_predictions_to_csv(predictions, csv_save_path):
    output_np = predictions.detach().cpu().numpy()  # convert to numpy array
    output_np = output_np.squeeze()  # remove channel dimension new shape: (N, 32, 64)
    unseen_labels_flat = output_np.reshape(
        output_np.shape[0], -1)  # flatten to shape (N, 32*64)

    # create a dataframe for your submission, this will have the format
    # id, 0, 1, 2, 3, ..., 2047
    # 0, value, value, value, value, ..., value
    # 1, value, value, value, value, ..., value
    # 2, value, value, value, value, ..., value
    # etc
    df_pred = pd.DataFrame(unseen_labels_flat, columns=[
                           str(i) for i in range(unseen_labels_flat.shape[1])])
    df_pred.insert(0, 'id', range(unseen_labels_flat.shape[0]))

    # save your submission file
    df_pred.to_csv(csv_save_path, index=False)


def num_params(model):
    layers_store = []
    layer_idx = 0

    for param in model.parameters():
        if len(param.shape) > 1:  # weights
            layers_store.append(
                {"layer": layer_idx, "weight": param.numel(), "bias": 0})
            layer_idx += 1
        else:  # bias
            # this bias is from the last layer, so add to that dict
            layers_store[-1]["bias"] = param.numel()
    # now we can loop over the list of dicts and print the info
    for layer in layers_store:
        print(f"Layer {layer['layer']}: {
              layer['weight']} weights, {layer['bias']} biases")
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")


def homogenise_params(params: pd.DataFrame) -> pd.DataFrame:
    params["q"] = params["delta_A"] * \
        params["delta_p"] / (params["visc"] * params["L"]) * params["delta_A"]
    return params


def homogenise_labels(labels: np.array, params: pd.DataFrame):
    labels_h = labels.clone()
    for i, label in enumerate(labels):
        labels_h[i] = label / params['q'][i]
    return labels_h


def plot_comparison(input_img, prediction_img):
    input_img = input_img.detach().cpu() if isinstance(
        input_img, torch.Tensor) else input_img
    prediction_img = prediction_img.detach().cpu() if isinstance(
        prediction_img, torch.Tensor) else prediction_img

    # Reshape to 2D
    input_img = input_img.view(32, 64)
    prediction_img = prediction_img.view(32, 64)

    # Plot all three images side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input", "Prediction"]
    images = [input_img, prediction_img]
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, orientation="horizontal")
    plt.show()


class TensorData(Dataset):
    def __init__(self, input_tensor, device="cpu"):
        self.input = input_tensor.to(device)

    def __len__(self):
        return self.input.size()[0]

    def __getitem__(self, index):
        return self.input[index], self.labels[index]


def main():
    parser = argparse.ArgumentParser(description="NN for fence")
    parser.add_argument("-m", "--model", type=str,
                        help="Use model from file", required=True)
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot errors")

    parser.add_argument(
        "-n", "--num_params", action="store_true", help="Plot errors")
    args = parser.parse_args()

    # data = pd.read_csv("./kaggle_hidden_test_fences.csv")

    model = None
    m = torch.load(args.model)
    model = m['model']

    pred_error = np.min(m["loss_dict"]["test"])
    print(f"Predicted error {pred_error}")
    if args.plot:
        # plt.axvline(x=m['best_epoch'], color='r', linewidth=2)
        plt.plot(m["loss_dict"]["train"])
        plt.plot(m["loss_dict"]["test"])
        plt.yscale("log")
        plt.show()
        return
    if args.num_params:
        num_params(model)
        return

    params = pd.read_csv("./inputs/hidden_test_params.csv", index_col=False)
    params = homogenise_params(params)
    inputs = np.load("./inputs/hidden_test_inputs.npy")

    inputs = torch.from_numpy(inputs)
    inputs = inputs.unsqueeze(1)

    device = next(model.parameters()).device 
    inputs = inputs.to(device)
    pred = model(inputs)
    pred = homogenise_labels(pred, params)
    plot_comparison(inputs[3], pred[3])
    save_predictions_to_csv(
        pred, f"./kaggle/kaggle_{os.path.basename(args.model)}.csv")
    # df = pd.DataFrame(test_pred.detach().numpy(), columns=["prediction"])
    # df.to_csv(f"ce_kaggle_{os.path.basename(args.model)}.csv", index=True)

    return


class ViscosityNet2(nn.Module):
    def __init__(self, n, drop=0.5, pool=2, k_size=3, device="cpu") -> None:
        b = True
        super().__init__()
        layer_count = int(np.ceil(63.0 / k_size))
        layers = []
        channels_int = 1
        for x in range(layer_count):
            layers.append(nn.Conv2d(channels_int, n, k_size,
                          padding="same", bias=b))
            layers.append(nn.ReLU())
            if x % 2:
                layers.append(nn.Dropout2d(drop))
            channels_int = n
        layers.append(nn.Conv2d(n, 1, 1, padding="same", bias=b))

        self.conv = nn.Sequential(*layers)

    def mult(self, x: torch.Tensor, y):
        y = y.view(-1, 1, 32, 64)
        return x * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # y = self.dense(y)
        y = self.mult(y, x)
        return y  # x.view(-1, 1, 32, 64)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # k_size=3
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2) # check what this does
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for id in range(0, len(self.ups), 2):
            x = self.ups[id](x)
            skip = skip_connections[id//2]
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[id+1](concat_skip)

        return self.final_conv(x)

if __name__ == "__main__":
    torch.serialization.add_safe_globals(
        [UNET, nn.ModuleList, nn.ConvTranspose2d, DoubleConv, nn.BatchNorm2d, ViscosityNet2, nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Dropout2d, nn.Flatten, nn.Linear])
    main()
