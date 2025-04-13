#!/bin/python
import argparse
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import datetime


def plot_image(img, title=None, ax=None):
    img = img.detach().cpu() if isinstance(
        img, torch.Tensor) else img
    img = img.view(32, 64)
    plt.imshow(img)
    plt.colorbar()
    plt.axis('off')
    plt.show()
    return ax


def plot_comparison(input_img, label_img, prediction_img):
    input_img = input_img.detach().cpu() if isinstance(
        input_img, torch.Tensor) else input_img
    label_img = label_img.detach().cpu() if isinstance(
        label_img, torch.Tensor) else label_img
    prediction_img = prediction_img.detach().cpu() if isinstance(
        prediction_img, torch.Tensor) else prediction_img

    # Reshape to 2D
    input_img = input_img.view(32, 64)
    label_img = label_img.view(32, 64)
    prediction_img = prediction_img.view(32, 64)

    # Plot all three images side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input", "Label", "Prediction"]
    images = [input_img, label_img, prediction_img]
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, orientation="horizontal")
    plt.show()


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
    def __init__(self, in_channels=3, out_channels=1, features = [1, 2, 4, 8]):
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


class ViscosityNet2(nn.Module):
    def __init__(self, n, drop=0.5, pool=2, k_size=3) -> None:
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


def parse_arguments() -> argparse.Namespace:
    settings = {
        "epochs": 42,
        "lr": 1e-3,
        "batch": 256,
        "device": None,
        "split": 80,
        "report": 10,
        "neurons": 64,
        "seed": None,
        "kernel_size": 3
    }
    parser = argparse.ArgumentParser(description="NN for fence")
    parser.add_argument("-e", "--epochs", type=int, help="A number of epochs")
    parser.add_argument("--cpu", action="store_true",
                        dest="device", help="Force cpu usage")
    parser.add_argument("-m", "--model", type=str, help="Use model from file")
    parser.add_argument("-n", "--neurons", type=int, help="Neuron count")
    parser.add_argument("-b", "--batch", type=int, help="Batch size")
    parser.add_argument("-r", "--report", type=int,
                        help="Report progress each n epochs")
    parser.add_argument("-lr", "--lr", type=float, help="Set learning rate")
    parser.add_argument("--split", type=int,
                        help="Set percent of split for data")
    parser.add_argument("--seed", type=int, help="Set seed")
    parser.add_argument("--kernel_size", "-k", type=int, help="Set seed")

    parser.set_defaults(**settings)
    settings = parser.parse_args()
    settings.device = "cpu" if settings.device or not torch.cuda.is_available() else "cuda:0"
    return settings


class TensorData(Dataset):
    def __init__(self, input_tensor, label_tensor, device="cpu"):
        self.input = input_tensor.to(device)
        self.labels = label_tensor.to(device)

    def __len__(self):
        return self.input.size()[0]

    def __getitem__(self, index):
        return self.input[index], self.labels[index]


def split_data(data, percent=80):
    n_train = int(len(data) * percent / 100)
    n_test = len(data) - n_train
    return torch.utils.data.random_split(data, [n_train, n_test])


def homogenise_params(params: pd.DataFrame) -> pd.DataFrame:
    params["q"] = params["delta_A"] * \
        params["delta_p"] / (params["visc"] * params["L"]) * params["delta_A"]
    return params


def homogenise_inputs(inputs: np.array, params: pd.DataFrame):
    inputs_h = inputs.copy()
    for i, input in enumerate(inputs):
        inputs_h[i] = input * params["q"][i]
    return inputs_h


def homogenise_labels(labels: np.array, params: pd.DataFrame):
    labels_h = labels.clone()
    for i, label in enumerate(labels):
        labels_h[i] = label / params['q'][i]
    return labels_h


def mean_loss(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-9):
    alpha = 0
    assert isinstance(eps, float)
    rel = torch.abs((truth - pred) / (truth + eps))
    abs = torch.abs(truth - pred)
    error = alpha * rel + (1 - alpha) * abs
    return error.mean(dim=(1, 2)).mean()


def train_model(
    train_data,
    test_input,
    test_labels,
    model,
    loss_fn,
    epochs=10,
    lr=0.01,
    batch_size=1,
    print_every=1,
):
    loss_dict = {"train": [], "test": []}

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # We use a `DataLoader` to get batching for free!
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)

    # Print header.
    print(f"Epoch    Train loss      Test loss")
    best_model = None
    best_loss = float('inf')
    e = None

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss_sum = 0
            for x_batch, y_batch in train_data_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                # print(torch.min(x_batch))
                # loss = loss_fn(y_pred, x_batch, y_batch)
                loss = loss_fn(y_pred, y_batch)
                epoch_loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            loss_dict["train"].append(epoch_loss_sum / len(train_data_loader))
            with torch.no_grad():
                model.eval()
                test_pred = model(test_input)
                # test_loss = loss_fn(
                #     test_pred, test_input, test_labels)
                test_loss = loss_fn(test_pred, test_labels)
                loss_dict["test"].append(test_loss.item())
                if test_loss.item() < best_loss:
                    best_model = copy.deepcopy(model)
                    best_loss = copy.copy(test_loss.item())
                    e = copy.copy(epoch)

            if (epoch + 1) % print_every == 0:
                print(
                    f"{epoch+1: <7}  {loss_dict['train'][-1]: <14.6e}  {loss_dict['test'][-1]: <13.6e}"
                )
    except KeyboardInterrupt:
        pass

    return best_model, loss_dict, e

def plot_params(params) -> None:
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].scatter(params.id, params
                     ["delta_p"], label="delta_p")
    ax[0, 0].set_title("delta_p")
    ax[0, 1].scatter(params.id, params["L"], label="L")
    ax[0, 1].set_title("L")
    ax[1, 0].scatter(params.id, params["visc"], label="visc")
    ax[1, 0].set_title("visc")
    ax[1, 1].scatter(params.id, params
                     ["delta_A"], label="delta_A")
    ax[1, 1].set_title("delta_A")
    plt.legend()
    plt.show()


def main() -> None:
    settings = parse_arguments()
    print(settings)
    if settings.seed is not None:
        torch.manual_seed(settings.seed)

    params = pd.read_csv("./inputs/train_params.csv", index_col=False)
    params = homogenise_params(params)

    inputs = np.load("./inputs/train_inputs.npy")
    labels = torch.from_numpy(np.load("./inputs/train_labels.npy"))
    inputs_h = torch.from_numpy(inputs)
    # inputs_h = torch.from_numpy(homogenise_inputs(inputs, params))
    labels = homogenise_labels(labels, params)

    # for i, _ in enumerate(labels):
    #     labels[i] = labels[i] / torch.max(labels[i])
    #     inputs_h[i] = inputs_h[i] / torch.max(inputs_h[i])
    inputs_h = inputs_h.unsqueeze(1)
    labels = labels.unsqueeze(1)

    mean_labels = labels.mean()
    labels_new = labels.clone()
    inputs_new = inputs_h.clone()


    labels = labels_new
    inputs_h = inputs_new

    data = TensorData(inputs_h, labels, settings.device)
    train, test = split_data(data, settings.split)
    test_input, test_labels = test[:]
    test_input = test_input.to(settings.device)
    test_labels = test_labels.to(settings.device)

    model = None
    old_loss_dict = {"train": [], "test": []}
    if settings.model is None:
        # model = ViscosityNet2(n=settings.neurons, k_size=settings.kernel_size,
        #                       ).to(settings.device)
        model = UNET(in_channels=inputs_h.shape[1]).to(settings.device)
    else:
        l = torch.load(settings.model)
        model = l['model']
        old_loss_dict = l['loss_dict']


    model, loss_dict, best_epoch = train_model(
        train, test_input, test_labels, model, mean_loss, settings.epochs, settings.lr, settings.batch, settings.report)
    time = datetime.datetime.now().strftime("%a_%H_%M")

    combined = {}
    for key in old_loss_dict:
        combined[key] = old_loss_dict[key] + loss_dict.get(key, [])

    for key in loss_dict:
        if key not in combined:
            combined[key] = loss_dict[key]

    loss_dict = combined

    torch.save({
        "model": model,
        "loss_dict": loss_dict,
        }, f"./outputs/{time}.pt")
    plt.plot(loss_dict["train"])
    plt.plot(loss_dict["test"])
    plt.yscale("log")
    plt.show()
    train = train[:5]
    t, tt = train[:]
    print(f"Best epoch {best_epoch}")
    predicitons = model(t)
    predicitons = predicitons * 2
    plot_comparison(t[3], tt[3], predicitons[3].detach())


if __name__ == "__main__":
    torch.serialization.add_safe_globals(
        [UNET, nn.ModuleList, nn.ConvTranspose2d, DoubleConv, nn.BatchNorm2d, ViscosityNet2, nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Dropout2d, nn.Flatten, nn.Linear])
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time:.3f} seconds")
