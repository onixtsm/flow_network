#!/bin/python
import argparse
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def plot_image(img, title=None, ax=None):
    img = img.detach().cpu() if isinstance(
        img, torch.Tensor) else img
    img = img.view(32, 64)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return ax


def plot_comparison(input_img, label_img, prediction_img):
    """
    Plots the input, label, and prediction images side by side.
    """
    # Detach tensors if they require gradients
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
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


class ViscosityNet2(nn.Module):
    def __init__(self, n, drop=0.2, pool=2, k_size=3, device="cpu") -> None:
        super().__init__()
        layer_count = int(np.ceil(63.0 / k_size))
        layers = []
        channels_int = 1
        for x in range(layer_count):
            layers.append(nn.Conv2d(channels_int, n, k_size, padding="same", bias=False))
            layers.append(nn.ReLU())
            if x % 2:
                layers.append(nn.Dropout2d(drop))
            channels_int = n
        layers.append(nn.Conv2d(n, 1, 1, padding="same", bias=False))

        self.conv = nn.Sequential(*layers)


    def mult(self, x: torch.Tensor, y):
        y = y.view(-1, 1, 32, 64)
        return y * x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # y = self.dense(y)
        y = self.mult(y, x)
        return y #x.view(-1, 1, 32, 64)


class ViscosityNet(nn.Module):
    def __init__(self, n, drop=0.2, pool=2, k_size=3, device="cpu") -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(pool),
            nn.Dropout2d(drop),

            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(pool),
            nn.Dropout2d(drop),

            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(pool),
            nn.Dropout2d(drop),

            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(pool),
            nn.Dropout2d(drop),

            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n, n, k_size, bias=False, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(pool),
            nn.Dropout2d(drop),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n * 2, 32*64, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # y = self.dense(y)
        # y = y.view(-1, 1, 32, 64)
        # x = y * x
        return y #x.view(-1, 1, 32, 64)


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
    parser.add_argument("--kernel_size","-k", type=int, help="Set seed")

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
    # [delta_A] = m^2
    # [delta_p] = Pa
    # [visc] = Pa*s
    # [L] = m
    params["q"] = params["delta_A"] * \
        params["delta_p"] / (params["visc"] * params["L"])
    return params


def homogenise_inputs(inputs: np.array, params: pd.DataFrame):
    inputs_h = inputs.copy()
    for i, input in enumerate(inputs):
        inputs_h[i] = input * params["q"][i]
    return inputs_h


def homogenise_labels(labels: np.array, params: pd.DataFrame):
    labels_h = labels.clone()
    for i, label in enumerate(labels):
        labels_h[i] = label * params['q'][i]
    return labels_h


def plot_params(params) -> None:
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].scatter(params["train"].id, params["train"]
                     ["delta_p"], label="delta_p")
    ax[0, 0].set_title("delta_p")
    ax[0, 1].scatter(params["train"].id, params["train"]["L"], label="L")
    ax[0, 1].set_title("L")
    ax[1, 0].scatter(params["train"].id, params["train"]["visc"], label="visc")
    ax[1, 0].set_title("visc")
    ax[1, 1].scatter(params["train"].id, params["train"]
                     ["delta_A"], label="delta_A")
    ax[1, 1].set_title("delta_A")
    plt.legend()
    plt.show()

    # flow_loss = torch.mean(torch.log(torch.cosh((pred - labels + eps))))
    # no_flow_loss = n1 * torch.mean(labels-inputs_rev * pred) / eps  # mean [q] m/s
    # flow_loss = n2 * ((labels - pred) / (labels + eps)).mean(dim=(1, 2)).mean()  # mean number
# [inputs] = m/s


# def loss_in_flowable(pred, inputs, labels):
#     n = 0
#     loss = 0
#     for i, pic in enumerate(inputs):
#         for j, p in enumerate(pic):
#             for k, p_1 in enumerate(p):
#                 for l, p_2 in enumerate(p_1):
#                   if p_2 != 0:
#                       loss += pred[i][j][k][l] - labels[i][j][k][l]
#                       n += 1
#     return loss / n

# def phys_loss(pred: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor, nfp: float = 1e-3, fp: float = 1) -> float:
#     pred = pred.view(inputs.shape)

def phys_loss(pred: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor, nfp: float = 1, fp: float = 1) -> float:
    eps = 1e-8
    # pred = pred.view(inputs.shape)
    inputs_n = inputs.clone()
    # for i, input in enumerate(inputs):
    #     inputs_n[i] = input / torch.max(input)
    # inputs_r = inputs_n.clone()
    # for i, input in enumerate(inputs_n):
    #     inputs_r[i] = torch.max(input) - input
    inputs_r = (inputs == 0).float()
    
    # inputs_r = 1 - inputs_n

    no_flow_loss = nfp * torch.square(torch.sum(pred * inputs_r))
    flow_loss = fp * \
        torch.square(torch.sum(inputs_n * ((labels - pred) / (labels + eps))))
    return no_flow_loss + flow_loss

def mse_loss(pred, truth, eps = 1e-8):
    assert isinstance(eps, float)
    error = torch.square((truth - pred) / (truth + eps))
    return error.mean(dim=(1, 2)).mean()

def mean_loss(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8):
    assert isinstance(eps, float)
    error = torch.abs((truth - pred) / (truth + eps))
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
                test_pred = model(test_input)  # m/s
                # test_loss = loss_fn(
                #     test_pred, test_input, test_labels)
                test_loss = loss_fn(test_pred, test_labels)
                loss_dict["test"].append(test_loss.item())

            if (epoch + 1) % print_every == 0:
                print(
                    f"{epoch+1: <7}  {loss_dict['train'][-1]                                      : <14.6e}  {loss_dict['test'][-1]: <13.6e}"
                )
    except KeyboardInterrupt:
        pass

    return model, loss_dict


def main() -> None:
    settings = parse_arguments()
    print(settings)
    if settings.seed is not None:
        torch.manual_seed(settings.seed)

    params = pd.read_csv("./inputs/train_params.csv", index_col=False)
    params = homogenise_params(params)

    inputs = np.load("./inputs/train_inputs.npy")
    labels = torch.from_numpy(np.load("./inputs/train_labels.npy"))
    if 1:
        inputs_h = torch.from_numpy(homogenise_inputs(inputs, params))
    else:
        labels = homogenise_labels(labels, params)
        inputs_h = torch.from_numpy(inputs)

    inputs_h = inputs_h.unsqueeze(1)
    labels = labels.unsqueeze(1)
    data = TensorData(inputs_h, labels, settings.device)
    train, test = split_data(data, settings.split)
    test_input, test_labels = test[:]

    test_input = test_input.to(settings.device)
    test_labels = test_labels.to(settings.device)

    model = None
    if settings.model is None:
        model = ViscosityNet2(n=settings.neurons, k_size=settings.kernel_size,
                             ).to(settings.device)
    else:
        model = torch.load(settings.model)

    model, loss_dict = train_model(
        train, test_input, test_labels, model, mean_loss, settings.epochs, settings.lr, settings.batch, settings.report)
    torch.save(model, "./model.pt")
    plt.plot(loss_dict["train"])
    plt.plot(loss_dict["test"])
    plt.yscale("log")
    plt.show()
    t, tt = train[:]
    predicitons = model(t)
    plot_comparison(t[17], tt[17], predicitons[17].detach())


if __name__ == "__main__":
    torch.serialization.add_safe_globals(
        [ViscosityNet, nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Dropout2d, nn.Flatten, nn.Linear])
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time:.3f} seconds")
