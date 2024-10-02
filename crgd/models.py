import torch
import torch.nn.functional as F
import numpy as np

from .decor import DecorLinear, DecorConv2d
from .fa import FALinear, FAConv2d
from .np import NPLinear, NPConv2d
from .bp import BPLinear, BPConv2d


class DenseNet(torch.nn.Module):
    def __init__(
        self,
        in_size=28 * 28,
        out_size=10,
        layer_type=BPLinear,
        activation_function=torch.nn.LeakyReLU,
        biases=True,
        decor_lr=1e-5,
        layer_kwargs={},
    ):
        super(DenseNet, self).__init__()
        num_hidden_layers = 4
        num_hidden_nodes = 1000
        self.layers = []

        self.layer_type = layer_type

        for i in range(num_hidden_layers + 1):
            in_dim = in_size if i == 0 else num_hidden_nodes
            out_dim = num_hidden_nodes if i < num_hidden_layers else out_size

            # Add decorrelation layer
            if decor_lr != 0:
                self.layers.append(
                    DecorLinear(
                        layer_type, in_dim, out_dim, decor_lr=decor_lr, **layer_kwargs
                    )
                )
            else:
                # Add linear layer
                self.layers.append(
                    layer_type(in_dim, out_dim, bias=biases, **layer_kwargs)
                )

            # Add activation function
            if (i + 1) < num_hidden_layers:
                self.layers.append(activation_function())

        self.model = torch.nn.Sequential(*self.layers)
        print(self.model)

    def forward(self, x):
        # Flatten any channels
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        return self.model(x)

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        # If we are doing NP, we need a different pipeline
        if self.layer_type == NPLinear:
            with torch.no_grad():
                output = self(torch.cat([data, data.clone()]))
                clean_loss = loss_func(output[: len(data)], target, onehots)
                noisy_loss = loss_func(output[len(data) :], target, onehots)
                # Multiply grad by loss differential and normalize with unit norms
                loss_differential = (clean_loss - noisy_loss) / len(data)
                multiplication = loss_differential
                for layer in self.layers:
                    if hasattr(layer, "update_grads"):
                        layer.update_grads(multiplication)
                total_loss = clean_loss.mean()
        else:
            # If not NP, we have a simple regular pipeline for loss backward
            output = self(data)
            loss = loss_func(output[: len(data)], target, onehots)
            total_loss = loss.mean()
            total_loss.backward()

        return total_loss

    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        with torch.inference_mode():
            if self.layer_type == NPLinear:
                output = self(torch.cat([data, data.clone()]))
            else:
                output = self(data)
            loss = torch.mean(
                loss_func(output[: len(data)], target, onehots)
            ).item()  # sum up batch loss
            return loss, output[: len(data)]


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        in_size,
        hidden_size=100,
        out_size=10,
        layer_type=BPConv2d,
        activation_function=torch.nn.LeakyReLU,
        biases=True,
        decor_lr=1e-5,
        layer_kwargs={},
    ):
        super(ConvNet, self).__init__()
        self.layers = []
        self.layer_type = layer_type
        self.dense_layer_type = BPLinear
        if layer_type == FAConv2d:
            self.dense_layer_type = FALinear
        if layer_type == NPConv2d:
            self.dense_layer_type = NPLinear

        self.in_shape = in_size
        current_shape = in_size

        # Fixed conv params
        padding = 1
        stride = 1
        kernel_size = 3

        # Fixed num_conv_layers
        num_conv_layers = 4

        for i in range(num_conv_layers):
            in_dim = 3 if i == 0 else 32 * (2 ** (int((i - 1) / 2)))
            out_dim = 32 * (2 ** (int(i / 2)))

            if decor_lr != 0:
                self.layers.append(
                    DecorConv2d(
                        layer_type,
                        in_dim,
                        out_dim,
                        kernel_size,
                        stride,
                        padding,
                        bias=biases,
                        decor_lr=decor_lr,
                        **layer_kwargs,
                    )
                )
            else:
                self.layers.append(
                    layer_type(
                        in_dim,
                        out_dim,
                        [kernel_size, kernel_size],
                        padding=padding,
                        stride=stride,
                        bias=biases,
                        **layer_kwargs,
                    )
                )

            # Add activation function
            self.layers.append(activation_function())

            current_shape = [
                out_dim,
                int(((current_shape[1] - kernel_size + 2 * padding) / stride + 1)),
                int(((current_shape[2] - kernel_size + 2 * padding) / stride + 1)),
            ]

            if i != 0 and (i + 1) % 2 == 0:
                # Adding a max pool between pair of convs
                self.layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

                # Max pool causes another reshape
                current_shape = [
                    out_dim,
                    int((((current_shape[1] - 2) / 2) + 1)),
                    int((((current_shape[2] - 2) / 2) + 1)),
                ]

        # Adding a flatten layer at the end of convs
        self.layers.append(torch.nn.Flatten())

        if decor_lr != 0:
            self.layers.append(
                DecorLinear(
                    self.dense_layer_type,
                    int(np.prod(current_shape)),
                    1000,
                    decor_lr=decor_lr,
                    **layer_kwargs,
                )
            )
        else:
            self.layers.append(
                self.dense_layer_type(
                    int(np.prod(current_shape)),
                    1000,
                    bias=biases,
                    **layer_kwargs,
                )
            )

        self.layers.append(activation_function())

        if decor_lr != 0:
            self.layers.append(
                DecorLinear(
                    self.dense_layer_type,
                    1000,
                    out_size,
                    decor_lr=decor_lr,
                    **layer_kwargs,
                )
            )
        else:
            self.layers.append(
                self.dense_layer_type(
                    1000,
                    out_size,
                    bias=biases,
                    **layer_kwargs,
                )
            )

        self.model = torch.nn.Sequential(*self.layers)
        print(self.model)

    def forward(self, x):
        x = x.view(x.size(0), *self.in_shape)
        return self.model(x)

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        # If we are doing NP, we need a different pipeline
        if self.layer_type == NPConv2d:
            with torch.no_grad():
                output = self(torch.cat([data, data.clone()]))
                clean_loss = loss_func(output[: len(data)], target, onehots)
                noisy_loss = loss_func(output[len(data) :], target, onehots)
                # Multiply grad by loss differential and normalize with unit norms
                loss_differential = (clean_loss - noisy_loss) / len(data)
                multiplication = loss_differential
                for layer in self.layers:
                    if hasattr(layer, "update_grads"):
                        layer.update_grads(multiplication)
                total_loss = clean_loss.mean()
        else:
            # If not NP, we have a simple regular pipeline for loss backward
            output = self(data)
            loss = loss_func(output[: len(data)], target, onehots)
            total_loss = loss.mean()
            total_loss.backward()

        return total_loss

    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        with torch.inference_mode():
            if self.layer_type == NPConv2d:
                output = self(torch.cat([data, data.clone()]))
            else:
                output = self(data)
            loss = torch.mean(
                loss_func(output[: len(data)], target, onehots)
            ).item()  # sum up batch loss
            return loss, output[: len(data)]
