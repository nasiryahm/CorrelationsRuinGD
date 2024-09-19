import torch
import torch.nn.functional as F
import numpy as np

from decor import DecorLinear, HalfBatchDecorLinear, MultiDecor, DecorConv
from fa import FALinear, FAConv2d
from np import NPLinear
from bp import BPLinear, BPConv2d


class Net(torch.nn.Module):
    def __init__(
        self,
        in_size=28 * 28,
        out_size=10,
        hidden_size=1000,
        n_hidden_layers=0,
        layer_type=BPLinear,
        activation_function=torch.nn.LeakyReLU,
        biases=True,
        decorrelation_method="copi",
        layer_kwargs={},
    ):
        super(Net, self).__init__()
        self.layers = []
        self.decorrelation = decorrelation_method is None

        for i in range(n_hidden_layers + 1):
            in_dim = in_size if i == 0 else hidden_size
            out_dim = hidden_size if i < n_hidden_layers else out_size
            if decorrelation_method is not None:
                self.layers.append(
                    DecorLinear(
                        in_dim,
                        out_dim,
                        decorrelation_method=decorrelation_method,
                        bias=biases,
                        fwd_layer=layer_type,
                        **layer_kwargs,
                    )
                )
            else:
                self.layers.append(
                    layer_type(
                        in_dim,
                        out_dim,
                        bias=biases,
                        **layer_kwargs,
                    )
                )

            # add activation function
            if (i + 1) < n_hidden_layers:
                self.layers.append(activation_function())

        self.activation_function = activation_function
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        # Flatten any convs
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        for indx, layer in enumerate(self.layers):
            x = layer(x)

        return x

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        # Duplicate data for network clean/noisy pass
        output = self(data)
        loss = loss_func(output[: len(data)], target, onehots)
        total_loss = loss.mean()
        total_loss.backward()
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, "update_grads"):
                    layer.update_grads(None)

        return total_loss

    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        with torch.no_grad():
            output = self(data)
            loss = torch.mean(
                loss_func(output, target, onehots)
            ).item()  # sum up batch loss
            return loss, output

    def get_fwd_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_fwd_params"):
                params.extend(layer.get_fwd_params())
        return params

    def get_decor_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_decor_params"):
                params.extend(layer.get_decor_params())
        return params


class PerturbNet(torch.nn.Module):
    def __init__(
        self,
        in_size=28 * 28,
        out_size=10,
        hidden_size=1000,
        n_hidden_layers=0,
        layer_type=NPLinear,
        activation_function=torch.nn.LeakyReLU,
        biases=True,
        decorrelation_method="copi",
        layer_kwargs={},
    ):
        super(PerturbNet, self).__init__()
        self.layers = []
        self.decorrelation = decorrelation_method is not None
        self.N = hidden_size * n_hidden_layers + out_size

        for i in range(n_hidden_layers + 1):
            in_dim = in_size if i == 0 else hidden_size
            out_dim = hidden_size if i < n_hidden_layers else out_size
            if self.decorrelation:
                self.layers.append(
                    HalfBatchDecorLinear(
                        in_dim,
                        out_dim,
                        decorrelation_method=decorrelation_method,
                        bias=biases,
                        fwd_layer=layer_type,
                        **layer_kwargs,
                    )
                )
            else:
                self.layers.append(
                    layer_type(in_dim, out_dim, bias=biases, **layer_kwargs)
                )

            # add activation function
            if (i + 1) < n_hidden_layers:
                self.layers.append(activation_function())

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        # Flatten any convs
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        for indx, layer in enumerate(self.layers):
            x = layer(x)

        return x

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        with torch.inference_mode():
            # Duplicate data for network clean/noisy pass
            output = self(torch.concatenate([data, data.clone()]))
            clean_loss = loss_func(
                output[: len(data)], target, onehots
            )  # sum up batch loss
            noisy_loss = loss_func(
                output[len(data) :], target, onehots
            )  # sum up batch loss
            # Multiply grad by loss differential and normalize with unit norms
            loss_differential = (clean_loss - noisy_loss) / len(data)
            multiplication = loss_differential
            for layer in self.layers:
                if hasattr(layer, "update_grads"):
                    layer.update_grads(multiplication)

            return clean_loss.mean()

    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        with torch.inference_mode():
            output = self(torch.concatenate([data, data.clone()]))
            loss = torch.mean(
                loss_func(output[: len(data)], target, onehots)
            ).item()  # sum up batch loss
            return loss, output[: len(data)]

    def get_fwd_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_fwd_params"):
                params.extend(layer.get_fwd_params())
        return params

    def get_decor_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_decor_params"):
                params.extend(layer.get_decor_params())
        return params


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        in_size,
        hidden_size=100,
        n_hidden_layers=0,
        out_size=10,
        layer_type=BPConv2d,
        activation_function=torch.nn.ReLU,
        biases=True,
        decorrelation_method="copi",
        layer_kwargs={},
    ):
        super(ConvNet, self).__init__()
        self.layers = []

        assert n_hidden_layers > 0, "Need at least one hidden layer"
        conv_layer_type = layer_type
        assert conv_layer_type in [BPConv2d, FAConv2d]
        dense_layer_type = BPLinear if layer_type == BPConv2d else FALinear

        self.in_shape = in_size  # This is a shape for 3D input
        current_shape = in_size
        padding = 1  # Fixing padding and stride
        stride = 1
        for i in range(n_hidden_layers):
            in_dim = 3 if i == 0 else 32 * (2 ** (int((i - 1) / 2)))
            out_dim = 32 * (2 ** (int(i / 2)))  # Fixed output channel size
            kernel_size = 3

            if decorrelation_method is not None:
                self.layers.append(
                    DecorConv(
                        current_shape,
                        out_dim,
                        kernel_size,
                        stride,
                        padding,
                        decorrelation_method=decorrelation_method,
                        conv_layer=conv_layer_type,
                        bias=biases,
                        **layer_kwargs,
                    )
                )

            else:
                self.layers.append(
                    conv_layer_type(
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

        # Adding a flatten layer
        self.layers.append(torch.nn.Flatten())

        # Adding one dense layer before output
        if decorrelation_method is not None:

            self.layers.append(
                DecorLinear(
                    int(np.prod(current_shape)),
                    hidden_size,
                    decorrelation_method=decorrelation_method,
                    fwd_layer=dense_layer_type,
                    bias=biases,
                    **layer_kwargs,
                )
            )

            self.layers.append(activation_function())

            self.layers.append(
                DecorLinear(
                    hidden_size,
                    out_size,
                    decorrelation_method=decorrelation_method,
                    fwd_layer=dense_layer_type,
                    **layer_kwargs,
                )
            )
        else:

            self.layers.append(
                dense_layer_type(
                    int(np.prod(current_shape)),
                    hidden_size,
                    bias=biases,
                    **layer_kwargs,
                )
            )

            self.layers.append(activation_function())

            self.layers.append(
                dense_layer_type(
                    hidden_size,
                    out_size,
                    bias=biases,
                    **layer_kwargs,
                )
            )

        self.layers = torch.nn.ModuleList(self.layers)

        print(self.layers)

    def forward(self, x):
        x = x.view(x.size(0), *self.in_shape)
        for indx, layer in enumerate(self.layers):
            x = layer(x)

        return x

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        # Duplicate data for network clean/noisy pass
        output = self(data)
        loss = loss_func(output, target, onehots)
        total_loss = loss.mean()
        total_loss.backward()
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, "update_grads"):
                    layer.update_grads(None)

        return total_loss

    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        with torch.no_grad():
            output = self(data)
            loss = torch.mean(
                loss_func(output, target, onehots)
            ).item()  # sum up batch loss
            return loss, output

    def get_fwd_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_fwd_params"):
                params.extend(layer.get_fwd_params())
            else:
                params.extend(layer.parameters())
        return params

    def get_decor_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "get_decor_params"):
                params.extend(layer.get_decor_params())
        return params
