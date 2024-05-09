import torch
import torch.nn.functional as F
import numpy as np


class DecorLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decorrelation_method: str = "copi",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert in_features == out_features, "DecorLinear only supports square matrices"
        assert decorrelation_method in ["copi", "scaled", "foldiak"]
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.decorrelation_method = decorrelation_method

        self.weight = torch.nn.Parameter(
            torch.empty(in_features, in_features, **factory_kwargs),
        )

        if self.decorrelation_method == "scaled":
            self.gains = torch.nn.Parameter(torch.empty(in_features, **factory_kwargs))

        self.weight.requires_grad = False

        self.eye = torch.nn.Parameter(
            torch.eye(in_features, **factory_kwargs),
        )

        self.reset_decor_parameters()

    def reset_decor_parameters(self):
        torch.nn.init.eye_(self.weight)
        if self.decorrelation_method == "scaled":
            torch.nn.init.ones_(self.gains)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.undecorrelated_state = input
        gain = self.gains if self.decorrelation_method == "scaled" else 1.0
        if self.decorrelation_method == "foldiak":
            self.decorrelated_state = gain * F.linear(
                input, torch.linalg.inv(self.weight).detach()
            )
        else:
            self.decorrelated_state = gain * F.linear(input, self.weight)
        return self.decorrelated_state

    def update_grads(self, _) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"
        assert self.undecorrelated_state is not None, "Call forward() first"

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        corr = (1 / len(self.decorrelated_state)) * (
            self.decorrelated_state.transpose(0, 1) @ self.decorrelated_state
        )

        if self.decorrelation_method == "copi":
            off_diag_corr = corr * (1.0 - self.eye)
            w_grads = off_diag_corr @ self.weight.data
        elif self.decorrelation_method == "scaled":
            w_grads = corr @ self.weight.data
            normalizer = torch.sqrt(
                torch.sum(self.undecorrelated_state**2, axis=0)
                / (torch.sum(self.decorrelated_state**2, axis=0) + 1e-8)
            )
            g_grads = (
                normalizer * self.gains.data + (1.0 - normalizer) * self.gains.data
            )
            # self.gains *= normalizer
        elif self.decorrelation_method == "foldiak":
            w_grads = -corr

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None

        # Update grads of decorrelation matrix
        self.weight.grad = w_grads
        if self.decorrelation_method == "scaled":
            self.gains.grad = g_grads

    def get_fwd_params(self):
        return []

    def get_decor_params(self):
        params = [self.weight]
        if self.decorrelation_method == "scaled":
            params += [self.gains]
        return params


class BPLinear(torch.nn.Linear):
    """BP Linear layer"""

    def __str__(self):
        return "BPLinear"

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []


class DecorNet(torch.nn.Module):
    def __init__(
        self,
        in_size=28 * 28,
        out_size=10,
        hidden_size=1000,
        n_hidden_layers=0,
        layer_type=BPLinear,
        activation_function=torch.nn.LeakyReLU(),
        biases=True,
        decorrelation=False,
        decorrelation_method="copi",
    ):
        super(DecorNet, self).__init__()
        self.layers = []
        self.decorrelation = decorrelation

        for i in range(n_hidden_layers + 1):
            in_dim = in_size if i == 0 else hidden_size
            out_dim = hidden_size if i < n_hidden_layers else out_size
            if decorrelation_method is not None:
                self.layers.append(
                    DecorLinear(
                        in_dim, in_dim, decorrelation_method=decorrelation_method
                    )
                )
            self.layers.append(
                layer_type(
                    in_dim,
                    out_dim,
                    bias=biases,
                )
            )

        self.activation_function = activation_function
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for indx, layer in enumerate(self.layers):
            x = layer(x)
            # TODO: Don't activation func if decor
            if (indx + 1) < len(self.layers) and not isinstance(layer, DecorLinear):
                x = self.activation_function(x)
        return x

    def train_step(self, data, target, onehots, loss_func):
        # Duplicate data for network clean/noisy pass
        output = self(data)
        loss = loss_func(output[: len(data)], target, onehots)
        total_loss = loss.sum()
        total_loss.backward()
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, DecorLinear):
                    layer.update_grads(None)

        return total_loss

    def test_step(self, data, target, onehots, loss_func):
        with torch.no_grad():
            output = self(data)
            loss = torch.sum(
                loss_func(output, target, onehots)
            ).item()  # sum up batch loss
            return loss, output

    def get_fwd_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_fwd_params())
        return params

    def get_decor_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_decor_params())
        return params
