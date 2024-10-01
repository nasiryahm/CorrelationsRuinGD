import torch
import torch.nn.functional as F
import bp
import numpy as np


class DeMeaner(torch.nn.Module):
    def __init__(self, num_features, momentum=0.1, **kwargs):
        super(DeMeaner, self).__init__()
        self.num_features = num_features
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(num_features, **kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()

    def forward(self, input):
        assert self.num_features == input.shape[1], "Input shape mismatch"

        if self.training:
            avg = torch.mean(input, axis=0)
            while len(avg.shape) > 1:
                avg = torch.mean(avg, axis=-1)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * avg
        else:
            avg = self.running_mean

        avg = avg[None, :]
        while len(avg.shape) < len(input.shape):
            avg = avg.unsqueeze(-1)

        return input - avg


class DecorLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decorrelation_method: str = "scaled",
        fwd_layer: torch.nn.Module = bp.BPLinear,
        bias: bool = True,
        device=None,
        dtype=None,
        **layer_kwargs,
    ) -> None:
        super().__init__()
        assert decorrelation_method in ["demeaned-scaled", "scaled"]
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.decorrelation_method = decorrelation_method

        self.decor_weight = torch.nn.Parameter(
            torch.empty(in_features, in_features, **factory_kwargs),
        )

        self.decor_weight.requires_grad = False

        self.eye = torch.nn.Parameter(
            torch.eye(in_features, **factory_kwargs),
        )

        self.fwd_layer = fwd_layer(in_features, out_features, bias=bias, **layer_kwargs)

        self.demeaner = DeMeaner(in_features, **factory_kwargs)

        self.reset_decor_parameters()

    def reset_decor_parameters(self):
        torch.nn.init.eye_(self.decor_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2, "Input must be 2D"

        self.undecorrelated_state = input
        if self.decorrelation_method == "demeaned-scaled":
            input = self.demeaner(self.undecorrelated_state).reshape(input.shape)

        # First decorrelate
        self.decorrelated_state = F.linear(input, self.decor_weight)

        # Next forward pass
        output = self.fwd_layer(self.decorrelated_state)

        return output

    def compute_normalization(self):
        self.normalization = torch.sqrt(
            (torch.mean(self.undecorrelated_state**2))
        ) / (torch.sqrt((torch.mean(self.decorrelated_state**2))) + 1e-8)

    def update_grads(self, feedback) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"
        assert self.undecorrelated_state is not None, "Call forward() first"

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        num_samples = int(0.1 * len(self.undecorrelated_state)) + 1
        self.decorrelated_state = self.decorrelated_state[:num_samples]
        self.undecorrelated_state = self.undecorrelated_state[:num_samples]
        corr = (1 / len(self.decorrelated_state)) * (
            self.decorrelated_state.transpose(0, 1) @ self.decorrelated_state
        )

        if self.decorrelation_method in ["demeaned-scaled", "scaled"]:
            self.compute_normalization()
            self.decor_weight.data *= self.normalization
            w_grads = corr @ self.decor_weight.data

        elif self.decorrelation_method == "foldiak":
            w_grads = -corr

        # Update grads of decorrelation matrix
        self.decor_weight.grad = w_grads

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None
        self.normalization = None

        self.fwd_layer.update_grads(feedback)

    def get_fwd_params(self):
        return [] + self.fwd_layer.get_fwd_params()

    def get_decor_params(self):
        params = [self.decor_weight] + self.fwd_layer.get_decor_params()
        return params


class HalfBatchDecorLinear(DecorLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        half_batch_width = len(input) // 2

        self.undecorrelated_state = self.undecorrelated_state[:half_batch_width]
        self.decorrelated_state = self.decorrelated_state[:half_batch_width]
        return output


class DecorConv(torch.nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        decorrelation_method: str = "scaled",
        conv_layer: torch.nn.Module = bp.BPConv2d,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        # Assumes padding of zero and stride of 1
        super().__init__()
        assert len(in_shape) == 3, "ConvDecor only supports 3D inputs, (C, H, W)"
        assert decorrelation_method in ["demeaned-scaled", "scaled"]
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_shape = in_shape
        self.kernel_size = kernel_size
        self.matmulshape = in_shape[0] * kernel_size * kernel_size
        self.decorrelation_method = decorrelation_method
        self.stride = stride
        self.padding = padding

        self.folder = torch.nn.Fold(
            output_size=(
                int((in_shape[1] - self.kernel_size + 2 * padding) / stride + 1),
                int((in_shape[1] - self.kernel_size + 2 * padding) / stride + 1),
            ),
            kernel_size=(1, 1),
        )

        self.unfolder = torch.nn.Unfold(
            kernel_size=(kernel_size, kernel_size),
            dilation=1,
            padding=padding,
            stride=stride,
        )

        self.decor_conv = torch.nn.Conv2d(
            in_channels=in_shape[0],
            out_channels=self.matmulshape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # This layer then acts as a normal convolutional layer over decorrelated patches
        self.fwd_conv = conv_layer(
            self.matmulshape,  # input channels
            out_channels,  # output channels
            [1, 1],
            padding=0,
            stride=1,
            bias=bias,
            **factory_kwargs,
        )

        self.demeaner = DeMeaner(in_shape[0], **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.decor_conv.weight.data = (
            torch.eye(self.matmulshape)
            .reshape(
                self.matmulshape, self.in_shape[0], self.kernel_size, self.kernel_size
            )
            .to(self.decor_conv.weight.device)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.undecorrelated_state = input
        if self.decorrelation_method == "demeaned-scaled":
            input = self.demeaner(self.undecorrelated_state)

        self.decorrelated_state = self.decor_conv(input)
        output = self.fwd_conv(self.decorrelated_state)

        return output

    def compute_normalization(self):
        self.normalization = torch.sqrt(
            (
                torch.mean(
                    self.undecorrelated_state[:, :, :] ** 2,
                ).flatten()
            )
        ) / (
            torch.sqrt(
                torch.mean(
                    self.decorrelated_state[:, :, :] ** 2,
                ).flatten()
            )
            + 1e-8
        )

    def update_grads(self, feedback) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"
        assert self.undecorrelated_state is not None, "Call forward() first"

        # Conv outputs are (batch, out_channels, H, W)

        # Height and width dimensions are now equivalent to a batch dimension
        num_samples = int(0.1 * len(self.undecorrelated_state)) + 1
        self.decorrelated_state = self.decorrelated_state[:num_samples]
        self.undecorrelated_state = self.undecorrelated_state[:num_samples]
        self.undecorrelated_state = self.folder(
            self.unfolder(self.undecorrelated_state)
        )

        self.compute_normalization()

        modified_decorrelated_state = (
            torch.permute(
                self.decorrelated_state.reshape(num_samples, self.matmulshape, -1),
                (0, 2, 1),
            )
            .contiguous()
            .reshape(-1, self.matmulshape)
        )

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        corr = (1 / len(modified_decorrelated_state)) * (
            modified_decorrelated_state.transpose(0, 1) @ modified_decorrelated_state
        )

        self.decor_conv.weight.data *= self.normalization
        w_grads = corr @ self.decor_conv.weight.reshape(
            self.matmulshape, self.matmulshape
        )

        # Update grads of decorrelation matrix
        self.decor_conv.weight.grad = w_grads.reshape(
            self.matmulshape, self.in_shape[0], self.kernel_size, self.kernel_size
        )

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None
        self.normalization = None

        self.fwd_conv.update_grads(feedback)

    def get_fwd_params(self):
        return [] + self.fwd_conv.get_fwd_params()

    def get_decor_params(self):
        params = [self.decor_conv.weight] + self.fwd_conv.get_decor_params()
        return params
