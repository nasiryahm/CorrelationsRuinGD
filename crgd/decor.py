import torch
import torch.nn.functional as F
from typing import Sequence


class Decorrelator(torch.nn.Module):

    def __init__(
        self,
        num_features: int,
        lr: float = 1e-5,
        mean_momentum: float = 0.1,
        perc_samples: float = 0.1,
        **kwargs
    ):
        super(Decorrelator, self).__init__()
        self.num_features = num_features
        self.mean_momentum = mean_momentum
        self.lr = lr
        self.perc_samples = perc_samples

        self.update = None

        # Register buffer is used for variables that are not updated during backprop
        self.register_buffer("decor_weight", torch.eye(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("I", torch.eye(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.decor_weight = torch.eye(
            self.num_features, device=self.decor_weight.device
        )
        self.I = torch.eye(self.num_features, device=self.decor_weight.device)
        self.running_mean.zero_()

    def forward(self, input):
        assert self.num_features == input.shape[1], "Input shape mismatch"

        # Decorrelation
        output = torch.einsum("ji, ni...->nj...", self.decor_weight, input)
        if self.training:
            with torch.no_grad():
                # Sub-sample the data
                num_samples = int(self.perc_samples * len(input)) + 1
                decor_state = output[:num_samples]

                # Reshape data if there are multiple channels
                if len(decor_state.shape) == 3:
                    decor_state = decor_state.permute(0, 2, 1)
                    decor_state = decor_state.reshape(-1, decor_state.shape[-1])

                # Compute the correlation matrix
                corr = (1 / len(decor_state)) * (
                    decor_state.transpose(0, 1) @ decor_state
                )

                # Update the decorrelation matrix
                corr_distance = corr - self.I
                self.update = self.I - self.lr * corr_distance
                self.decor_weight = self.update @ self.decor_weight

        return output


# class Decorrelator2D(torch.nn.Module):
#     def __init__(
#         self,
#         num_features: int,
#         kernel_size: int | Sequence[int],
#         stride: int | Sequence[int],
#         padding: int | Sequence[int],
#         dilation: int | Sequence[int],
#         lr: float = 1e-5,
#         mean_momentum: float = 0.1,
#         perc_samples: float = 0.1,
#         **kwargs
#     ):
#         super(Decorrelator2D, self).__init__()
#         self.num_features = num_features
#         self.kernel_size = (
#             kernel_size
#             if isinstance(kernel_size, Sequence)
#             else (kernel_size, kernel_size)
#         )
#         self.padding = padding if isinstance(padding, Sequence) else (padding, padding)
#         self.dilation = (
#             dilation if isinstance(dilation, Sequence) else (dilation, dilation)
#         )
#         self.stride = stride if isinstance(stride, Sequence) else (stride, stride)
#         self.mean_momentum = mean_momentum
#         self.lr = lr
#         self.perc_samples = perc_samples

#         # TODO: please relax this requirement and get rid of the check
#         assert (
#             kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
#         ), "Kernel size must be odd"

#         # Register buffer is used for variables that are not updated during backprop
#         self.mid_dim = num_features * kernel_size[0] * kernel_size[1]
#         self.register_buffer(
#             "decor_weight",
#             torch.zeros(
#                 self.mid_dim,
#                 num_features,
#                 kernel_size[0],
#                 kernel_size[1],
#             ),
#         )
#         self.register_buffer(
#             "I",
#             torch.zeros(
#                 self.mid_dim,
#                 self.mid_dim,
#             ),
#         )
#         self.register_buffer("running_mean", torch.zeros(num_features))
#         self.update = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.decor_weight = torch.eye(self.mid_dim).view(
#             self.mid_dim,
#             self.num_features,
#             self.kernel_size[0],
#             self.kernel_size[1],
#         )
#         self.I = torch.eye(self.mid_dim, device=self.decor_weight.device)
#         self.running_mean.zero_()

#     def forward(self, input):
#         assert self.num_features == input.shape[1], "Input shape mismatch"

#         # Decorrelation
#         output = F.conv2d(
#             input,
#             self.decor_weight,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#         )

#         # If we are in a training pass, update decorrelation
#         if self.training:
#             with torch.no_grad():
#                 # Sub-sample the data
#                 num_samples = int(self.perc_samples * len(input)) + 1
#                 decor_state = output[:num_samples]

#                 # Set up patches as if they are batch-samples
#                 # Permuting this data is easier than permuting every matrix instead
#                 mod_decor_state = (
#                     torch.permute(decor_state, (0, 2, 3, 1)).reshape(-1, self.mid_dim)
#                 ).contiguous()

#                 # Compute the correlation matrix
#                 corr = (1 / len(mod_decor_state)) * (
#                     mod_decor_state.T @ mod_decor_state
#                 )

#                 # Update the decorrelation matrix
#                 corr_dist = corr - self.I

#                 # Assign the update
#                 self.update = self.I - self.lr * corr_dist

#                 new_decor_weight = self.update @ (
#                     self.decor_weight.view(self.mid_dim, -1)
#                 )

#                 self.decor_weight = new_decor_weight.reshape(
#                     self.mid_dim,
#                     self.num_features,
#                     self.kernel_size[0],
#                     self.kernel_size[1],
#                 )

#         return output


class DecorLinear(torch.nn.Module):

    def __init__(
        self,
        layer_type: torch.nn.Module,
        in_features: int,
        out_features: int,
        bias: bool = True,
        decor_lr: float = 1e-5,
        **kwargs
    ) -> None:
        super(DecorLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decor = Decorrelator(
            self.in_features,
            lr=decor_lr,
        )

        self.linear = layer_type(
            self.in_features, self.out_features, bias=bias, **kwargs
        )

    def __str__(self):
        return "DecorLinear"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # We only want to include the inverse after any updates are done
        decor_input = self.decor(input)
        output = self.linear(decor_input)
        if self.decor.update is not None:
            with torch.no_grad():
                inverse = torch.inverse(self.decor.update)
                self.linear.weight.data = self.linear.weight.data @ inverse
                self.decor.update = None
        return output


class DecorConv2d(torch.nn.Module):

    def __init__(
        self,
        layer_type: torch.nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = (1, 1),
        padding: int | Sequence[int] = (0, 0),
        dilation: int | Sequence[int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        decor_lr: float = 1e-5,
        **kwargs
    ) -> None:
        super(DecorConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, Sequence)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, Sequence) else (stride, stride)
        self.padding = padding if isinstance(padding, Sequence) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, Sequence) else (dilation, dilation)
        )
        self.mid_dim = in_channels * self.kernel_size[0] * self.kernel_size[1]

        self.decor = Decorrelator(
            self.in_channels * self.kernel_size[0] * self.kernel_size[1],
            lr=decor_lr,
            perc_samples=0.01,
        )

        self.conv = layer_type(
            self.mid_dim,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            **kwargs
        )
        self.counter = 0

    def __str__(self):
        return "DecorConv2d"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        decor_input = self.decor(patches)

        batch_size, _, H_in, W_in = input.shape
        # We need to calculate the output height and width.
        H_out = (
            H_in
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        W_out = (
            W_in
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        # Reshape the result from (B, C_out, L) to (B, C_out, H_out, W_out)
        decor_input = decor_input.view(batch_size, self.mid_dim, H_out, W_out)
        output = self.conv(decor_input)
        with torch.no_grad():
            if self.decor.update is not None:
                self.conv.weight.data = (
                    self.conv.weight.data.squeeze() @ torch.inverse(self.decor.update)
                ).view(self.conv.weight.data.shape)
                self.decor.update = None
        return output
