import torch
import torch.nn.functional as F
import bp


class DecorLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decorrelation_method: str = "copi",
        fwd_layer: torch.nn.Module = bp.BPLinear,
        bias: bool = True,
        device=None,
        dtype=None,
        **layer_kwargs,
    ) -> None:
        super().__init__()
        assert decorrelation_method in ["copi", "scaled", "foldiak"]
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

        self.reset_decor_parameters()

    def reset_decor_parameters(self):
        torch.nn.init.eye_(self.decor_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2, "Input must be 2D"

        self.undecorrelated_state = input
        if self.decorrelation_method == "foldiak":
            self.decorrelated_state = F.linear(
                input, torch.linalg.inv(self.decor_weight).detach()
            )
        else:
            self.decorrelated_state = F.linear(
                self.undecorrelated_state, self.decor_weight
            )

        return self.fwd_layer(self.decorrelated_state)

    def update_grads(self, feedback) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"
        assert self.undecorrelated_state is not None, "Call forward() first"

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        corr = (1 / len(self.decorrelated_state)) * (
            self.decorrelated_state.transpose(0, 1) @ self.decorrelated_state
        )

        if self.decorrelation_method == "copi":
            off_diag_corr = corr * (1.0 - self.eye)
            w_grads = off_diag_corr @ self.decor_weight.data
        elif self.decorrelation_method == "scaled":
            normalizer = torch.sqrt(
                (torch.mean(self.undecorrelated_state**2, axis=0))
            ) / torch.sqrt((torch.mean(self.decorrelated_state**2, axis=0)) + 1e-8)

            w_grads = corr @ self.decor_weight.data

            self.decor_weight.data *= normalizer[:, None]

        elif self.decorrelation_method == "foldiak":
            w_grads = -corr

        # Update grads of decorrelation matrix
        self.decor_weight.grad = w_grads

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None

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
        decorrelation_method: str = "copi",
        conv_layer: torch.nn.Module = bp.BPConv2d,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        # Assumes padding of zero and stride of 1
        super().__init__()
        assert len(in_shape) == 3, "ConvDecor only supports 3D inputs, (C, H, W)"
        assert decorrelation_method in ["scaled", "foldiak"]
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_shape = in_shape
        self.kernel_size = kernel_size
        self.matmulshape = in_shape[0] * kernel_size * kernel_size
        self.decorrelation_method = decorrelation_method

        self.decor_conv = torch.nn.Conv2d(
            in_shape[0],  # input channels
            self.matmulshape,  # output channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            **factory_kwargs,
        )

        self.fwd_conv = conv_layer(
            self.matmulshape,  # input channels
            out_channels,  # output channels
            [1, 1],
            padding=0,
            stride=1,
            bias=bias,
            **factory_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        eye = torch.eye(self.matmulshape)
        self.decor_conv.weight.data = eye.view(
            self.matmulshape, self.in_shape[0], self.kernel_size, self.kernel_size
        ).to(self.decor_conv.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(-1, *self.in_shape)
        self.undecorrelated_state = input
        self.decorrelated_state = self.decor_conv(input)
        return self.fwd_conv(self.decorrelated_state)

    def update_grads(self, feedback) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"
        assert self.undecorrelated_state is not None, "Call forward() first"

        single_patch = self.decorrelated_state[:, :, 0, 0]

        self.decorrelated_state = torch.permute(
            self.decorrelated_state, (0, 2, 3, 1)
        ).contiguous()

        self.decorrelated_state = self.decorrelated_state.reshape(-1, self.matmulshape)

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        corr = (1 / len(self.decorrelated_state)) * (
            self.decorrelated_state.transpose(0, 1) @ self.decorrelated_state
        )

        if self.decorrelation_method == "copi":
            off_diag_corr = corr * (1.0 - self.eye)
            w_grads = off_diag_corr @ self.weight.data
        elif self.decorrelation_method == "scaled":
            normalizer = torch.sqrt(
                (
                    torch.mean(
                        self.undecorrelated_state[
                            :, :, : self.kernel_size, : self.kernel_size
                        ]
                        ** 2
                    )
                )
            ) / torch.sqrt((torch.mean(single_patch**2)) + 1e-8)

            self.decor_conv.weight.data *= normalizer

            w_grads = corr @ self.decor_conv.weight.data.reshape(
                self.matmulshape, self.matmulshape
            )

        # Update grads of decorrelation matrix
        self.decor_conv.weight.grad = w_grads.reshape(
            self.matmulshape, self.in_shape[0], self.kernel_size, self.kernel_size
        )

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None

        self.fwd_conv.update_grads(feedback)

    def get_fwd_params(self):
        return [] + self.fwd_conv.get_fwd_params()

    def get_decor_params(self):
        params = [self.decor_conv.weight] + self.fwd_conv.get_decor_params()
        return params


class MultiDecor(torch.nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        decorrelation_method: str = "copi",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert len(in_shape) == 3, "MultiDecor only supports 3D inputs, (C, H, W)"
        assert decorrelation_method in ["copi", "scaled", "foldiak"]
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_shape = in_shape
        self.channel_dim = in_shape[0]
        self.image_dim = int(in_shape[1] * in_shape[2])

        self.channel_decor = DecorLinear(
            self.channel_dim,
            self.channel_dim,
            decorrelation_method,
            **factory_kwargs,
        )
        self.image_decor = DecorLinear(
            self.image_dim,
            self.image_dim,
            decorrelation_method,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(-1, *self.in_shape)
        init_shape = input.shape
        input = input.view(init_shape[0] * self.channel_dim, self.image_dim)
        input = self.image_decor.forward(input)
        input = input.view(init_shape)

        input = input.permute(0, 2, 3, 1).reshape(-1, self.channel_dim).contiguous()
        input = self.channel_decor.forward(input)
        input = input.view(init_shape[0], init_shape[2], init_shape[3], init_shape[1])
        input = input.permute(0, 3, 1, 2).contiguous()

        image_renorm = torch.sqrt(
            torch.sum(self.image_decor.decorrelated_state**2) / torch.sum(input**2)
        )

        self.image_decor.decorrelated_state = (
            image_renorm
            * input.clone().view(init_shape[0] * self.channel_dim, self.image_dim)[
                : init_shape[0]
            ]
        )

        return input

    # Define a function to take the input and run the channel decor forward
    def channel_decor_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.channel_decor.forward(input)

    def update_grads(self, _) -> None:
        self.channel_decor.update_grads(_)
        self.image_decor.update_grads(_)

    def get_fwd_params(self):
        params = self.channel_decor.get_fwd_params() + self.image_decor.get_fwd_params()
        return params

    def get_decor_params(self):
        params = (
            self.channel_decor.get_decor_params() + self.image_decor.get_decor_params()
        )
        return params
