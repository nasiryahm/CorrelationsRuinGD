import torch
import torch.nn.functional as F


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

        self.weight.requires_grad = False

        self.eye = torch.nn.Parameter(
            torch.eye(in_features, **factory_kwargs),
        )

        self.reset_decor_parameters()

    def reset_decor_parameters(self):
        torch.nn.init.eye_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        init_shape = input.shape
        input = input.view(init_shape[0], -1)
        self.undecorrelated_state = input
        if self.decorrelation_method == "foldiak":
            self.decorrelated_state = F.linear(
                input, torch.linalg.inv(self.weight).detach()
            )
        else:
            self.decorrelated_state = F.linear(self.undecorrelated_state, self.weight)
        return self.decorrelated_state.view(init_shape)

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
            normalizer = torch.sqrt(
                (torch.mean(self.undecorrelated_state**2))
            ) / torch.sqrt((torch.mean(self.decorrelated_state**2)) + 1e-8)

            w_grads = corr @ self.weight.data

            self.weight.data *= normalizer

        elif self.decorrelation_method == "foldiak":
            w_grads = -corr

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None

        # Update grads of decorrelation matrix
        self.weight.grad = w_grads

    def get_fwd_params(self):
        return []

    def get_decor_params(self):
        params = [self.weight]
        return params


class HalfBatchDecorLinear(DecorLinear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        half_batch_width = len(input) // 2

        self.undecorrelated_state = self.undecorrelated_state[:half_batch_width]
        self.decorrelated_state = self.decorrelated_state[:half_batch_width]
        return output


class ConvDecor(torch.nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        kernel_size: int,
        stride: int,
        padding: int,
        decorrelation_method: str = "copi",
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

        self.conv = torch.nn.Conv2d(
            in_shape[0],  # input channels
            self.matmulshape,  # output channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            **factory_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        eye = torch.eye(self.matmulshape)
        self.conv.weight.data = eye.view(
            self.matmulshape, self.in_shape[0], self.kernel_size, self.kernel_size
        ).to(self.conv.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(-1, *self.in_shape)
        self.undecorrelated_state = input
        self.decorrelated_state = self.conv(input)
        return self.decorrelated_state

    def update_grads(self, _) -> None:
        assert self.decorrelated_state is not None, "Call forward() first"
        assert self.undecorrelated_state is not None, "Call forward() first"

        self.decorrelated_state = torch.permute(self.decorrelated_state, (0, 2, 3, 1))

        self.decorrelated_state = self.decorrelated_state.view(-1, *self.matmulshape)

        # The off-diagonal correlation = (1/batch_size)*(x.T @ x)*(1.0 - I)
        corr = (1 / len(self.decorrelated_state)) * (
            self.decorrelated_state.transpose(0, 1) @ self.decorrelated_state
        )

        if self.decorrelation_method == "copi":
            off_diag_corr = corr * (1.0 - self.eye)
            w_grads = off_diag_corr @ self.weight.data
        elif self.decorrelation_method == "scaled":
            normalizer = torch.sqrt(
                (torch.mean(self.undecorrelated_state**2))
            ) / torch.sqrt((torch.mean(self.decorrelated_state**2)) + 1e-8)

            w_grads = corr @ self.weight.data.view(self.matmulshape, self.matmulshape)

            self.weight.data *= normalizer

        # Zero-ing the decorrelated state so that it cannot be re-used
        self.undecorrelated_state = None
        self.decorrelated_state = None

        # Update grads of decorrelation matrix
        self.weight.grad = w_grads.view(
            self.matmulshape, self.in_shape[0], self.kernel_size, self.kernel_size
        )

    def get_fwd_params(self):
        return []

    def get_decor_params(self):
        params = [self.conv.weight]
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
