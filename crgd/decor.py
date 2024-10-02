import torch
import torch.nn.functional as F


class Decorrelator(torch.nn.Module):
    def __init__(
        self, num_features, lr=1e-5, mean_momentum=0.1, perc_samples=0.1, **kwargs
    ):
        super(Decorrelator, self).__init__()
        self.num_features = num_features
        self.mean_momentum = mean_momentum
        self.lr = lr
        self.perc_samples = perc_samples

        # Register buffer is used for variables that are not updated during backprop
        self.register_buffer("decor_weight", torch.eye(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.decor_weight = torch.eye(
            self.num_features, device=self.decor_weight.device
        )
        self.running_mean.zero_()

    def demean(self, input):
        # Demeaning
        if self.training:
            mean = torch.mean(input, axis=0)
            while len(mean.shape) > 1:
                mean = torch.mean(mean, axis=-1)
            self.running_mean = (
                1 - self.mean_momentum
            ) * self.running_mean + self.mean_momentum * mean
        else:
            mean = self.running_mean

        mean = mean[None, :]
        while len(mean.shape) < len(input.shape):
            mean = mean.unsqueeze(-1)

        return input - mean

    def forward(self, input):
        assert self.num_features == input.shape[1], "Input shape mismatch"

        # Demean
        demeaned_input = self.demean(input)

        # Decorrelation
        output = torch.einsum("ni...,ij->nj...", demeaned_input, self.decor_weight)
        if self.training:
            with torch.no_grad():
                # Sub-sample the data
                num_samples = int(self.perc_samples * len(input)) + 1
                undecor_state = input[:num_samples]
                decor_state = output[:num_samples]

                # Reshape data if there are multiple channels
                if len(decor_state.shape) > 2:
                    decor_state = decor_state.permute(0, 2, 3, 1)
                    undecor_state = undecor_state.permute(0, 2, 3, 1)
                    decor_state = decor_state.reshape(-1, decor_state.shape[-1])
                    undecor_state = undecor_state.reshape(-1, undecor_state.shape[-1])

                # Compute the correlation matrix
                corr = (1 / len(decor_state)) * (
                    decor_state.transpose(0, 1) @ decor_state
                )

                # Compute the normalization
                normalization = torch.mean(
                    torch.sqrt((torch.sum(undecor_state**2, axis=1)))
                    / (torch.sqrt((torch.sum(decor_state**2, axis=1)) + 1e-8))
                )

                # Update the decorrelation matrix
                self.decor_weight = self.decor_weight * normalization
                decor_grad = corr @ self.decor_weight
                self.decor_weight = self.decor_weight - self.lr * decor_grad

        return output


class Decorrelator2D(torch.nn.Module):
    def __init__(
        self,
        num_features,
        kernel_size,
        stride,
        padding,
        dilation,
        lr=1e-5,
        mean_momentum=0.1,
        perc_samples=0.1,
        **kwargs
    ):
        super(Decorrelator2D, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.mean_momentum = mean_momentum
        self.lr = lr
        self.perc_samples = perc_samples

        # Register buffer is used for variables that are not updated during backprop
        self.mid_dim = num_features * kernel_size * kernel_size
        self.register_buffer(
            "decor_weight",
            torch.zeros(
                self.mid_dim,
                num_features,
                kernel_size,
                kernel_size,
            ),
        )
        self.register_buffer("running_mean", torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.decor_weight = torch.eye(self.mid_dim).reshape(
            self.mid_dim, self.num_features, self.kernel_size, self.kernel_size
        )
        self.running_mean.zero_()

    def demean(self, input):
        return Decorrelator.demean(self, input)

    def forward(self, input):
        assert self.num_features == input.shape[1], "Input shape mismatch"

        # Demean
        demeaned_input = self.demean(input)

        # Decorrelation
        output = F.conv2d(
            demeaned_input,
            self.decor_weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # If we are in a training pass, update decorrelation
        if self.training:
            with torch.no_grad():
                # Sub-sample the data
                num_samples = int(0.1 * len(input)) + 1
                undecor_state = input[:num_samples]
                decor_state = output[:num_samples]

                # Patchifying the data
                undecor_state = torch.nn.functional.unfold(
                    undecor_state,
                    self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
                output_size = int(
                    (demeaned_input.shape[-1] - self.kernel_size + 2 * self.padding)
                    / self.stride
                    + 1
                )
                undecor_state = torch.nn.functional.fold(
                    undecor_state,
                    output_size=(output_size, output_size),
                    kernel_size=(1, 1),
                )

                # Compute the normalization
                normalization = torch.mean(
                    torch.sqrt((torch.sum(undecor_state**2, axis=1)))
                    / (torch.sqrt((torch.sum(decor_state**2, axis=1)) + 1e-8))
                )

                # Set up patches as if they are batch-samples
                mod_decor_state = (
                    torch.permute(
                        decor_state.reshape(num_samples, self.mid_dim, -1),
                        (0, 2, 1),
                    )
                    .contiguous()
                    .reshape(-1, self.mid_dim)
                )

                # Compute the correlation matrix
                corr = (1 / len(mod_decor_state)) * (
                    mod_decor_state.T @ mod_decor_state
                )

                # Update the decorrelation matrix
                self.decor_weight = self.decor_weight * normalization
                decor_grad = corr @ self.decor_weight.reshape(
                    self.mid_dim, self.mid_dim
                )

                # Assign the update
                self.decor_weight = self.decor_weight - self.lr * decor_grad.reshape(
                    self.mid_dim,
                    self.num_features,
                    self.kernel_size,
                    self.kernel_size,
                )

        return output


class DecorLinear(torch.nn.Module):
    def __init__(
        self, layer_type, in_features, out_features, bias=True, decor_lr=1e-5, **kwargs
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
        decor_input = self.decor(input)
        return self.linear(decor_input)


class DecorConv2d(torch.nn.Module):
    def __init__(
        self,
        layer_type,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        decor_lr=1e-5,
        **kwargs
    ) -> None:
        super(DecorConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mid_dim = in_channels * kernel_size * kernel_size

        self.decor = Decorrelator2D(
            num_features=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            decor_lr=decor_lr,
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

    def __str__(self):
        return "DecorConv2d"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        decor_input = self.decor(input)
        return self.conv(decor_input)
