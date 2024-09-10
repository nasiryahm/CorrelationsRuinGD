import torch


class NPLinearFunc(torch.autograd.Function):
    """Linear layer with noise injection at nodes"""

    @staticmethod
    def forward(ctx, input, weights, biases, sigma, dist_sampler):
        # Matrix multiplying both the clean and noisy forward signals
        output = torch.mm(input, weights.t())

        if biases is not None:
            output += biases

        # Determining the shape of the noise
        noise_shape = [s for s in output.shape]
        noise_shape[0] = noise_shape[0] // 2  # int division

        # Only simulating clean passes
        noise_1 = torch.zeros(noise_shape).to(input.device)
        noise_2 = sigma * dist_sampler(noise_shape).to(input.device)

        # Generating the noise
        noise = torch.concat([noise_1, noise_2])

        # Adding the noise to the output
        output += noise

        # compute the output
        return output, noise

    @staticmethod
    def backward(ctx, grad_output, _):
        return None, None, None, None, None


class NPLinear(torch.nn.Linear):
    """Node Perturbation layer with saved noise"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float,
        dist_sampler: torch.distributions.Distribution,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.sigma = sigma
        self.dist_sampler = dist_sampler

    def __str__(self):
        return "NPLinear"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        output, noise = NPLinearFunc().apply(
            input,
            self.weight,
            self.bias,
            self.sigma,
            self.dist_sampler,
        )
        half_batch_width = len(input) // 2

        self.clean_input = input[:half_batch_width]
        self.output_diff = -noise[half_batch_width:]

        return output

    def update_grads(self, scaling_factor) -> None:
        with torch.no_grad():
            # Rescale grad data - to be used at end of gradient pass
            scaled_out_diff = (
                scaling_factor[:, None] * self.output_diff / (self.sigma**2)
            )
            self.weight.grad = torch.einsum(
                "ni,nj->ij", scaled_out_diff, self.clean_input
            )
            if self.bias is not None:
                self.bias.grad = scaled_out_diff.sum(0)

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []
