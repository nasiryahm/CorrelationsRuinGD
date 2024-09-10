import torch
import torch.nn.functional as F


class FAFunction(torch.autograd.Function):

    @staticmethod
    def forward(context, input, weight, backward, bias=None):
        context.save_for_backward(input, weight, backward, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, backward, bias = context.saved_tensors
        grad_input = grad_weight = grad_backward = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(backward.to(grad_output.device))
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_backward, grad_bias


class FALinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FALinear, self).__init__(in_features, out_features, bias)

        self.backward = torch.nn.Parameter(
            torch.FloatTensor(out_features, in_features), requires_grad=False
        )
        self.initialized = False

    def __str__(self):
        return "FALinear"

    def initialize_backward(self):
        if not self.initialized:
            torch.nn.init.kaiming_uniform_(self.backward)
            self.initialized = True

    def forward(self, input):
        self.initialize_backward()
        input = input.view(input.size(0), -1)
        return FAFunction.apply(input, self.weight, self.backward, self.bias)

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []

    def update_grads(self, _):
        pass


class FAConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        backward,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input, weight, backward, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, backward, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, backward, grad_output, stride, padding, dilation, groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, stride, padding, dilation, groups
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class FAConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(FAConv2d, self).__init__(*args, **kwargs)

        self.backward = torch.nn.Parameter(
            torch.FloatTensor(*self.weight.shape), requires_grad=False
        )
        self.initialized = False

    def __str__(self):
        return "FALinear"

    def initialize_backward(self):
        if not self.initialized:
            torch.nn.init.kaiming_uniform_(self.backward)
            self.initialized = True

    def forward(self, input):
        self.initialize_backward()
        return FAConv2dFunction.apply(
            input, self.weight, self.backward, self.bias, self.stride, self.padding
        )

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []

    def update_grads(self, _):
        pass
