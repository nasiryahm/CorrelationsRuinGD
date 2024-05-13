import torch


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
        torch.nn.init.kaiming_uniform_(self.backward)

    def __str__(self):
        return "FALinear"

    def forward(self, input):
        return FAFunction.apply(input, self.weight, self.backward, self.bias)

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []
