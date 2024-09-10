import torch


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

    def update_grads(self, _):
        pass


class BPConv2d(torch.nn.Conv2d):
    """BP Conv2d layer"""

    def __str__(self):
        return "BPConv2d"

    def get_fwd_params(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def get_decor_params(self):
        return []

    def update_grads(self, _):
        pass
