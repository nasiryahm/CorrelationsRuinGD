import torch


class BPLinear(torch.nn.Linear):
    """BP Linear layer"""

    def __str__(self):
        return "BPLinear"


class BPConv2d(torch.nn.Conv2d):
    """BP Conv2d layer"""

    def __str__(self):
        return "BPConv2d"
