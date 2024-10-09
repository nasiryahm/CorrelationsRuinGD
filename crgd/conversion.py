import torch
from torch.nn import Module
from .decor import DecorLinear, DecorConv2d


def analyze_model(model: Module, indent=0) -> None:
    """
    Helper function to analyze a model.
    Analyzes the model by printing the type of each layer as well as all their parameter shapes.

    :param model: the model to analyze
    :param indent: the indentation level

    :return: None
    """

    prefix = "  " * indent

    for child in model.children():
        print(prefix, "class name:", child.__class__.__name__)

        # check if child is a module
        if isinstance(child, Module):
            # check if the child has children
            if len(list(child.children())) > 0:
                analyze_model(child, indent + 1)
            else:
                # check all the parameters of the child
                for name, param in child.named_parameters():
                    print(prefix, "  parameter", name + ":", param.shape)


def convert_linear_layer(
    layer: torch.nn.Linear, decor_lr: float = 1e-5, **kwargs
) -> Module:
    """
    Converts a linear layer to the decorrelated equivalent.

    :param layer: the linear layer to convert
    :param decor_lr: the decorrelation learning rate
    :param kwargs: additional arguments

    :return: the converted layer
    """

    assert isinstance(layer, torch.nn.Linear), "Layer must be a Linear layer"

    # get the arguments for the new layer
    layer_kwargs = {
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "bias": layer.bias is not None,
    }
    layer_kwargs.update(kwargs)

    # create the new layer
    new_layer = DecorLinear(
        layer_type=torch.nn.Linear,
        decor_lr=decor_lr,
        **layer_kwargs
    ).to(layer.weight.device)

    # copy the weights and biases
    new_layer.linear.weight.data = layer.weight.data
    if layer.bias is not None:
        new_layer.linear.bias.data = layer.bias.data

    return new_layer


def convert_conv2d_layer(
    layer: torch.nn.Conv2d, decor_lr: float = 1e-5, **kwargs
) -> Module:
    """
    Converts a Conv2d layer to the decorrelated equivalent.

    :param layer: the Conv2d layer to convert
    :param decor_lr: the decorrelation learning rate
    :param kwargs: additional arguments

    :return: the converted layer
    """

    assert isinstance(layer, torch.nn.Conv2d), "Layer must be a Conv2d layer"

    # get the arguments for the new layer
    layer_kwargs = {
        "in_channels": layer.in_channels,
        "out_channels": layer.out_channels,
        "kernel_size": layer.kernel_size,
        "stride": layer.stride,
        "padding": layer.padding,
        "dilation": layer.dilation,
        "groups": layer.groups,
        "bias": layer.bias is not None,
    }
    layer_kwargs.update(kwargs)

    # create the new layer
    new_layer = DecorConv2d(
        layer_type=torch.nn.Conv2d,
        decor_lr=decor_lr,
        **layer_kwargs
    ).to(layer.weight.device)

    # copy the weights and biases
    new_layer.conv.weight.data = layer.weight.data.reshape(
        new_layer.conv.weight.data.shape
    )
    if layer.bias is not None:
        new_layer.conv.bias.data = layer.bias.data

    return new_layer


def convert_model(model: Module, decor_lr: float = 1e-5, **kwargs) -> Module:
    """
    Converts a model to the decorrelated equivalent.

    :param model: the model to convert
    :param decor_lr: the decorrelation learning rate
    :param kwargs: additional arguments

    :return: the converted model
    """

    new_model = model

    for name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            new_layer = convert_linear_layer(child, decor_lr=decor_lr, **kwargs)
        elif isinstance(child, torch.nn.Conv2d):
            new_layer = convert_conv2d_layer(child, decor_lr=decor_lr, **kwargs)
        else:
            new_layer = convert_model(child, decor_lr=decor_lr, **kwargs)

        setattr(new_model, name, new_layer)

    return new_model
