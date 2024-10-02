from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
import os
import numpy as np
from utils import *
from models import DenseNet, ConvNet
from bp import BPLinear, BPConv2d
from fa import FALinear, FAConv2d
from np import NPLinear, NPConv2d
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig
import random


def train_network(
    batch_size=256,
    dataset="MNIST",
    device="cuda",
    bias=True,
    regularizer_strength=0.0,
    decor_lr=1e-3,
    network_type=DenseNet,
    layer_type=BPLinear,
    loss_func_type="CCE",  # "MSE"
    activation_function=torch.nn.LeakyReLU(),
    optimizer_type="Adam",
    fwd_lr=1e-2,
    seed=42,
    nb_epochs=10,
    loud=True,
    wandb=None,
    validation=False,
):

    betas = [0.9, 0.9999]
    eps = 1e-8

    # Initializing random seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset
    tv_dataset = dataset
    if dataset == "MNIST":
        tv_dataset = torchvision.datasets.MNIST
    elif dataset == "CIFAR10":
        tv_dataset = torchvision.datasets.CIFAR10
    elif dataset == "CIFAR100":
        tv_dataset = torchvision.datasets.CIFAR100

    train_loader, test_loader = construct_dataloaders(
        tv_dataset, batch_size=batch_size, validation=validation, device=device
    )

    # If dataset is CIFAR, change input shape
    in_size = 28 * 28
    out_size = 10
    if tv_dataset == torchvision.datasets.CIFAR10:
        in_size = 32 * 32 * 3
    if tv_dataset == torchvision.datasets.CIFAR100:
        in_size = 32 * 32 * 3
        out_size = 100
    if tv_dataset == "TIN":
        # After cropping
        in_size = 56 * 56 * 3
        out_size = 200

    # Initialize model
    layer_kwargs = {}

    if layer_type in [NPLinear, NPConv2d]:
        distribution = torch.distributions.Normal(
            torch.tensor([0.0]).to(torch.float32).to(device),
            torch.tensor([1.0]).to(device),
        )
        dist_sampler = lambda x: distribution.sample([1] + x).squeeze_(-1).sum(0)
        layer_kwargs = {
            "sigma": 1e-6,
            "dist_sampler": dist_sampler,
        }
    if layer_type in [BPConv2d, FAConv2d, NPConv2d]:
        # TIN After cropping
        in_size = [3, 56, 56]
        if dataset in ["CIFAR10", "CIFAR100"]:
            in_size = [3, 32, 32]
        if dataset == "MNIST":
            in_size = [1, 28, 28]

    model = network_type(
        in_size=in_size,
        out_size=out_size,
        layer_type=layer_type,
        decor_lr=decor_lr,
        biases=bias,
        activation_function=activation_function,
        layer_kwargs=layer_kwargs,
    )
    model.to(device)

    # Initialize metric storage
    metrics = init_metric(validation=validation)

    # Define optimizer
    optimizer = None
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            betas=betas,
            eps=eps,
            lr=fwd_lr,
            weight_decay=regularizer_strength,
        )
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=fwd_lr, weight_decay=regularizer_strength
        )

    loss_func = None
    if loss_func_type == "CCE":
        # loss_obj = torch.nn.NLLLoss(reduction="none")
        loss_obj = torch.nn.CrossEntropyLoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, target)
    elif loss_func_type == "MSE":
        loss_obj = torch.nn.MSELoss(reduction="none")
        loss_func = lambda input, target, onehot: torch.sum(
            loss_obj(input, onehot), axis=1
        )

    # Train loop
    test_val = "test"
    if validation:
        test_val = "val"
    for e in tqdm(range(nb_epochs + 1), disable=not loud):
        metrics = update_metrics(
            model,
            metrics,
            device,
            "train",
            train_loader,
            loss_func,
            e,
            loud=loud,
            wandb=wandb,
            top5=(dataset == "TIN"),
            num_classes=out_size,
        )
        metrics = update_metrics(
            model,
            metrics,
            device,
            test_val,
            test_loader,
            loss_func,
            e,
            loud=loud,
            wandb=wandb,
            top5=(dataset == "TIN"),
            num_classes=out_size,
        )
        if e < nb_epochs:
            train(
                model,
                device,
                train_loader,
                optimizer,
                e,
                loss_func,
                loud=False,
                num_classes=out_size,
            )
        if np.isnan(metrics[test_val]["loss"][-1]) or np.isnan(
            metrics["train"]["loss"][-1]
        ):
            print("NaN detected, aborting training")
            break
    return metrics


@hydra.main(version_base="1.3", config_path="conf/", config_name="config")
def run(config: DictConfig) -> None:
    torch.set_num_threads(2)

    cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.init(
        config=cfg,
        entity=config.wandb.entity,
        project=config.wandb.project,
        mode=config.wandb.mode,
    )

    layer_type = BPLinear
    assert config.layer_type in [
        "BP",
        "FA",
        "DFA",
        "NP",
        "BPConv",
        "FAConv",
        "DFAConv",
        "NPConv",
    ], "Invalid layer type"
    if config.layer_type in ["FA", "DFA"]:
        layer_type = FALinear
    elif config.layer_type == "NP":
        layer_type = NPLinear
    elif config.layer_type == "BPConv":
        layer_type = BPConv2d
    elif config.layer_type in ["FAConv", "DFAConv"]:
        layer_type = FAConv2d
    elif config.layer_type == "NPConv":
        layer_type = NPConv2d

    network_type = DenseNet
    if layer_type in [BPConv2d, FAConv2d, NPConv2d]:
        network_type = ConvNet

    act_func = torch.nn.LeakyReLU
    if config.layer_type == "DFA" or config.layer_type == "DFAConv":
        act_func = ST_LeakyReLU

    metrics = train_network(
        batch_size=config.batch_size,
        dataset=config.dataset,
        device=config.device,
        bias=config.bias,
        regularizer_strength=config.regularizer_strength,
        decor_lr=config.decor_lr,
        network_type=network_type,
        layer_type=layer_type,
        loss_func_type=config.loss_func_type,
        activation_function=act_func,
        optimizer_type=config.optimizer_type,
        fwd_lr=config.fwd_lr,
        seed=config.seed,
        nb_epochs=config.nb_epochs,
        loud=config.loud,
        wandb=wandb,
        validation=config.validation,
    )

    print(metrics)


if __name__ == "__main__":
    run()
